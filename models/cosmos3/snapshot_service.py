"""Reference: GPU-memory-snapshot service for Cosmos 3 (the 10x cold-start path).

Why this exists separately from ``modal_app.py``:

``modal_app.py`` runs inference through a Daft class UDF (``@daft.cls``). Daft
instantiates a GPU UDF in *its own worker process*, spun up inside
``run_on_modal``'s body — i.e. *after* the container's snapshot checkpoint. So a
GPU snapshot taken in ``@modal.enter(snap=True)`` cannot capture a model that the
Daft UDF loads. To get the snapshot win you must load the engine in the snapshot
window of a ``modal.Cls``, which is what this file does.

What a snapshot can and can't skip (per Modal's guide): it checkpoints the
container *after* the model is resident, so a cold start restores past CUDA init,
library imports, JIT/torch.compile, and weight load. The first-ever call still
pays that cost (the snapshot is built once); every cold start after reuses it.

vLLM caveat: vLLM-Omni runs a multiprocess engine. Modal's SGLang/vLLM snapshot
examples show multiprocess engines often need explicit GPU-memory release before
the checkpoint and resume after restore (``/release_memory_occupation`` etc.).
This reference loads the engine straight in ``@modal.enter(snap=True)``; if the
restore path misbehaves, that release/resume handshake is the next lever. Treat
this as the experiment that produces the real number, not a settled recipe.

Run an A/B cold-start measurement::

    uv run --extra models modal run models/cosmos3/snapshot_service.py::measure
"""

import time

import modal

from models.cosmos3.modal_app import image as cosmos3_image
from models.cosmos3.model import DEFAULT_MODEL, DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT
from models.weights import MODEL_CACHE, OUTPUTS, cls_kwargs

GPU_TYPE = "A100-80GB"

app = modal.App("daft-cosmos3-snapshot")

# Reuse the exact modal_app image and add only the snapshot-tuning env as one
# extra layer, so deploys hit the cached vLLM build instead of rebuilding it.
# TORCHINDUCTOR_COMPILE_THREADS=1 improves torch-compile/snapshot compatibility
# (Modal GPU-snapshot guide).
image = cosmos3_image.env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})


def _load_engine(model: str, model_revision: str, guardrails: bool):
    """Mirror of Cosmos3Omni.__init__ — kept here so the engine loads in the
    snapshot window rather than inside a Daft worker process."""
    from huggingface_hub import snapshot_download
    from vllm_omni.entrypoints.omni import Omni

    model_path = snapshot_download(repo_id=model, revision=model_revision or None)
    omni = Omni(
        model=model_path,
        model_class_name="Cosmos3OmniDiffusersPipeline",
        trust_remote_code=True,
        enforce_eager=True,
        model_config={"guardrails": guardrails},
    )
    return omni, model_path


@app.cls(**cls_kwargs(image, gpu=GPU_TYPE, memory=98304, with_outputs=True, scaledown_window=300))
class Cosmos3Snapshot:
    model: str = modal.parameter(default=DEFAULT_MODEL)
    model_revision: str = modal.parameter(default="")

    @modal.enter(snap=True)
    def load(self):
        """Loads the vLLM-Omni engine to GPU during the snapshot window so the
        whole resident state (CUDA ctx + weights + JIT) is captured."""
        t0 = time.perf_counter()
        self.omni, self.model_path = _load_engine(self.model, self.model_revision, guardrails=False)
        self.load_seconds = time.perf_counter() - t0
        print(f"[snap] engine loaded in {self.load_seconds:.1f}s")

    @modal.method()
    def generate(
        self,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        modality: str = "image",
        width: int = 1024,
        height: int = 1024,
        steps: int = 10,
        guidance_scale: float = 7.0,
        flow_shift: float = 10.0,
        seed: int = 42,
    ) -> dict:
        import torch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        t0 = time.perf_counter()
        sampling_params = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_outputs_per_prompt=1,
            num_frames=1,
            fps=None,
            extra_args={"flow_shift": flow_shift},
        )
        output = self.omni.generate(
            {"prompt": prompt, "negative_prompt": negative_prompt, "modalities": [modality]},
            sampling_params,
            use_tqdm=False,
        )[0]
        _ = output.request_output.images[0]
        return {"generate_seconds": time.perf_counter() - t0, "model_path": self.model_path}


@app.local_entrypoint()
def measure(prompt: str = DEFAULT_PROMPT):
    """Cold-start A/B: call once to build the snapshot, then again cold to read
    the restore-accelerated startup. Wall time around the first call of a fresh
    container is the cold start; compare snapshot-on vs the plain function path.
    """
    cosmos = Cosmos3Snapshot()
    t0 = time.perf_counter()
    first = cosmos.generate.remote(prompt=prompt)
    print(f"first call (includes build + snapshot warm): {time.perf_counter() - t0:.1f}s wall")
    print(f"  inner generate: {first['generate_seconds']:.1f}s")

    t1 = time.perf_counter()
    second = cosmos.generate.remote(prompt=prompt)
    print(f"second call (warm container): {time.perf_counter() - t1:.1f}s wall")
    print(f"  inner generate: {second['generate_seconds']:.1f}s")

    MODEL_CACHE.commit()
    OUTPUTS.commit()
    print("\nNote: to measure true *cold* restore, redeploy and invoke a fresh container")
    print("(`modal run` reuses the same container for both calls above).")
