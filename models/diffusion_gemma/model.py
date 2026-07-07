"""Google DiffusionGemma text generation as a Daft class UDF.

Backend: offline vLLM (`LLM` engine via the block-diffusion data path added in
vllm-project/vllm#45163). DiffusionGemma iteratively denoises fixed 256-token
canvases in parallel instead of decoding left-to-right, so the engine needs a
handful of diffusion-specific flags — see README.md for what each one does and
why `max_num_seqs` must stay low.

The ``@daft.cls`` instance loads the engine once per worker and returns the
generated text plus generation metadata.

This module never imports ``modal`` — see ``modal_app.py`` for deployment.
"""

from __future__ import annotations

import argparse

import daft
from daft import DataType, col
from daft.ai.metrics import record_token_metrics
from daft.functions import unnest

DEFAULT_MODEL = "google/diffusiongemma-26B-A4B-it"
DEFAULT_PROMPT = "Why is the sky blue?"

# Recipe-verified engine defaults (vllm-project/recipes#520, single 80 GB GPU).
DEFAULT_CANVAS_LENGTH = 256
DEFAULT_ENTROPY_BOUND = 0.1
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85

DiffusionGemmaResult = DataType.struct(
    {
        "text": DataType.string(),
        "prompt": DataType.string(),
        "model": DataType.string(),
        "model_revision": DataType.string(),
        "model_path": DataType.string(),
        "canvas_length": DataType.int64(),
        "entropy_bound": DataType.float64(),
        "max_tokens": DataType.int64(),
        "temperature": DataType.float64(),
        "seed": DataType.int64(),
        "enable_thinking": DataType.bool(),
        "num_prompt_tokens": DataType.int64(),
        "num_generated_tokens": DataType.int64(),
        "num_cached_tokens": DataType.int64(),
        "finish_reason": DataType.string(),
        "generation_time_s": DataType.float64(),
    }
)


@daft.cls(gpus=1.0, max_concurrency=1)
class DiffusionGemma:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        model_revision: str = "",
        canvas_length: int = DEFAULT_CANVAS_LENGTH,
        entropy_bound: float = DEFAULT_ENTROPY_BOUND,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    ):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model = model
        self.model_revision = model_revision
        self.canvas_length = canvas_length
        self.entropy_bound = entropy_bound
        self.model_path = snapshot_download(repo_id=model, revision=model_revision or None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llm = LLM(
            model=self.model_path,
            max_model_len=max_model_len,
            # Diffusion state buffers pre-allocate max_seqs x canvas_length x vocab_size
            # tensors; with Gemma's 262K vocab anything above ~4 OOMs an 80 GB GPU.
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            diffusion_config={"canvas_length": canvas_length},
            # The checkpoint's generation_config.json caps max_tokens at one canvas
            # (256); "vllm" ignores it so per-request limits win.
            generation_config="vllm",
            enable_chunked_prefill=True,
            hf_overrides={
                "diffusion_sampler": "entropy_bound",
                "diffusion_entropy_bound": entropy_bound,
            },
        )

    @daft.method(return_dtype=DiffusionGemmaResult)
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 42,
        enable_thinking: bool = False,
    ) -> dict:
        return self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            enable_thinking=enable_thinking,
        )

    @daft.method(return_dtype=DiffusionGemmaResult)
    def generate_from_image(
        self,
        prompt: str,
        image_file: daft.File,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 42,
        enable_thinking: bool = False,
    ) -> dict:
        from PIL import Image

        with image_file.to_tempfile() as tmp:
            image = Image.open(tmp.name).convert("RGB")
            image.load()

        return self._generate(
            prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            enable_thinking=enable_thinking,
        )

    def _generate(
        self,
        prompt: str,
        *,
        image=None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 42,
        enable_thinking: bool = False,
    ) -> dict:
        import time

        from vllm import SamplingParams

        content = [{"type": "text", "text": prompt}]
        if image is not None:
            # DiffusionGemma prefers image content before text for VQA-style prompts.
            content.insert(0, {"type": "image"})
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        llm_input = chat_prompt
        if image is not None:
            llm_input = {"prompt": chat_prompt, "multi_modal_data": {"image": image}}

        start = time.perf_counter()
        request_output = self.llm.generate(
            llm_input,
            SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed),
            use_tqdm=False,
        )[0]
        generation_time_s = time.perf_counter() - start
        output = request_output.outputs[0]
        input_tokens = len(request_output.prompt_token_ids or [])
        output_tokens = len(output.token_ids)
        record_token_metrics(
            protocol="prompt",
            model=self.model,
            provider="vllm",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return {
            "text": output.text,
            "prompt": prompt,
            "model": self.model,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "canvas_length": self.canvas_length,
            "entropy_bound": self.entropy_bound,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "enable_thinking": enable_thinking,
            "num_prompt_tokens": input_tokens,
            "num_generated_tokens": output_tokens,
            "num_cached_tokens": request_output.num_cached_tokens,
            "finish_reason": str(output.finish_reason),
            "generation_time_s": generation_time_s,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append", dest="prompts", default=[DEFAULT_PROMPT])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="")
    parser.add_argument(
        "--image-path", default="", help="Optional local or remote image path for multimodal prompting."
    )
    parser.add_argument("--canvas-length", type=int, default=DEFAULT_CANVAS_LENGTH)
    parser.add_argument("--entropy-bound", type=float, default=DEFAULT_ENTROPY_BOUND)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompts = args.prompts

    gemma = DiffusionGemma(
        model=args.model,
        model_revision=args.model_revision,
        canvas_length=args.canvas_length,
        entropy_bound=args.entropy_bound,
        max_model_len=args.max_model_len,
    )

    data = {
        "prompt": prompts,
        "seed": [args.seed + index for index in range(len(prompts))],
        "max_tokens": [args.max_tokens] * len(prompts),
        "temperature": [args.temperature] * len(prompts),
        "enable_thinking": [args.enable_thinking] * len(prompts),
    }
    if args.image_path:
        data["image_path"] = [args.image_path] * len(prompts)
    df = daft.from_pydict(data)
    if args.image_path:
        from daft.functions import file

        df = df.with_column("image_file", file(col("image_path"))).with_column(
            "result",
            gemma.generate_from_image(
                col("prompt"),
                col("image_file"),
                max_tokens=col("max_tokens"),
                temperature=col("temperature"),
                seed=col("seed"),
                enable_thinking=col("enable_thinking"),
            ),
        )
    else:
        df = df.with_column(
            "result",
            gemma.generate(
                col("prompt"),
                max_tokens=col("max_tokens"),
                temperature=col("temperature"),
                seed=col("seed"),
                enable_thinking=col("enable_thinking"),
            ),
        )
    df = df.select(unnest(col("result")))
    df = df.collect()

    df.show(format="fancy", max_width=100)
