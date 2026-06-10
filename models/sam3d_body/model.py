# /// script
# description = "Meta SAM 3D Body PyTorch inference as a Daft cls UDF (local CLI needs Python 3.11 + the sam-3d-body repo)"
# requires-python = ">=3.11, <3.12"
# dependencies = [
#   "appdirs",
#   "braceexpand",
#   "cython",
#   "daft>=0.7.10",
#   "dill",
#   "einops",
#   "fvcore",
#   "huggingface_hub",
#   "hydra-colorlog",
#   "hydra-core",
#   "hydra-submitit-launcher",
#   "jsonlines",
#   "loguru",
#   "networkx==3.2.1",
#   "opencv-python-headless",
#   "optree",
#   "pandas",
#   "pycocotools",
#   "pyrootutils",
#   "pyrender",
#   "pytorch-lightning",
#   "rich",
#   "roma",
#   "scikit-image",
#   "seaborn",
#   "tensorboard",
#   "timm",
#   "torch",
#   "torchvision",
#   "trimesh",
#   "wandb",
#   "webdataset",
#   "xtcocotools",
#   "yacs",
# ]
# ///
"""Meta SAM 3D Body as a Daft class UDF.

Backend: PyTorch (the upstream ``sam-3d-body`` estimator; vLLM does not support
this model). The ``@daft.cls`` instance loads the model once per worker, writes
rendered previews and mesh artifacts to disk, and returns a lightweight
metadata struct.

This module never imports ``modal`` — see ``modal_app.py`` for deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Anchor the repo root so `models.*` imports resolve when this file is loaded
# as a loose script (e.g. `uv run` / `modal run`) instead of an installed package.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import DataType, Series, col
from daft.functions import unnest
from models.common.modal_infra import MODEL_CACHE_DIR
from models.common.weights import normalize_hf_token_env, resolve_hf_model_path

DEFAULT_MODEL = "facebook/sam-3d-body-vith"
DEFAULT_OUTPUT_DIR = ".context/sam3d-body-outputs"
DEFAULT_LOCAL_REPO = ".context/sam-3d-body"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

Sam3DBodyResult = DataType.struct(
    {
        "image_path": DataType.string(),
        "output_dir": DataType.string(),
        "render_path": DataType.string(),
        "metadata_path": DataType.string(),
        "mesh_paths": DataType.list(DataType.string()),
        "npz_paths": DataType.list(DataType.string()),
        "model": DataType.string(),
        "model_revision": DataType.string(),
        "model_path": DataType.string(),
        "detector_name": DataType.string(),
        "fov_name": DataType.string(),
        "inference_type": DataType.string(),
        "num_people": DataType.int64(),
        "bbox_thr": DataType.float64(),
        "use_mask": DataType.bool(),
    }
)


def add_sam3d_repo_to_path(repo_path: str) -> None:
    repo = Path(repo_path)
    if not repo.exists():
        raise FileNotFoundError(
            f"SAM 3D Body repo not found at {repo}. Clone https://github.com/facebookresearch/sam-3d-body.git "
            "or set SAM3D_BODY_REPO."
        )
    sys.path.insert(0, str(repo))


def parse_bbox_json(bbox_json: str):
    if not bbox_json:
        return None

    import numpy as np

    bboxes = np.asarray(json.loads(bbox_json), dtype=np.float32)
    return bboxes.reshape(-1, 4)


def image_digest(image_path: str) -> str:
    digest = hashlib.sha1()
    with open(image_path, "rb") as image:
        for chunk in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@daft.cls(gpus=1.0, max_concurrency=1)
class Sam3DBody:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        model_revision: str = "",
        output_dir: str = DEFAULT_OUTPUT_DIR,
        repo_path: str = DEFAULT_LOCAL_REPO,
        checkpoint_path: str = "",
        mhr_path: str = "",
        detector_name: str = "",
        segmentor_name: str = "",
        segmentor_path: str = "",
        fov_name: str = "",
        bbox_thr: float = 0.8,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        add_sam3d_repo_to_path(repo_path)
        hf_token = normalize_hf_token_env()

        import cv2
        import torch
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        self.model_name = model
        self.model_revision = model_revision
        self.checkpoint_path = checkpoint_path
        self.model_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector_name = detector_name
        self.fov_name = fov_name
        self.bbox_thr = bbox_thr
        self.use_mask = use_mask
        self.inference_type = inference_type
        self.cv2 = cv2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint_path:
            model_impl, model_cfg = load_sam_3d_body(
                checkpoint_path=checkpoint_path,
                device=device,
                mhr_path=mhr_path,
            )
        else:
            local_dir = resolve_hf_model_path(
                model,
                MODEL_CACHE_DIR,
                revision=model_revision or None,
                token=hf_token,
            )
            self.model_path = str(local_dir)
            model_impl, model_cfg = load_sam_3d_body(
                checkpoint_path=str(Path(local_dir) / "model.ckpt"),
                device=device,
                mhr_path=str(Path(local_dir) / "assets/mhr_model.pt"),
            )

        human_detector, human_segmentor, fov_estimator = None, None, None
        if detector_name:
            from tools.build_detector import HumanDetector

            human_detector = HumanDetector(name=detector_name, device=device)
        if segmentor_name and segmentor_path:
            from tools.build_sam import HumanSegmentor

            human_segmentor = HumanSegmentor(name=segmentor_name, device=device, path=segmentor_path)
        if fov_name:
            from tools.build_fov_estimator import FOVEstimator

            fov_estimator = FOVEstimator(name=fov_name, device=device)

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model_impl,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=human_segmentor,
            fov_estimator=fov_estimator,
        )

    def create_output_dir(self, image_path: str, bbox_json: str) -> Path:
        image = Path(image_path)
        digest = hashlib.sha1(
            (
                f"{self.model_name}:{self.model_revision}:{self.checkpoint_path}:"
                f"{self.detector_name}:{self.fov_name}:{self.inference_type}:"
                f"{self.bbox_thr}:{self.use_mask}:{bbox_json}:{image_digest(image_path)}"
            ).encode()
        ).hexdigest()[:12]
        return self.output_dir / f"sam3d-body-{image.stem}-{digest}"

    def process_one(self, image_path: str, bbox_json: str) -> dict:
        import numpy as np
        from sam_3d_body.visualization.renderer import Renderer

        sample_dir = self.create_output_dir(image_path, bbox_json)
        sample_dir.mkdir(parents=True, exist_ok=True)

        render_path = sample_dir / "render.jpg"
        metadata_path = sample_dir / "metadata.json"
        bboxes = parse_bbox_json(bbox_json)

        outputs = self.estimator.process_one_image(
            image_path,
            bboxes=bboxes,
            bbox_thr=self.bbox_thr,
            use_mask=self.use_mask,
            inference_type=self.inference_type,
        )

        img_bgr = self.cv2.imread(image_path)
        mesh_paths = []
        npz_paths = []
        people = []

        for person_id, person_output in enumerate(outputs):
            mesh_path = sample_dir / f"person_{person_id:03d}.ply"
            npz_path = sample_dir / f"person_{person_id:03d}.npz"

            renderer = Renderer(focal_length=person_output["focal_length"], faces=self.estimator.faces)
            mesh = renderer.vertices_to_trimesh(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                LIGHT_BLUE,
            )
            mesh.export(mesh_path)

            np.savez_compressed(
                npz_path,
                bbox=person_output["bbox"],
                focal_length=person_output["focal_length"],
                pred_vertices=person_output["pred_vertices"],
                pred_cam_t=person_output["pred_cam_t"],
                pred_keypoints_2d=person_output["pred_keypoints_2d"],
                pred_keypoints_3d=person_output["pred_keypoints_3d"],
                pred_joint_coords=person_output["pred_joint_coords"],
            )

            mesh_paths.append(str(mesh_path))
            npz_paths.append(str(npz_path))
            people.append(
                {
                    "person_id": person_id,
                    "bbox": person_output["bbox"].tolist(),
                    "focal_length": float(person_output["focal_length"]),
                    "mesh_path": str(mesh_path),
                    "npz_path": str(npz_path),
                }
            )

        if outputs:
            render = self.render_outputs(img_bgr, outputs, Renderer)
            self.cv2.imwrite(str(render_path), render.astype(np.uint8))
        else:
            self.cv2.imwrite(str(render_path), img_bgr)

        write_json(
            metadata_path,
            {
                "image_path": image_path,
                "model": self.model_name,
                "model_revision": self.model_revision,
                "model_path": self.model_path,
                "detector_name": self.detector_name,
                "fov_name": self.fov_name,
                "inference_type": self.inference_type,
                "num_people": len(outputs),
                "people": people,
            },
        )

        return {
            "image_path": image_path,
            "output_dir": str(sample_dir),
            "render_path": str(render_path),
            "metadata_path": str(metadata_path),
            "mesh_paths": mesh_paths,
            "npz_paths": npz_paths,
            "model": self.model_name,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "detector_name": self.detector_name,
            "fov_name": self.fov_name,
            "inference_type": self.inference_type,
            "num_people": len(outputs),
            "bbox_thr": self.bbox_thr,
            "use_mask": self.use_mask,
        }

    @daft.method.batch(return_dtype=Sam3DBodyResult, batch_size=1)
    def process(self, image_paths: Series, bbox_jsons: Series) -> list[dict]:
        return [
            self.process_one(str(image_path), str(bbox_json or ""))
            for image_path, bbox_json in zip(image_paths.to_pylist(), bbox_jsons.to_pylist())
        ]

    def render_outputs(self, img_bgr, outputs: list[dict], renderer_cls):
        import numpy as np

        outputs_sorted = sorted(outputs, key=lambda person: person["pred_cam_t"][2], reverse=True)
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        for person_output in outputs_sorted:
            vertices = person_output["pred_vertices"] + person_output["pred_cam_t"]
            all_vertices.append(vertices)
            all_faces.append(self.estimator.faces + vertex_offset)
            vertex_offset += len(vertices)

        all_vertices = np.concatenate(all_vertices, axis=0)
        all_faces = np.concatenate(all_faces, axis=0)
        fake_cam_t = (np.max(all_vertices, axis=0) + np.min(all_vertices, axis=0)) / 2
        all_vertices = all_vertices - fake_cam_t

        renderer = renderer_cls(focal_length=outputs_sorted[-1]["focal_length"], faces=all_faces)
        mesh_overlay = (
            renderer(
                all_vertices,
                fake_cam_t,
                img_bgr.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )
        side_view = (
            renderer(
                all_vertices,
                fake_cam_t,
                np.ones_like(img_bgr) * 255,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )
        return np.concatenate([img_bgr, mesh_overlay, side_view], axis=1)


def build_dataframe(
    image_paths: list[str],
    *,
    bbox_jsons: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    repo_path: str = DEFAULT_LOCAL_REPO,
    checkpoint_path: str = "",
    mhr_path: str = "",
    detector_name: str = "",
    segmentor_name: str = "",
    segmentor_path: str = "",
    fov_name: str = "",
    bbox_thr: float = 0.8,
    use_mask: bool = False,
    inference_type: str = "full",
) -> daft.DataFrame:
    sam3d = Sam3DBody(
        model=model,
        model_revision=model_revision,
        output_dir=output_dir,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=detector_name,
        segmentor_name=segmentor_name,
        segmentor_path=segmentor_path,
        fov_name=fov_name,
        bbox_thr=bbox_thr,
        use_mask=use_mask,
        inference_type=inference_type,
    )
    bbox_jsons = bbox_jsons or [""] * len(image_paths)

    return (
        daft.from_pydict({"image_path": image_paths, "bbox_json": bbox_jsons})
        .with_column("result", sam3d.process(col("image_path"), col("bbox_json")))
        .select(unnest(col("result")))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", action="append", dest="image_paths")
    parser.add_argument("--bbox", action="append", dest="bboxes", help='JSON bbox prompt, e.g. "[0,0,512,768]"')
    parser.add_argument("--model", default=os.environ.get("SAM3D_BODY_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-revision", default=os.environ.get("SAM3D_BODY_MODEL_REVISION", ""))
    parser.add_argument("--checkpoint-path", default=os.environ.get("SAM3D_BODY_CHECKPOINT_PATH", ""))
    parser.add_argument("--mhr-path", default=os.environ.get("SAM3D_BODY_MHR_PATH", ""))
    parser.add_argument("--output-dir", default=os.environ.get("SAM3D_BODY_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repo-path", default=os.environ.get("SAM3D_BODY_REPO", DEFAULT_LOCAL_REPO))
    parser.add_argument("--detector-name", default="")
    parser.add_argument("--fov-name", default="")
    parser.add_argument("--bbox-thr", type=float, default=0.8)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--inference-type", choices=["full", "body", "hand"], default="full")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.download_only:
        model_path = resolve_hf_model_path(
            args.model,
            MODEL_CACHE_DIR,
            revision=args.model_revision or None,
            token=normalize_hf_token_env(),
        )
        print({"model": args.model, "model_revision": args.model_revision, "model_path": str(model_path)})
        raise SystemExit(0)

    default_image = str(Path(args.repo_path) / "notebook/images/dancing.jpg")
    image_paths = args.image_paths or [default_image]
    df = build_dataframe(
        image_paths,
        bbox_jsons=args.bboxes,
        model=args.model,
        model_revision=args.model_revision,
        output_dir=args.output_dir,
        repo_path=args.repo_path,
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        detector_name=args.detector_name,
        fov_name=args.fov_name,
        bbox_thr=args.bbox_thr,
        use_mask=args.use_mask,
        inference_type=args.inference_type,
    ).collect()
    df.show(format="fancy", max_width=100)
