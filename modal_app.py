import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import modal
import yaml


APP_NAME = "3d-re-gen"
LOCAL_REPO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path("/root/3D-RE-GEN")
DATA_ROOT = Path("/data")
CACHE_ROOT = Path("/cache")
ARTIFACT_ROOT = Path("/artifacts")
DEFAULT_GPU = os.environ.get("THREED_REGEN_MODAL_GPU", "L40S")

MAIN_STEPS = {1, 2, 3, 5, 6, 7, 9}

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("3d-regen-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("3d-regen-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("3d-regen-artifacts", create_if_missing=True)
job_registry = modal.Dict.from_name("3d-regen-job-registry", create_if_missing=True)

if modal.is_local():
    local_secret_env = {}
    if os.environ.get("GOOGLE_API_KEY"):
        local_secret_env["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        local_secret_env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    runtime_secret = modal.Secret.from_dict(local_secret_env)
else:
    runtime_secret = modal.Secret.from_dict({})

main_image = modal.Image.from_dockerfile(
    str(LOCAL_REPO_ROOT / "docker" / "Dockerfile.main"),
    context_dir=str(LOCAL_REPO_ROOT),
)
vggt_image = modal.Image.from_dockerfile(
    str(LOCAL_REPO_ROOT / "docker" / "Dockerfile.vggt"),
    context_dir=str(LOCAL_REPO_ROOT),
)
blender_image = modal.Image.from_dockerfile(
    str(LOCAL_REPO_ROOT / "docker" / "Dockerfile.blender"),
    context_dir=str(LOCAL_REPO_ROOT),
)


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


def _artifact_job_dir(job_id: str) -> Path:
    return ARTIFACT_ROOT / "jobs" / job_id


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_job_record(job_id: str, **fields) -> dict:
    record = job_registry.get(job_id, {}) or {}
    record.update(fields)
    job_registry.put(job_id, record)
    return record


def _ensure_runtime_env() -> None:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
    os.environ.setdefault("THREED_REGEN_USE_CURRENT_PYTHON", "1")

    for path in {
        CACHE_ROOT / "huggingface",
        CACHE_ROOT / "huggingface" / "hub",
        CACHE_ROOT / "transformers",
        CACHE_ROOT / "torch",
        CACHE_ROOT / "xdg",
    }:
        path.mkdir(parents=True, exist_ok=True)


def _resolve_hdri_path() -> str | None:
    preferred = REPO_ROOT / "input_images" / "raw" / "brown_photostudio_02_8k.hdr"
    fallback = REPO_ROOT / "input_images" / "raw" / "kloofendal_overcast_puresky_1k.hdr"
    if preferred.exists():
        return str(preferred)
    if fallback.exists():
        return str(fallback)
    return None


def _build_job_config(job_dir: Path, input_image_name: str, gt_scene_name: str | None = None) -> Path:
    base_config_path = REPO_ROOT / "src" / "config.yaml"
    config = yaml.safe_load(base_config_path.read_text())

    output_dir = job_dir / "output"
    tmp_dir = job_dir / "tmp"
    input_dir = job_dir / "input"
    pre_3d_dir = output_dir / "pre_3D"
    findings_dir = output_dir / "findings"
    banana_dir = findings_dir / "banana"
    pointcloud_dir = output_dir / "pointclouds"
    glb_scene_dir = output_dir / "glb" / "scene"

    for path in (
        job_dir,
        input_dir,
        tmp_dir,
        output_dir,
        pre_3d_dir,
        findings_dir,
        banana_dir,
        pointcloud_dir,
        pointcloud_dir / "scene",
        pointcloud_dir / "meshed",
        glb_scene_dir,
        output_dir / "rendering",
        output_dir / "evaluation",
        output_dir / "midi",
    ):
        path.mkdir(parents=True, exist_ok=True)

    config["config_path"] = str(job_dir / "config.yaml")
    config["input_image"] = str(input_dir / input_image_name)
    config["image_url"] = str(tmp_dir / "converted_from_webp.png")
    config["GT_scene"] = str(job_dir / "gt" / gt_scene_name) if gt_scene_name else None

    config["output"] = str(output_dir)
    config["temp"] = str(tmp_dir)
    config["output_seg"] = str(findings_dir)
    config["output_seg_banana"] = str(banana_dir)
    config["depth_scene"] = str(findings_dir / "depth.png")
    config["output_inp_banana"] = str(banana_dir / "inpaint_nanoBanana")
    config["prepped_for_hunyuan"] = str(banana_dir / "prepped")
    config["input_folder_hy"] = str(findings_dir / "upscaled" / "cropped")
    config["output_folder_hy"] = str(output_dir / "3D")

    config["tmp_dir"] = str(pre_3d_dir)
    config["camera"] = str(pre_3d_dir / "camera.npz")
    config["vggt_cloud"] = str(pre_3d_dir / "scene_vggt.ply")
    config["output_vggt"] = str(output_dir / "vggt" / "sparse")

    config["mask_folder"] = str(output_dir / "masks")
    config["glb_output_folder"] = str(output_dir / "glb")
    config["full_size"] = str(findings_dir / "fullSize")
    config["output_ply"] = str(pointcloud_dir)
    config["point_cloud_folder"] = str(pointcloud_dir)
    config["glb_scene_path"] = str(glb_scene_dir / "combined_scene.glb")
    config["ply_scene_bp_path"] = str(pointcloud_dir / "scene" / "combined_scene_bp.ply")
    config["ply_pred_points"] = str(pointcloud_dir / "scene" / "pred_points.ply")
    config["ply_gt_points"] = str(pointcloud_dir / "scene" / "gt_points.ply")
    config["out_pc_meshed"] = str(pointcloud_dir / "meshed")

    config["output_render"] = str(output_dir / "rendering")
    config["predicted_image"] = str(output_dir / "rendering" / "render_cam1_white_bg.png")
    config["eval_output_dir"] = str(output_dir / "evaluation")
    config["glb_scene_path_midi"] = str(output_dir / "glb" / "scene" / "combined_scene_midi.glb")
    config["midi_output"] = str(output_dir / "midi")
    config["midi_tmp"] = str(tmp_dir / "midi")

    hdri_path = _resolve_hdri_path()
    if hdri_path:
        config["hdri_path"] = hdri_path

    if config.get("model_id") == "gemini-2.5-flash-image-preview":
        config["model_id"] = "gemini-2.5-flash-image"

    config["conda_env"] = None
    config["device"] = "cuda:0"
    config["device_global"] = "cuda:0"
    config["use_all_available_cuda"] = False
    config["jobs_per_gpu"] = 1
    config["interactive_edit"] = False
    config["disable_transformers_torch_load_safety_check"] = True

    config_path = job_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _run_pipeline_part(job_id: str, steps: list[int], blender_executable: str | None = None) -> dict:
    _ensure_runtime_env()
    job_dir = _job_dir(job_id)
    config_path = job_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Job config not found: {config_path}")
    if 2 in steps and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not set in the Modal runtime environment. "
            "Export one of them locally before running `modal run modal_app.py ...`."
        )

    env = os.environ.copy()
    if blender_executable:
        env["THREED_REGEN_BLENDER_EXECUTABLE"] = blender_executable

    cmd = ["python", "run.py", "--config", str(config_path), "-p", *map(str, steps)]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    data_volume.commit()
    record = job_registry.get(job_id, {}) or {}
    completed_steps = sorted(set(record.get("completed_steps", [])).union(steps))
    _update_job_record(
        job_id,
        completed_steps=completed_steps,
        last_run_at=_utcnow(),
        status="completed" if set(completed_steps) >= {1, 2, 3, 4, 5, 6, 7} else "running",
    )
    return {"job_id": job_id, "steps": steps}


@app.function(
    image=main_image,
    timeout=60 * 20,
    startup_timeout=60 * 10,
    volumes={"/data": data_volume, "/cache": cache_volume},
)
def prepare_job(
    input_image_name: str,
    input_image_bytes: bytes,
    gt_scene_name: str | None = None,
    gt_scene_bytes: bytes | None = None,
) -> dict:
    data_volume.reload()

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / input_image_name).write_bytes(input_image_bytes)

    if gt_scene_name and gt_scene_bytes is not None:
        gt_dir = job_dir / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / gt_scene_name).write_bytes(gt_scene_bytes)

    config_path = _build_job_config(job_dir, input_image_name, gt_scene_name)
    data_volume.commit()
    _update_job_record(
        job_id,
        created_at=_utcnow(),
        status="prepared",
        input_image_name=input_image_name,
        config_path=str(config_path),
        completed_steps=[],
    )
    return {"job_id": job_id, "config_path": str(config_path)}


@app.function(
    image=main_image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    startup_timeout=60 * 20,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[runtime_secret],
)
def run_main_steps(job_id: str, steps: list[int]) -> dict:
    unsupported = sorted(set(steps) - MAIN_STEPS)
    if unsupported:
        raise ValueError(f"run_main_steps does not support steps: {unsupported}")

    data_volume.reload()
    return _run_pipeline_part(job_id, steps)


@app.function(
    image=vggt_image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 2,
    startup_timeout=60 * 20,
    volumes={"/data": data_volume, "/cache": cache_volume},
)
def run_vggt_step(job_id: str) -> dict:
    data_volume.reload()
    return _run_pipeline_part(job_id, [4])


@app.function(
    image=blender_image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 2,
    startup_timeout=60 * 20,
    volumes={"/data": data_volume, "/cache": cache_volume},
)
def run_blender_render(job_id: str) -> dict:
    data_volume.reload()
    return _run_pipeline_part(job_id, [8], blender_executable="blender")


@app.function(
    image=main_image,
    timeout=60 * 60,
    startup_timeout=60 * 10,
    volumes={"/cache": cache_volume},
)
def warm_main_model_cache(include_optional_scene_models: bool = False) -> dict:
    _ensure_runtime_env()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
    from rembg import new_session

    config = yaml.safe_load((REPO_ROOT / "src" / "config.yaml").read_text())
    warmed: dict[str, list[str]] = {
        "transformers": [],
        "huggingface": [],
        "rembg": [],
    }

    detector_id = config["detector_id"]
    segmenter_id = config["segmenter_id"]
    pipeline("zero-shot-object-detection", model=detector_id, device=-1)
    warmed["transformers"].append(detector_id)

    AutoModelForMaskGeneration.from_pretrained(segmenter_id)
    AutoProcessor.from_pretrained(segmenter_id)
    warmed["transformers"].append(segmenter_id)

    for repo_id in ("tencent/Hunyuan3D-2", "tencent/Hunyuan3D-2mini"):
        snapshot_download(repo_id=repo_id)
        warmed["huggingface"].append(repo_id)

    if include_optional_scene_models:
        for repo_id in (
            "prs-eth/marigold-iid-appearance-v1-1",
            "prs-eth/marigold-normals-v1-1",
        ):
            snapshot_download(repo_id=repo_id)
            warmed["huggingface"].append(repo_id)

    for model_name in ("isnet-general-use", "u2netp", "u2net"):
        try:
            new_session(model_name, providers=["CPUExecutionProvider"])
            warmed["rembg"].append(model_name)
        except Exception:
            pass

    cache_volume.commit()
    return warmed


@app.function(
    image=vggt_image,
    timeout=60 * 30,
    startup_timeout=60 * 10,
    volumes={"/cache": cache_volume},
)
def warm_vggt_model_cache() -> dict:
    _ensure_runtime_env()

    import torch

    urls = {
        "vggt": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        "vggsfm_tracker": "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt",
    }
    model_dir = os.environ["TORCH_HOME"]

    for url in urls.values():
        torch.hub.load_state_dict_from_url(url, model_dir=model_dir, progress=True)

    cache_volume.commit()
    return urls


@app.function(
    image=main_image,
    timeout=60 * 20,
    startup_timeout=60 * 10,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
)
def publish_job_artifacts(
    job_id: str,
    *,
    include_rendering: bool = True,
    include_evaluation: bool = False,
    include_object_glbs: bool = False,
    include_full_output: bool = False,
) -> dict:
    data_volume.reload()
    artifact_volume.reload()

    job_dir = _job_dir(job_id)
    output_dir = job_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"Job output not found: {output_dir}")

    artifact_dir = _artifact_job_dir(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []

    if include_full_output:
        _copy_path(output_dir, artifact_dir / "output")
        copied.append("output/")
    else:
        selected_paths = [
            job_dir / "config.yaml",
            output_dir / "glb" / "scene" / "combined_scene.glb",
            output_dir / "pointclouds" / "scene" / "combined_scene_bp.ply",
            output_dir / "pointclouds" / "scene" / "pred_points.ply",
            output_dir / "pointclouds" / "meshed" / "ground_aligned.glb",
        ]
        if include_rendering:
            selected_paths.append(output_dir / "rendering")
        if include_evaluation:
            selected_paths.append(output_dir / "evaluation")
        if include_object_glbs:
            selected_paths.append(output_dir / "3D")

        for src in selected_paths:
            if not src.exists():
                continue
            relative = src.relative_to(job_dir)
            _copy_path(src, artifact_dir / relative)
            copied.append(str(relative))

    manifest = {
        "job_id": job_id,
        "published_at": _utcnow(),
        "copied_paths": copied,
        "artifact_root": str(artifact_dir),
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    copied.append("manifest.json")

    artifact_volume.commit()
    _update_job_record(
        job_id,
        published_at=manifest["published_at"],
        artifact_root=str(artifact_dir),
        artifact_paths=copied,
    )
    return manifest


@app.function(image=main_image)
def get_job_record(job_id: str) -> dict | None:
    return job_registry.get(job_id, None)


@app.local_entrypoint()
def main(
    local_image_path: str = "",
    local_gt_scene_path: str = "",
    include_blender: bool = False,
    include_evaluation: bool = False,
    warm_cache: bool = False,
    include_optional_scene_models: bool = False,
    publish_artifacts: bool = False,
    include_object_artifacts: bool = False,
    include_full_output_artifacts: bool = False,
):
    if warm_cache:
        print("Warming main model cache")
        warm_main_model_cache.remote(include_optional_scene_models)
        print("Warming VGGT model cache")
        warm_vggt_model_cache.remote()
        if not local_image_path:
            print("Model cache warmup finished")
            return

    image_path = Path(local_image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    gt_name = None
    gt_bytes = None
    if local_gt_scene_path:
        gt_path = Path(local_gt_scene_path)
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT scene not found: {gt_path}")
        gt_name = gt_path.name
        gt_bytes = gt_path.read_bytes()
    elif include_evaluation:
        raise ValueError("include_evaluation=True requires local_gt_scene_path.")

    job = prepare_job.remote(
        image_path.name,
        image_path.read_bytes(),
        gt_name,
        gt_bytes,
    )
    job_id = job["job_id"]

    print(f"Prepared Modal job {job_id}")
    print("Running steps 1,2,3")
    run_main_steps.remote(job_id, [1, 2, 3])

    print("Running step 4")
    run_vggt_step.remote(job_id)

    print("Running steps 5,6,7")
    run_main_steps.remote(job_id, [5, 6, 7])

    if include_blender:
        print("Running step 8")
        run_blender_render.remote(job_id)

    if include_evaluation:
        print("Running step 9")
        run_main_steps.remote(job_id, [9])

    if publish_artifacts:
        print("Publishing curated artifacts to Modal storage")
        publish_job_artifacts.remote(
            job_id,
            include_rendering=include_blender,
            include_evaluation=include_evaluation,
            include_object_glbs=include_object_artifacts,
            include_full_output=include_full_output_artifacts,
        )

    print(f"Modal pipeline finished for job_id={job_id}")
    print(f"Outputs are stored under /data/jobs/{job_id} in the attached Volume.")
    if publish_artifacts:
        print(f"Published artifacts are stored under /artifacts/jobs/{job_id} in the artifact Volume.")
