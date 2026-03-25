# Modal Deployment

This repository now includes a starter Modal setup for the default pipeline path.
It uses Modal managed storage primitives directly:

- `Volume` for working data, model cache, and published artifacts
- `Dict` for lightweight job metadata

The runtime scope is:

- `Use_VGGT: true`
- `use_hunyuan21: false`
- `Use_MIDI: false`
- `Use_DPA: false`

## Files

- `modal_app.py`
- `docker/Dockerfile.main`
- `docker/Dockerfile.vggt`
- `docker/Dockerfile.blender`

## What Runs Where

- `main` image: steps `1,2,3,5,6,7,9`
- `vggt` image: step `4`
- `blender` image: step `8`

## Setup

Set the API key locally when you want to run step 2:

```bash
export GOOGLE_API_KEY=...
```

The app uses three Volumes and one Dict:

- `3d-regen-data`
- `3d-regen-cache`
- `3d-regen-artifacts`
- `3d-regen-job-registry`

## Warm Model Cache

Warm the model cache only:

```bash
modal run modal_app.py --warm-cache
```

Warm the cache and include optional Marigold scene models:

```bash
modal run modal_app.py --warm-cache --include-optional-scene-models
```

## Run

Run the default pipeline without Blender:

```bash
modal run modal_app.py --local-image-path input_images/4529.jpg
```

Include Blender rendering:

```bash
modal run modal_app.py --local-image-path input_images/4529.jpg --include-blender
```

Include evaluation:

```bash
modal run modal_app.py --local-image-path input_images/4529.jpg --local-gt-scene-path /path/to/scene.glb --include-evaluation
```

Publish curated artifacts to the artifact Volume after the run:

```bash
modal run modal_app.py --local-image-path input_images/4529.jpg --publish-artifacts
```

## Notes

- The current Modal setup rewrites the runtime config into `/data/jobs/<job_id>/config.yaml`.
- Model caches are redirected to `/cache`.
- Published artifacts are copied into `/artifacts/jobs/<job_id>/`.
- Job metadata is written to the Modal Dict `3d-regen-job-registry`.
- The Blender image is the least verified part and may need extra package adjustments depending on the target Modal GPU environment.
