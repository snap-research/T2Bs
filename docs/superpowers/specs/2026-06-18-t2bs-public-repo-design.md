# T2Bs — View-Conditioned Deformable Gaussian Splatting: Public Repo Design

**Date:** 2026-06-18
**Paper:** T2Bs (arXiv:2509.10678)
**Goal:** Turn the research code at `/nfs/usr/jluo2/T2Bs_web` into a clean, runnable public GitHub repo containing **only** the View-Conditioned Deformable Gaussian Splatting component.

## Decisions (from brainstorming)

| Topic | Decision |
|-------|----------|
| Entry points | Ship **both** single-GPU (`train.py`) and multi-GPU (`train_multi.py`, rewritten for `torchrun`) |
| Cleanup depth | **Moderate, behavior-preserving** refactor — remove cruft/dead code/hardcoded paths, parameterize magic numbers, add README/requirements/scripts. No algorithm changes. |
| Verification | **Run it.** A100 80GB + torch 2.5.1+cu124 available. Install deps, build the rasterizer, smoke-test `train.py` on the example asset. |
| Mesh generation (Step1X-3D/Trellis) | **Excluded, not mentioned** anywhere. Repo documents deformable-GS training given pre-made meshes. |
| License | **No LICENSE file** for now. README notes: bundled rasterizer is under the Inria/MPII non-commercial Gaussian-Splatting license, so practical use is non-commercial research. |
| Example data | Ship **one toy identity** (Antelope plastic-toy, ASCII name), renamed to a clean slug `antelope_toy`, neutral `halfo_m_o_e`, 4 expressions. |

## Environment (verified)

- GPU: NVIDIA A100-SXM4-80GB, driver 580, CUDA 12.4
- Python 3.11.10 (conda), torch 2.5.1+cu124 (CUDA available)
- Already present: torch, torchvision, tqdm, cv2
- Missing (to install): pytorch3d, lpips, trimesh, plyfile, diff_gaussian_rasterization. **kaolin/simple_knn not needed** (see below).

## Target repository structure

```
T2Bs_public/
├── README.md
├── requirements.txt
├── install.sh
├── .gitignore
├── train.py                      # cleaned single-GPU entry
├── train_multi.py                # cleaned multi-GPU (torchrun RANK/WORLD_SIZE)
├── define_camera.py              # cleaned camera generator (documented tool)
├── scripts/
│   ├── train_example.sh          # clean single-GPU example (from run.sh)
│   └── train_multi.sh            # clean torchrun example (from run_node.sh)
├── arguments/__init__.py
├── scene/{__init__.py, cameras.py, gaussian_model.py}
├── src/deform_model.py
├── gaussian_renderer/{__init__.py, network_gui.py}
├── objrenderer/renderer.py
├── utils/{camera_utils,general_utils,graphics_utils,loss_utils,sh_utils,system_utils}.py
├── cameras/view_*.pt             # precomputed cameras (regenerable via define_camera.py)
├── submodules/
│   └── diff-gaussian-rasterization/   # vendored rasterizer source (moved from top level)
└── assets/
    └── antelope_toy/obj/{close_m_close_e,close_m_halfo_e,close_m_o_e,halfo_m_o_e}/
        {textured.obj, material.png, material_0.mtl}
```

## Files to DELETE

- Duplicates/cruft: `train copy.py`, `train copy 2.py`, `train_multi copy.py`, `src/deform_model copy.py`, `scene/__init__ copy.py`, `scene/gaussian_model copy.py`, `remove-correlation.py`, `remove-correlation copy.py`, `copy_ids.sh`, `run.sh`, `run_id.sh`, `run_node.sh` (replaced by `scripts/`).
- Root artifacts: `kpt.ply` (unreferenced), `texture.jpg` (debug artifact written to cwd by deform_model).
- Empty placeholders: `submodules/diff-gaussian-rasterization/` (empty) and `submodules/simple-knn/` (empty).
- Already excluded during copy: `core.49385`, `assets/*/runs/`, `__pycache__`, `*.egg-info`, `build/`, nested `.git`.

## Code cleanup (behavior-preserving)

### train.py
- Remove both raster-overflow capture hacks (`call_render_break_on_overflow`, `call_and_detect_overflow_fd2`) and their duplicate `RASTER_OVERFLOW_PAT` definitions. If an overflow guard is still wanted, keep a single minimal version; otherwise call `render` directly.
- Delete the `if False:` rigid-fit block.
- Remove unused `import cv2`.
- Replace magic literals: fixed `view = 12` → `--view` arg (default 12); the `30_000` sampled-point count → reference `DeformModel.num_samples` (or a named constant).
- Fix `--start_checkpoint` default (currently `ckpt/chkpnt25000.pth`, which does not exist) — default to `None` / no restore.

### train_multi.py
- Replace hostname sharding (`socket.gethostname() == 'jiahao-dev-4node-worker-N'`) with `RANK`/`WORLD_SIZE` from the torchrun environment.
- Remove the `/nfs/usr/jluo2/...` identity filter and `../trellis2_*` `data_root` defaults; default `data_root` to `assets/`.
- Remove `if False:` blocks and any Step1X/trellis references/comments.

### deform_model.py
- Remove the single line that saves `texture.jpg` to cwd (`torchvision.utils.save_image(self.texture_map, 'texture.jpg')`). No replacement / redirect — just delete it.

### gaussian_model.py
- Resolve `distCUDA2`: the `simple_knn` import is commented out but two active lines call `distCUDA2`. Determine whether those paths execute. Prefer the pytorch3d `knn_points` approach already used in the file; eliminate the `simple_knn` dependency. If a path genuinely needs it and cannot be swapped, that path must be confirmed dead or replaced — **no NameError reachable in shipped code.**

### General
- Remove dead commented-out blocks where they add noise. Keep changes minimal and faithful to the method.

## Install / dependencies

`requirements.txt` (pip): `numpy`, `opencv-python`, `tqdm`, `lpips`, `trimesh`, `plyfile`, `torchvision`. (`torch` assumed pre-installed / matched to CUDA.)

Special installs documented in README + `install.sh`:
- `pytorch3d` (from source or matching wheel for torch 2.5 / cu124)
- `diff-gaussian-rasterization` built from `submodules/diff-gaussian-rasterization` via `pip install ./submodules/diff-gaussian-rasterization`

`install.sh` does NOT install: kaolin, torch-cluster, transformers, diffusers, step1x3d_texture.

## README contents

1. Title + paper link (arXiv:2509.10678) + 2–3 line method summary (view-conditioned deformable Gaussian splatting that registers a set of per-expression meshes into a shared deformable representation).
2. Installation (env, pytorch3d, rasterizer build).
3. Data format: the `assets/<id>/obj/<expr>/{textured.obj, material.png, material_0.mtl}` layout; `cameras/*.pt` and how to regenerate with `define_camera.py`.
4. Quickstart: `bash scripts/train_example.sh` (single-GPU on `antelope_toy`).
5. Multi-GPU: `bash scripts/train_multi.sh` (torchrun).
6. Outputs: where renders/checkpoints/registered meshes are written (`assets/<id>/runs/<log>/...`).
7. Citation (BibTeX).
8. License/acknowledgments: note Inria 3DGS rasterizer non-commercial license; acknowledge original 3DGS.
9. **No** mention of mesh generation / Step1X-3D / Trellis.

## .gitignore

`__pycache__/`, `*.pyc`, `*.egg-info/`, `build/`, `runs/`, `assets/*/runs/`, `*.pth`, `core.*`, `texture.jpg`, `*.so`.

## Verification plan

1. `pip install -r requirements.txt`; install pytorch3d; `pip install ./submodules/diff-gaussian-rasterization`.
2. Confirm all imports resolve (`train.py`, `train_multi.py`).
3. Smoke-test: run `train.py` on `antelope_toy` for a few hundred iterations (reduced), confirm it trains and writes output images to the run dir without crashing. Report actual output.
4. (If feasible) brief `torchrun` smoke test of `train_multi.py` on 1–2 processes.

## Out of scope

- Any mesh-generation code (Step1X-3D, Trellis, step1x3d_texture).
- Algorithm/method changes.
- Pretrained checkpoints.
- Hosting/uploading to GitHub (user does this later).
