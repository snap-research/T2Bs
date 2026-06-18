# T2Bs Public Repo Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the research code copied into `/nfs/usr/jluo2/T2Bs_public` into a clean, runnable public GitHub repo containing only the View-Conditioned Deformable Gaussian Splatting component, verified by an end-to-end training smoke test on an A100.

**Architecture:** A PyTorch3D + Gaussian-splatting training pipeline. `train.py` (single-GPU) and `train_multi.py` (torchrun multi-GPU) deform per-expression meshes of one identity into a shared view-conditioned deformable Gaussian representation. Cleanup is behavior-preserving: remove cruft/dead code/hardcoded infra, parameterize magic numbers, add packaging + docs, and verify by running.

**Tech Stack:** Python 3.11, torch 2.5.1+cu124, pytorch3d, lpips, trimesh, plyfile, diff-gaussian-rasterization (CUDA, vendored).

**Spec:** `docs/superpowers/specs/2026-06-18-t2bs-public-repo-design.md`

**Working dir:** `/nfs/usr/jluo2/T2Bs_public` (all paths below are relative to it unless absolute).

**Note on TDD:** This codebase has no test suite and adding one is out of scope (moderate refactor). Each task therefore verifies via import-checks / smoke-runs instead of unit tests, and commits after passing.

---

## Task 1: Initialize git + baseline commit

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Init repo**

```bash
cd /nfs/usr/jluo2/T2Bs_public
git init -q
git config user.email "andypinxinliu@gmail.com" || true
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
*.egg-info/
build/
dist/
*.so
runs/
assets/*/runs/
*.pth
core.*
texture.jpg
.DS_Store
```

- [ ] **Step 3: Baseline commit (raw copied state, minus ignored)**

```bash
git add -A
git commit -q -m "chore: import raw research code (pre-cleanup baseline)"
git log --oneline | head -1
```
Expected: one commit printed.

---

## Task 2: Delete cruft, duplicates, and unused root artifacts

**Files (delete):** `train copy.py`, `train copy 2.py`, `train_multi copy.py`, `src/deform_model copy.py`, `scene/__init__ copy.py`, `scene/gaussian_model copy.py`, `remove-correlation.py`, `remove-correlation copy.py`, `copy_ids.sh`, `run.sh`, `run_id.sh`, `run_node.sh`, `kpt.ply`, `texture.jpg`

- [ ] **Step 1: Delete**

```bash
cd /nfs/usr/jluo2/T2Bs_public
git rm -q "train copy.py" "train copy 2.py" "train_multi copy.py" \
  "src/deform_model copy.py" "scene/__init__ copy.py" "scene/gaussian_model copy.py" \
  "remove-correlation.py" "remove-correlation copy.py" \
  copy_ids.sh run.sh run_id.sh run_node.sh kpt.ply texture.jpg
```

- [ ] **Step 2: Verify no stray "copy" files remain**

```bash
ls | grep -i copy || echo "NONE"
```
Expected: `NONE`.

- [ ] **Step 3: Commit**

```bash
git commit -q -m "chore: remove duplicate/cruft files and unused root artifacts"
```

---

## Task 3: Restructure rasterizer submodule + clean install script

**Files:**
- Move: top-level `diff-gaussian-rasterization/` → `submodules/diff-gaussian-rasterization/`
- Delete: empty `submodules/simple-knn/`
- Rewrite: `install.sh`

- [ ] **Step 1: Replace empty placeholder with real rasterizer source**

```bash
cd /nfs/usr/jluo2/T2Bs_public
rm -rf submodules/diff-gaussian-rasterization submodules/simple-knn
git mv diff-gaussian-rasterization submodules/diff-gaussian-rasterization 2>/dev/null || \
  { mv diff-gaussian-rasterization submodules/diff-gaussian-rasterization; }
# remove build artifacts if any slipped in
rm -rf submodules/diff-gaussian-rasterization/build submodules/diff-gaussian-rasterization/*.egg-info
ls submodules/
```
Expected: `diff-gaussian-rasterization` only.

- [ ] **Step 2: Confirm rasterizer python package path**

```bash
ls submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
ls submodules/diff-gaussian-rasterization/setup.py
```
Expected: both exist.

- [ ] **Step 3: Rewrite `install.sh`** (GS-only; no kaolin/torch-cluster/transformers/diffusers/step1x3d)

```bash
#!/usr/bin/env bash
set -e

# Core Python deps (assumes torch matching your CUDA is already installed).
pip install -r requirements.txt

# PyTorch3D (build from source; pick the ref matching your torch/CUDA if needed).
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Differentiable Gaussian rasterizer (CUDA extension, vendored).
pip install ./submodules/diff-gaussian-rasterization
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -q -m "refactor: vendor rasterizer under submodules/, GS-only install.sh"
```

---

## Task 4: Reduce example assets to one clean identity

**Files:**
- Keep + rename: `assets/Antelope_Plastic_Toy_Render_round_face_a_headband____a_superhero_mask_1` → `assets/antelope_toy`
- Delete: the other three `assets/*` identities

- [ ] **Step 1: Rename chosen identity, drop the rest**

```bash
cd /nfs/usr/jluo2/T2Bs_public
git mv "assets/Antelope_Plastic_Toy_Render_round_face_a_headband____a_superhero_mask_1" assets/antelope_toy
git rm -rq "assets/Aardvark_Plastic_Toy_Render_mohawk_fur_a_Såanta_hat____a_chef_hat_1" \
  "assets/Akita_Dog_DreamWorks_Style_3D_long_snout_a_crown_of_leaves____a_wizard_hat_1" \
  "assets/Antelope-04"
ls assets/
find assets/antelope_toy -maxdepth 2 -type d
```
Expected: `assets/antelope_toy` with `obj/{close_m_close_e,close_m_halfo_e,close_m_o_e,halfo_m_o_e}`.

- [ ] **Step 2: Confirm each expression has the files train.py reads**

```bash
for e in assets/antelope_toy/obj/*/; do echo "$e"; ls "$e"; done
```
Expected: each has `textured.obj`, `material.png`, `material_0.mtl`.

- [ ] **Step 3: Commit**

```bash
git commit -q -m "chore: ship single example identity (antelope_toy)"
```

---

## Task 5: Clean `train.py` (single-GPU)

**Files:**
- Modify: `train.py`

Reference: current file has two raster-overflow capture helpers (lines ~32-88), an `if False:` rigid-fit block (~250-266), fixed `view = 12` (~243), `30_000` magic literals (~286, ~395), unused `import cv2` (~8), and bad `--start_checkpoint` default (~113).

- [ ] **Step 1: Remove the two overflow-capture helpers and call `render` directly**

Delete lines defining `call_render_break_on_overflow`, `call_and_detect_overflow_fd2`, the duplicate `RASTER_OVERFLOW_PAT`/imports (`io`, `contextlib`, `tempfile`, `warnings`) used only by them. Replace each call site:
- `render_pkg, overflow, dbg = call_render_break_on_overflow(render, viewpoint_cam, gaussians, ppt, background)` → `render_pkg = render(viewpoint_cam, gaussians, ppt, background)` and delete the following `if overflow: ... break` block.
- `mesh_image, overflow, tail = call_and_detect_overflow_fd2(renderer.render_mesh, mesh, background, viewpoint_cam.cam_dist, viewpoint_cam.elev, viewpoint_cam.azim)` → `mesh_image = renderer.render_mesh(mesh, background, viewpoint_cam.cam_dist, viewpoint_cam.elev, viewpoint_cam.azim)` and delete its `if overflow: ... break`.

- [ ] **Step 2: Delete the dead `if False:` rigid-fit block** (the `if False:` ... `else:` — keep only the body that was under `else:`, de-indented one level).

- [ ] **Step 3: Remove `import cv2`** (unused).

- [ ] **Step 4: Parameterize the fixed view.** Add argument near the other parser args:

```python
    parser.add_argument('--view', type=int, default=12, help='Camera view index used during training.')
```
Replace `view  = 12` with `view = args.view`.

- [ ] **Step 5: Replace `30_000` magic literals with the model attribute.** Both `verts_final[:, :-30_000]` and `verts_final[0, :-30_000]` → `verts_final[:, :-DeformModel.num_samples]` / `verts_final[0, :-DeformModel.num_samples]`.

- [ ] **Step 6: Fix checkpoint default.** `--start_checkpoint` default `'ckpt/chkpnt25000.pth'` → `None`.

- [ ] **Step 7: Verify it parses & imports (syntax/static).**

```bash
cd /nfs/usr/jluo2/T2Bs_public
python -c "import ast; ast.parse(open('train.py').read()); print('train.py parses OK')"
grep -n "call_render_break_on_overflow\|call_and_detect_overflow_fd2\|if False\|import cv2\|30_000\|view  = 12" train.py || echo "clean"
```
Expected: `train.py parses OK` then `clean`.

- [ ] **Step 8: Commit**

```bash
git add train.py
git commit -q -m "refactor(train): drop overflow hacks/dead code, parameterize view & sample count"
```

---

## Task 6: Clean `train_multi.py` (multi-GPU / torchrun)

**Files:**
- Modify: `train_multi.py`

Reference: `list_identities()` still has hostname sharding (lines ~57-65) + `/nfs` comment (~55); `data_root` default points at `../trellis2_*` (~333); `--start_checkpoint` bad default (~336); unused `import cv2` (~8); `if False:` viz block (~266-270); a latent NameError in the `use_loss_n` path — `mesh` is referenced (~229/231) but its definition (~214-216) is commented out.

- [ ] **Step 1: Strip hostname sharding from `list_identities`.** Remove the `/nfs/...` commented filter and the entire `import socket` + `if socket.gethostname() == ...` block; rank-based splitting already happens in `__main__` via `all_ids[rank::world_size]`.

- [ ] **Step 2: Fix `data_root` default.** Remove the commented `../trellis2_gpt_textured_test_t2bs` line; set the active default to `assets`:

```python
    parser.add_argument('--data_root', type=str, default='assets')
```

- [ ] **Step 3: Fix checkpoint default.** `--start_checkpoint` default → `None`.

- [ ] **Step 4: Remove `import cv2`** (unused).

- [ ] **Step 5: Delete the `if False:` visualization block** (~266-270).

- [ ] **Step 6: Fix the `use_loss_n` NameError.** Uncomment the mesh definition so `mesh` exists before it is used for normals:

```python
        mesh = Meshes(verts=verts_final[:, :-DeformModel.num_samples], faces=DeformModel.faces_idx[None],
                    textures=TexturesVertex(verts_features=DeformModel.uv_features_dc[:, :-DeformModel.num_samples]))
```
Place it right after `image = render_pkg["render"]` (replacing the commented version), so both the `use_loss_n` branch and the `% 500` visualization branch can reference `mesh`.

- [ ] **Step 7: Verify parse + no leftover infra strings.**

```bash
cd /nfs/usr/jluo2/T2Bs_public
python -c "import ast; ast.parse(open('train_multi.py').read()); print('train_multi.py parses OK')"
grep -n "gethostname\|jiahao-dev\|/nfs/usr\|trellis2\|import cv2\|if False" train_multi.py || echo "clean"
```
Expected: `train_multi.py parses OK` then `clean`.

- [ ] **Step 8: Commit**

```bash
git add train_multi.py
git commit -q -m "refactor(train_multi): torchrun-only sharding, assets data_root, fix use_loss_n bug"
```

---

## Task 7: Clean `src/deform_model.py` and `scene/gaussian_model.py`

**Files:**
- Modify: `src/deform_model.py` (remove `texture.jpg` save, line ~82)
- Modify: `scene/gaussian_model.py` (replace dead `distCUDA2` calls; drop commented simple_knn import)

- [ ] **Step 1: Remove the `texture.jpg` save line** in `src/deform_model.py`:
Delete: `torchvision.utils.save_image(self.texture_map, 'texture.jpg')       # # # # name of the texture map`

- [ ] **Step 2: In `scene/gaussian_model.py`, replace the two active `distCUDA2(points)` calls** (in `update_xyz_feature_pc`, ~lines 302 & 310) with the knn-based form already used at line 292:
```python
                dist2 = torch.clamp_min(knn_points(points[None, ...], points[None, ...], K=1).dists.squeeze()**2, 0.0000001)
```
Also delete the commented `# from simple_knn._C import distCUDA2` line (~20) and the now-orphaned `# dist2 = torch.clamp_min(distCUDA2(points), ...)` comments.

- [ ] **Step 3: Verify no `distCUDA2`/`simple_knn`/`texture.jpg` references remain.**

```bash
cd /nfs/usr/jluo2/T2Bs_public
python -c "import ast; ast.parse(open('src/deform_model.py').read()); ast.parse(open('scene/gaussian_model.py').read()); print('parse OK')"
grep -rn "distCUDA2\|simple_knn\|texture.jpg" src/deform_model.py scene/gaussian_model.py || echo "clean"
```
Expected: `parse OK` then `clean`.

- [ ] **Step 4: Commit**

```bash
git add src/deform_model.py scene/gaussian_model.py
git commit -q -m "refactor: drop texture.jpg debug save and simple_knn/distCUDA2 dependency"
```

---

## Task 8: Add `requirements.txt` and run scripts

**Files:**
- Create: `requirements.txt`, `scripts/train_example.sh`, `scripts/train_multi.sh`

- [ ] **Step 1: Write `requirements.txt`**

```
numpy
opencv-python
tqdm
lpips
trimesh
plyfile
torchvision
```
(Note in README: torch must be installed to match your CUDA; pytorch3d + diff-gaussian-rasterization are installed by `install.sh`.)

- [ ] **Step 2: Write `scripts/train_example.sh`**

```bash
#!/usr/bin/env bash
# Single-GPU example: register the example identity's expressions.
set -e
python train.py \
  --idname antelope_toy \
  --neutral halfo_m_o_e \
  --n_views 25 \
  --log 0000 \
  --deform_fc --normalize_mesh --view_independent --use_loss_n
```

- [ ] **Step 3: Write `scripts/train_multi.sh`**

```bash
#!/usr/bin/env bash
# Multi-GPU example via torchrun. Processes all identities under --data_root,
# split across ranks. Set --nproc_per_node to your GPU count.
set -e
torchrun --standalone --nproc_per_node=8 train_multi.py \
  --data_root assets \
  --neutral halfo_m_o_e \
  --n_views 25 \
  --run_log 0000 \
  --deform_fc --normalize_mesh --use_loss_n
```

- [ ] **Step 4: Make executable + commit**

```bash
cd /nfs/usr/jluo2/T2Bs_public
chmod +x scripts/train_example.sh scripts/train_multi.sh install.sh
git add requirements.txt scripts/ install.sh
git commit -q -m "build: add requirements.txt and example run scripts"
```

---

## Task 9: Install dependencies and build the rasterizer (verification env)

**Files:** none (environment setup)

- [ ] **Step 1: Install pip deps**

```bash
cd /nfs/usr/jluo2/T2Bs_public
pip install -r requirements.txt 2>&1 | tail -5
```

- [ ] **Step 2: Install pytorch3d** (matching torch 2.5/cu124). Try prebuilt; fall back to source.

```bash
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)" 2>/dev/null || \
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | tail -15
```

- [ ] **Step 3: Build & install the rasterizer**

```bash
pip install ./submodules/diff-gaussian-rasterization 2>&1 | tail -15
```

- [ ] **Step 4: Verify all imports resolve**

```bash
python -c "
import torch, torchvision, pytorch3d, lpips, trimesh, plyfile, tqdm, cv2
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
print('all imports OK; cuda', torch.cuda.is_available())
"
```
Expected: `all imports OK; cuda True`.

---

## Task 10: End-to-end smoke test of `train.py` and fix breakages

**Files:** possibly small fixes to modules surfaced by running.

- [ ] **Step 1: Confirm camera count supports `--n_views 25` and view index 12**

```bash
cd /nfs/usr/jluo2/T2Bs_public
ls cameras/view_*.pt | wc -l
```
Expected: ≥ 25. If fewer, lower `--n_views` in the smoke test accordingly and note it.

- [ ] **Step 2: Short smoke run** (small iteration count via a temporary cap). Run the example but interrupt early — set a tiny range by running with a low ceiling. Simplest: run the script in background and stop after it logs a few steps and writes a `train/` image.

```bash
cd /nfs/usr/jluo2/T2Bs_public
timeout 1200 python train.py --idname antelope_toy --neutral halfo_m_o_e \
  --n_views 25 --log smoke --deform_fc --normalize_mesh --view_independent --use_loss_n \
  2>&1 | tee /tmp/t2bs_smoke.log | tail -40
```
Expected: lines like `step: 0, huber: ...`, no traceback, and an image appears under `assets/antelope_toy/runs/smoke/train/`.

- [ ] **Step 3: Confirm outputs exist**

```bash
ls assets/antelope_toy/runs/smoke/train/ | head
```
Expected: at least `0.jpg` (and `500.jpg` if it ran that far).

- [ ] **Step 4: If it crashed,** debug with superpowers:systematic-debugging. Common likely fixes: missing arg defaults, camera index out of range (reduce `--n_views`/`--view`), or a module import path. Apply the minimal fix, re-run Step 2.

- [ ] **Step 5: Remove the smoke run output (it is gitignored anyway) and commit any code fixes**

```bash
rm -rf assets/antelope_toy/runs/smoke
git add -A && git commit -q -m "fix: resolve issues found during train.py smoke test" || echo "no code changes"
```

---

## Task 11: Multi-GPU smoke test of `train_multi.py`

**Files:** possibly small fixes.

- [ ] **Step 1: Run torchrun on 1 process for a short time**

```bash
cd /nfs/usr/jluo2/T2Bs_public
timeout 900 torchrun --standalone --nproc_per_node=1 train_multi.py \
  --data_root assets --neutral halfo_m_o_e --n_views 25 --run_log smoke_multi \
  --deform_fc --normalize_mesh --use_loss_n 2>&1 | tee /tmp/t2bs_smoke_multi.log | tail -40
```
Expected: `[rank 0/1] ...`, `step: 0, loss: ...`, no traceback.

- [ ] **Step 2: If it crashed,** apply minimal fix (debugging skill), re-run.

- [ ] **Step 3: Clean output + commit fixes**

```bash
rm -rf assets/antelope_toy/runs/smoke_multi
git add -A && git commit -q -m "fix: resolve issues found during train_multi.py smoke test" || echo "no code changes"
```

---

## Task 12: Write `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README** with these sections (no mention of mesh generation / Step1X / Trellis):
  1. **Title** — "T2Bs: View-Conditioned Deformable Gaussian Splatting" + link to arXiv:2509.10678; 2–3 sentence method summary (registers a set of per-expression meshes of one identity into a shared, view-conditioned deformable Gaussian representation).
  2. **Installation** — clone; create env (Python 3.11, torch matching CUDA); `bash install.sh` (installs requirements, pytorch3d, vendored rasterizer). Note A100/CUDA 12.4 tested.
  3. **Data format** — the `assets/<id>/obj/<expr>/{textured.obj, material.png, material_0.mtl}` layout; one expression is the neutral (e.g. `halfo_m_o_e`); `cameras/*.pt` precomputed views, regenerable with `python define_camera.py`.
  4. **Quickstart (single GPU)** — `bash scripts/train_example.sh`; describe key flags (`--idname`, `--neutral`, `--n_views`, `--view`, `--deform_fc`, `--normalize_mesh`, `--view_independent`, `--use_loss_n`).
  5. **Multi-GPU** — `bash scripts/train_multi.sh` (torchrun; identities under `--data_root` split across ranks).
  6. **Outputs** — written under `assets/<id>/runs/<log>/`: `train/*.jpg` (renders), `ckpt/*.pth`, `mesh_captures/*.obj` (registered meshes).
  7. **Citation** — BibTeX placeholder for the T2Bs paper (use arXiv entry).
  8. **License & acknowledgments** — no LICENSE file yet; state that `submodules/diff-gaussian-rasterization` is under the Inria/MPII Gaussian-Splatting **non-commercial research** license, so practical use is non-commercial research; acknowledge the original 3D Gaussian Splatting work.

- [ ] **Step 2: Sanity-check the commands in the README match the scripts/args** (idname `antelope_toy`, neutral `halfo_m_o_e`, flags exist in `train.py`).

```bash
cd /nfs/usr/jluo2/T2Bs_public
grep -- "--idname antelope_toy" scripts/train_example.sh
python train.py --help 2>/dev/null | grep -- "--view\b" && echo "args match"
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -q -m "docs: add README"
```

---

## Task 13: Final sweep

**Files:** none / minor.

- [ ] **Step 1: Confirm no hardcoded personal/infra paths remain**

```bash
cd /nfs/usr/jluo2/T2Bs_public
grep -rn --include='*.py' --include='*.sh' -E "/nfs/usr|jiahao-dev|trellis2|step1x|Step1X|kaolin" . | grep -v docs/ || echo "clean"
```
Expected: `clean`.

- [ ] **Step 2: Confirm tree is tidy**

```bash
git status --short
find . -name '__pycache__' -o -name '*.egg-info' -o -name 'core.*' | grep -v docs/ || echo "no junk"
ls
```
Expected: clean working tree, no junk dirs.

- [ ] **Step 3: Final commit if anything changed**

```bash
git add -A && git commit -q -m "chore: final cleanup sweep" || echo "nothing to commit"
git log --oneline
```

---

## Self-review notes (covered)
- Spec sections → tasks: delete cruft (T2), restructure rasterizer (T3), example asset (T4), train.py clean (T5), train_multi clean (T6), deform_model/gaussian_model (T7), requirements/install/scripts (T3,T8), README (T12), .gitignore (T1), verification (T9–T11), final sweep (T13). All covered.
- distCUDA2/simple_knn resolution: T7 replaces with knn_points, drops dependency.
- texture.jpg: removed in T7 (delete-only per user instruction); also gitignored (T1) as belt-and-suspenders.
- License: README-only note (T12), no LICENSE file, per decision.
