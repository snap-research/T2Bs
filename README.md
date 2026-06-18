# T2Bs: View-Conditioned Deformable Gaussian Splatting

This repository contains the **View-Conditioned Deformable Gaussian Splatting** component of

> **T2Bs: Text-to-Character Blendshapes via Video Generation**
> Jiahao Luo, Chaoyang Wang, Michael Vasilkovsky, Vladislav Shakhrai, Di Liu, Peiye Zhuang, Sergey Tulyakov, Peter Wonka, Hsin-Ying Lee, James Davis, Jian Wang
> [arXiv:2509.10678](https://arxiv.org/abs/2509.10678)

Given a set of per-expression meshes of a single character (e.g. several mouth/eye poses
of the same head), this module registers them into a shared, **view-conditioned deformable
Gaussian representation**. A small deformation network predicts per-vertex offsets,
rotations, scales, colors, and opacities — conditioned on a positional encoding of the
camera view and the target expression — and is optimized so that the deformed Gaussians and
mesh reproduce multi-view renders of each expression. The result is a consistent deformable
model from which registered meshes can be exported.

> **Scope:** This repo releases only the deformable Gaussian splatting / registration stage.
> It assumes you already have textured per-expression meshes (the mesh-generation stage is not
> included). One example identity is provided so the code runs out of the box.

---

## Installation

Tested on an NVIDIA A100 (80 GB), Python 3.11, CUDA 12.4, PyTorch 2.5.1.

```bash
git clone <your-repo-url> T2Bs_public
cd T2Bs_public

# 1. Install PyTorch matching your CUDA (example: CUDA 12.4).
#    See https://pytorch.org/get-started/locally/ — e.g.:
# pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install the remaining dependencies + build the CUDA extensions.
bash install.sh
```

`install.sh` installs the Python requirements, builds [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
from source, and builds the vendored differentiable Gaussian rasterizer in
`submodules/diff-gaussian-rasterization`. Building the CUDA extensions requires a CUDA
toolkit (`nvcc`) matching your PyTorch CUDA version. To speed up the build for a single GPU
architecture, set e.g. `export TORCH_CUDA_ARCH_LIST="8.0"` (A100) before running.

---

## Data format

Each identity lives under `assets/<id>/obj/`, with one subfolder per expression. One of the
expressions is designated the **neutral** template (passed via `--neutral`):

```
assets/
└── <id>/
    └── obj/
        ├── <neutral_expression>/
        │   ├── textured.obj      # mesh with UVs
        │   ├── material.png      # texture map
        │   └── material_0.mtl
        └── <other_expression>/
            ├── textured.obj
            ├── material.png
            └── material_0.mtl
```

Multi-view cameras are precomputed in `cameras/view_*.pt`. You can regenerate them with:

```bash
python define_camera.py
```

The bundled example identity is `assets/antelope_toy`, with neutral expression
`halfo_m_o_e`.

---

## Quickstart (single GPU)

```bash
bash scripts/train_example.sh
```

which runs:

```bash
python train.py \
  --idname antelope_toy \
  --neutral halfo_m_o_e \
  --n_views 25 \
  --log 0000 \
  --deform_fc --normalize_mesh --view_independent --use_loss_n
```

Useful arguments:

| Argument | Description |
|---|---|
| `--idname` | Identity folder name under `assets/`. |
| `--neutral` | Expression folder used as the neutral template. |
| `--n_views` | Number of camera views to use (must be ≤ files in `cameras/`). |
| `--view` | Camera view index used during single-GPU training (default 12). |
| `--normalize_mesh` | Normalize meshes into a canonical scale/position. |
| `--deform_fc` | Use the fully-connected deformation head. |
| `--view_independent` | Zero out the view part of the conditioning (expression-only). |
| `--use_loss_n` | Add the surface-normal supervision loss. |
| `--num_clusters`, `--k` | LBS clustering / neighborhood settings for skinning. |

---

## Multi-GPU

`train_multi.py` processes all identities under `--data_root`, split across `torchrun` ranks:

```bash
bash scripts/train_multi.sh
# or directly:
torchrun --standalone --nproc_per_node=<NUM_GPUS> train_multi.py \
  --data_root assets --neutral halfo_m_o_e --n_views 25 --run_log 0000 \
  --deform_fc --normalize_mesh --use_loss_n
```

Identities are distributed round-robin across ranks (`all_ids[rank::world_size]`), so each
GPU optimizes a disjoint subset.

---

## Outputs

Results are written under `assets/<id>/runs/<log>/`:

- `train/*.jpg` — periodic visualizations (ground-truth render, reconstruction, mesh renders).
- `ckpt/*.pth` — checkpoints (multi-GPU run).
- `mesh_captures/*.obj` — registered per-expression meshes exported during training.

---

## Citation

```bibtex
@article{luo2025t2bs,
  title   = {T2Bs: Text-to-Character Blendshapes via Video Generation},
  author  = {Luo, Jiahao and Wang, Chaoyang and Vasilkovsky, Michael and Shakhrai, Vladislav and Liu, Di and Zhuang, Peiye and Tulyakov, Sergey and Wonka, Peter and Lee, Hsin-Ying and Davis, James and Wang, Jian},
  journal = {arXiv preprint arXiv:2509.10678},
  year    = {2025}
}
```

---

## Acknowledgements & License

This project builds on the differentiable Gaussian rasterizer from
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Inria / MPII)
and on [PyTorch3D](https://github.com/facebookresearch/pytorch3d).

The vendored `submodules/diff-gaussian-rasterization` is distributed under the original
**Gaussian-Splatting license (Inria/MPII), which permits non-commercial research use only**.
Because this repository depends on that rasterizer, practical use of this code is limited to
non-commercial research. No separate top-level license file is provided yet; please respect
the rasterizer's license terms.
