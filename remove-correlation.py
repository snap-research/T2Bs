import os
import glob
import torch
import trimesh
import pandas as pd

# =============== Config ===============
idname='cow'
blendshape_folder = f"assets/{idname}/runs/0000/mesh_captures"
output_folder = f"assets/{idname}/runs/0000/de-correlated-mesh"
os.makedirs(output_folder, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify pairs to decorrelate: {target: [correlated_shapes]}
pairs_to_clean = {
    # # mouth motion
    "mouth_open": ["eyes_half_mouth_open_wide"],
    "mouth_open_wide": ["eyes_half_mouth_open_wide"],
    "smile_open": ["close_eyes"],
    "smile_closed": ["close_eyes"],
    "oo": ["close_eyes"],
    "ww": ["close_eyes"],
    "long": ["close_eyes", "brows_down", "frown"],
    "neutral": ["close_eyes", "brows_down", "frown"],
    "eyes_half_mouth_open_wide": ["close_eyes", "brows_down", "frown"],
    
    # # eye motion
    "close_eyes": ["mouth_open", "mouth_open_wide", "smile_open", "smile_closed", "oo", "ww", "long", "neutral"],
    "brows_down": ["mouth_open", "mouth_open_wide", "smile_open", "smile_closed", "oo", "ww", "long", "neutral"],
    "frown": ["mouth_open", "mouth_open_wide", "smile_open", "smile_closed", "oo", "ww", "long", "neutral"],
    "half_eyes_close": ["mouth_open", "mouth_open_wide", "smile_open", "smile_closed", "oo", "ww", "long", "neutral"],

}

# ======================================

# 1. Load meshes
blendshape_files = sorted(glob.glob(os.path.join(blendshape_folder, "*.obj")))
meshes = {
    os.path.splitext(os.path.basename(f))[0]: trimesh.load(f, process=False)
    for f in blendshape_files
}
names = list(meshes.keys())
num_shapes = len(names)
print(names)

# 2. Compute average neutral or choose neutral
all_vertices = [mesh.vertices for mesh in meshes.values()]
neutral_vertices = sum(all_vertices) / len(all_vertices)
# neutral_key = 'neutral'
# neutral_vertices = meshes[neutral_key].vertices
num_vertices = neutral_vertices.shape[0]

# 3. Convert to displacement tensors
blendshapes = {
    name: torch.tensor(mesh.vertices - neutral_vertices, dtype=torch.float32, device=device).reshape(-1)
    for name, mesh in meshes.items()
}

# 4. Stack displacement vectors into a [S, D] matrix (S = #shapes, D = 3 * num_vertices)
shape_names = list(blendshapes.keys())
X = torch.stack([blendshapes[n] for n in shape_names], dim=0)  # [S, D]
nonzero_mask = (X.norm(dim=1) > 0)
X_used = X[nonzero_mask]
names_used = [shape_names[i] for i, keep in enumerate(nonzero_mask.tolist()) if keep]

# 5. Mean-center across shapes (this is "minus mean" in PCA)
mu = X_used.mean(dim=0, keepdim=True)            # [1, D]
Xc = X_used - mu                                 # [S_used, D]

# 6. Economy SVD for PCA in torch (works on CPU or GPU)
#    Xc = U @ diag(S) @ Vh ; principal components are rows of Vh
U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # U:[S_used,r], S:[r], Vh:[r,D]; r=min(S_used, D)

# Explained variance and ratios
S2 = (S ** 2) / (Xc.shape[0] - 1)               # eigenvalues of covariance
explained_variance = S2                          # [r]
explained_variance_ratio = explained_variance / explained_variance.sum()

# 7. Choose how many principal components to keep, e.g., K=20 (tune as you like)
K = min(10, Vh.shape[0])                         # cap by rank
components = Vh[:K, :]                           # [K, D] — each row is an eigenvector

# 8. Fit (project) ALL shapes (including any zero/neutral if present) onto the PCA basis
#    First, center with the same mu computed from X_used:
Xc_all = X - mu                                  # broadcast [S, D] - [1, D]
coeffs_all = Xc_all @ components.T               # [S, K] PCA coefficients for each shape

# 9. Reconstruct from the PCA basis (optional, for checking error/visualization)
Xrec_all = coeffs_all @ components + mu          # [S, D]

for i, name in enumerate(shape_names):
    # Reshape back to [V,3] vertices
    disp = Xrec_all[i].reshape(num_vertices, 3).cpu().numpy()
    verts = (neutral_vertices + disp).astype("float32")  # add back neutral
    # meshes[name].vertices = verts
    
    mesh = meshes[name].copy()
    mesh.vertices = verts
    
    # Save as .obj
    out_path = os.path.join(output_folder, f"{name}_pca.obj")
    mesh.export(out_path)
    print(f"Saved {out_path}")


def decorrelate_against_set(c_t: torch.Tensor, S: torch.Tensor,
                            ridge: float = 1e-6, preserve_norm: bool = True) -> torch.Tensor:
    """
    Remove the projection of c_t onto span(S) (S is [m,K], rows are source coeffs).
    Uses ridge-regularized least squares to handle rank deficiency.
    """
    if S.numel() == 0:
        return c_t

    # Drop near-zero sources to avoid numerical issues
    norms = S.norm(dim=1)
    keep = norms > 1e-10
    S = S[keep]
    if S.shape[0] == 0:
        return c_t

    # Project c_t onto Col(A) where A = S^T (K x m)
    A = S.T                               # [K, m]
    G = A.T @ A                           # [m, m]
    I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    # Solve (G + λI) alpha = A^T c_t
    alpha = torch.linalg.solve(G + ridge * I, A.T @ c_t)   # [m]
    proj = A @ alpha                        # [K]
    c_new = c_t - proj

    if preserve_norm and c_new.norm() > 0:
        c_new = c_new * (c_t.norm() / c_new.norm())

    return c_new

def cos(a, b): 
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))

name_to_idx = {n: i for i, n in enumerate(shape_names)}
coeffs_decorr = coeffs_all.clone()

for tgt, src_list in pairs_to_clean.items():
    if tgt not in name_to_idx:
        print(f"[skip] target '{tgt}' not found among shapes.")
        continue
    tgt_i = name_to_idx[tgt]

    # Gather valid sources (present and not the target itself)
    src_idx = [name_to_idx[s] for s in src_list if s in name_to_idx and s != tgt]
    if len(src_idx) == 0:
        print(f"[skip] no valid sources for '{tgt}'.")
        continue

    c_t = coeffs_decorr[tgt_i]                # [K]
    S   = coeffs_decorr[src_idx]              # [m, K]

    # Debug: cosines before
    before = [(shape_names[i], cos(c_t, coeffs_decorr[i])) for i in src_idx]

    # Decorrelate
    c_t_new = decorrelate_against_set(c_t, S, ridge=1e-6, preserve_norm=True)
    coeffs_decorr[tgt_i] = c_t_new

    # Debug: cosines after
    after  = [(shape_names[i], cos(c_t_new, coeffs_decorr[i])) for i in src_idx]
    print(f"[decorr] {tgt}:")
    for (n_b, b), (n_a, a) in zip(before, after):
        print(f"         vs {n_b:>24s}: cos {b:+.4f} → {a:+.4f}")

# -------- reconstruct decorrelated targets (others unchanged) --------
Xrec_decorr = coeffs_decorr @ components + mu    # [S, D]

for i, name in enumerate(shape_names):
    # Reshape back to [V,3] vertices
    disp = Xrec_decorr[i].reshape(num_vertices, 3).cpu().numpy()
    verts = (neutral_vertices + disp).astype("float32")  # add back neutral
    # meshes[name].vertices = verts
    
    mesh = meshes[name].copy()
    mesh.vertices = verts
    
    # Save as .obj
    out_path = os.path.join(output_folder, f"{name}_decorr.obj")
    mesh.export(out_path)
    print(f"Saved {out_path}")


# # 4. Compute correlation matrix (dot products)
# correlation_matrix = torch.zeros((num_shapes, num_shapes), device=device)
# for i, name_i in enumerate(names):
#     for j, name_j in enumerate(names):
#         dot = (blendshapes[name_i] @ blendshapes[name_j]) / (
#             torch.norm(blendshapes[name_i]) * torch.norm(blendshapes[name_j]) + 1e-8
#         )
#         correlation_matrix[i, j] = dot

# # Save correlation matrix to CSV
# df_corr = pd.DataFrame(correlation_matrix.cpu().numpy(), index=names, columns=names)
# df_corr.to_csv(os.path.join(output_folder, "correlation_report.csv"))
# print(f"Saved correlation report to {output_folder}/correlation_report.csv")

# # 5. Selective orthogonalization
# cleaned_blendshapes = blendshapes.copy()
# for target_name, correlated_list in pairs_to_clean.items():
#     target = blendshapes[target_name].clone()
#     for corr_name in correlated_list:
#         corr = blendshapes[corr_name]
#         proj = (target @ corr) / (corr @ corr + 1e-8)
#         target -= proj * corr
#     cleaned_blendshapes[target_name] = target

# # 6. Save cleaned blendshapes
# for name, vec in cleaned_blendshapes.items():
#     mesh = meshes[name].copy()
#     new_vertices = (vec.cpu().numpy().reshape(num_vertices, 3) + neutral_vertices)
#     mesh.vertices = new_vertices
#     out_path = os.path.join(output_folder, f"clean_{name}.obj")
#     mesh.export(out_path)

# # 7. Save averaged neutral mesh
# neutral_mesh = meshes[names[0]].copy()
# neutral_mesh.vertices = neutral_vertices
# neutral_mesh.export(os.path.join(output_folder, "neutral_avg.obj"))

# print(f"Saved cleaned blendshapes and neutral mesh to {output_folder}")
