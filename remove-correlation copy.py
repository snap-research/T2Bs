import os
import glob
import torch
import trimesh
import pandas as pd

# =============== Config ===============
idname='fox'
blendshape_folder = f"assets/{idname}/runs/0000/mesh_captures"
output_folder = f"assets/{idname}/runs/0000/de-correlated-mesh"
os.makedirs(output_folder, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify pairs to decorrelate: {target: [correlated_shapes]}
pairs_to_clean = {
    # # mouth motion
    "mouth_open": ["eyes_half_mouth_open_wide"],
    "mouth_open_wide": ["eyes_half_mouth_open_wide"],
    "smile_open": ["close_eyes", "brows_down", "frown"],
    "smile_closed": ["close_eyes", "brows_down", "frown"],
    "oo": ["close_eyes", "brows_down", "frown"],
    "ww": ["close_eyes", "brows_down", "frown"],
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

# 4. Compute correlation matrix (dot products)
correlation_matrix = torch.zeros((num_shapes, num_shapes), device=device)
for i, name_i in enumerate(names):
    for j, name_j in enumerate(names):
        dot = (blendshapes[name_i] @ blendshapes[name_j]) / (
            torch.norm(blendshapes[name_i]) * torch.norm(blendshapes[name_j]) + 1e-8
        )
        correlation_matrix[i, j] = dot

# Save correlation matrix to CSV
df_corr = pd.DataFrame(correlation_matrix.cpu().numpy(), index=names, columns=names)
df_corr.to_csv(os.path.join(output_folder, "correlation_report.csv"))
print(f"Saved correlation report to {output_folder}/correlation_report.csv")

# 5. Selective orthogonalization
cleaned_blendshapes = blendshapes.copy()
for target_name, correlated_list in pairs_to_clean.items():
    target = blendshapes[target_name].clone()
    for corr_name in correlated_list:
        corr = blendshapes[corr_name]
        proj = (target @ corr) / (corr @ corr + 1e-8)
        target -= proj * corr
    cleaned_blendshapes[target_name] = target

# 6. Save cleaned blendshapes
for name, vec in cleaned_blendshapes.items():
    mesh = meshes[name].copy()
    new_vertices = (vec.cpu().numpy().reshape(num_vertices, 3) + neutral_vertices)
    mesh.vertices = new_vertices
    out_path = os.path.join(output_folder, f"clean_{name}.obj")
    mesh.export(out_path)

# 7. Save averaged neutral mesh
neutral_mesh = meshes[names[0]].copy()
neutral_mesh.vertices = neutral_vertices
neutral_mesh.export(os.path.join(output_folder, "neutral_avg.obj"))

print(f"Saved cleaned blendshapes and neutral mesh to {output_folder}")
