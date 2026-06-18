#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
import random
import math
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
import cv2
from tqdm import tqdm

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def PILtoTensor(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def cv2toTensor(cv2_image):
    resized_image = torch.from_numpy(cv2_image) / 255.0
    return resized_image[:, :, [2,1,0]].permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))




def quatProduct_batch(q1, q2):
    r1 = q1[:,0] # [B]
    r2 = q2[:,0]
    v1 = torch.stack((q1[:,1], q1[:,2], q1[:,3]), dim=-1) #[B,3]
    v2 = torch.stack((q2[:,1], q2[:,2], q2[:,3]), dim=-1)

    r = r1 * r2 - torch.sum(v1*v2, dim=1) # [B]
    v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.linalg.cross(v1, v2) #[B,3]
    q = torch.stack((r, v[:,0], v[:,1], v[:,2]), dim=1)

    return q

def load_binary_pickle(filepath):
    with open(filepath, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data

def a_in_b_torch(a, b):
    ainb = torch.isin(a, b)
    return ainb

def normalize_for_percep(input, mod_n = 64):
    h, w = input.shape[1:3]
    # delta_h = ((h-1)//mod_n + 1)*mod_n - h
    # delta_w = ((w-1)//mod_n + 1)*mod_n - w
    # input_padded = torch.nn.functional.pad(input, (delta_w//2, delta_w-delta_w//2, delta_h//2, delta_h-delta_h//2))
    return input*2.-1.

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices_gen(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])

    nd = vertices.shape[2]
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, nd))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}.")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def verts2D(verts, full_proj_transform, image_res):
    ones = torch.ones((1, verts.shape[1], 1), dtype=torch.float, device=verts.device)
    verts_homogeneous = torch.cat([verts, ones], dim=2)
    # (1, N, 4) x (1, 4, 4) -> (1, N, 4)
    projected_vertices_homogeneous = torch.bmm(verts_homogeneous, full_proj_transform.unsqueeze(0)).squeeze(0)
    verts_2D = projected_vertices_homogeneous[:, :3] / projected_vertices_homogeneous.squeeze(0)[:, 3:4]
    verts_img = ((verts_2D[:, :2] + 1) * 0.5 * image_res).long()
    verts_img = verts_img.clamp(0, image_res-1)
    return verts_img

def verts2D_visu(verts_img, image_res):
    verts_2D_visu = torch.ones((3, image_res, image_res), dtype=torch.float, device=verts_img.device)
    verts_2D_visu[:, verts_img[:, 1], verts_img[:, 0]] = 0
    # verts_2D_visu = (verts_2D_visu*255.).permute(1,2,0).detach().cpu().numpy()
    return verts_2D_visu

def verts2D_img(verts_img, img):
    return img[:, verts_img[:, 1], verts_img[:, 0]].permute(1, 0)


def process_p3d_camera(fov, intrinsic):
    flipping  = torch.eye(4, device=intrinsic.device)
    flipping[0, 0] = -1
    flipping[1, 1] = -1
    flipping[2, 2] = 1
    flipping[3, 0] = 0
    flipping[3, 1] = 0
    intrinsic = intrinsic @ flipping
    FoVx = fov / 180 * np.pi
    FoVy = fov / 180 * np.pi
    return intrinsic, FoVx, FoVy

def load_p3d_camera(cam_path):
    payload = torch.load(cam_path)
    elev = payload['elev']
    azim = payload['azim']
    cam_dist = payload['cam_dist']
    fov = payload['fov']
    intrinsic_matrix = payload['intrinsic_matrix']
    extrinsic_matrix = payload['extrinsic_matrix']
    # full_proj = payload['full_projection_matrix']
    intrinsic, FoVx, FoVy = process_p3d_camera(fov, intrinsic_matrix)
    return elev, azim, cam_dist, extrinsic_matrix, intrinsic, FoVx, FoVy

def batch_temporal_positional_encoding(N, d, device):
    position = torch.arange(0, N, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)).to(device)
    pe = torch.zeros(N, d).to(device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def positional_encoding(value, d, device):
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)).to(device)
    pe = torch.zeros(1, d).float().to(device)
    pe[:, 0::2] = torch.sin(value * div_term)
    pe[:, 1::2] = torch.cos(value * div_term)
    return pe

def rgb_to_hsv(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    max_val, _ = torch.max(rgb, dim=0)
    min_val, _ = torch.min(rgb, dim=0)
    delta = max_val - min_val

    # Hue calculation
    h = torch.zeros_like(max_val)
    mask = delta > 0
    h[mask & (max_val == r)] = ((g - b) / delta)[mask & (max_val == r)] % 6
    h[mask & (max_val == g)] = ((b - r) / delta)[mask & (max_val == g)] + 2
    h[mask & (max_val == b)] = ((r - g) / delta)[mask & (max_val == b)] + 4
    h = h / 6.0  # Normalize to [0, 1]
    h[h < 0] += 1

    # Saturation calculation
    s = torch.zeros_like(max_val)
    s[max_val > 0] = delta[max_val > 0] / max_val[max_val > 0]
    # Value calculation
    v = max_val
    return torch.stack([h, s, v], dim=0)

def hsv_to_rgb(hsv):
    h, s, v = hsv[0], hsv[1], hsv[2]
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    rgb = torch.zeros_like(hsv)
    h_ = (h * 6).long() % 6
    rgb[0] = torch.where((h_ == 0) | (h_ == 5), c, torch.where((h_ == 1) | (h_ == 4), x, 0)) + m
    rgb[1] = torch.where((h_ == 1) | (h_ == 2), c, torch.where((h_ == 0) | (h_ == 3), x, 0)) + m
    rgb[2] = torch.where((h_ == 3) | (h_ == 4), c, torch.where((h_ == 2) | (h_ == 5), x, 0)) + m
    return rgb

def arap_loss(A, B, neighbor_indices, percentage=1, weights=None):
    """
    Compute the as-rigid-as-possible (ARAP) loss between original points A and deformed points B.
    
    Args:
        A: Tensor of shape (1, N, 3), original points.
        B: Tensor of shape (1, N, 3), deformed points.
        neighbor_indices: Tensor of shape (1, N, 6), fixed indices of each point's neighbors.

    Returns:
        loss: ARAP loss (scalar).
    """
    # Get neighbor points for A and B
    A_neighbors = A[:, neighbor_indices, :]  # Shape: (1, N, 6, 3)
    B_neighbors = B[:, neighbor_indices, :]  # Shape: (1, N, 6, 3)
    
    # Compute edge vectors
    A_edges = A_neighbors - A.unsqueeze(2)  # Shape: (1, N, 6, 3)
    B_edges = B_neighbors - B.unsqueeze(2)  # Shape: (1, N, 6, 3)
    
    # Compute the ARAP loss
    # || R_i A_edge - B_edge ||^2, where R_i is the best-fit rotation matrix for each point
    A_edges_flat = A_edges.view(-1, 6, 3)  # Shape: (N, 6, 3)
    B_edges_flat = B_edges.view(-1, 6, 3)  # Shape: (N, 6, 3)
    
    # Compute cross covariance matrix
    covariance = torch.einsum('bij,bik->bjk', B_edges_flat, A_edges_flat)  # Shape: (N, 3, 3)
    
    # Perform SVD to get the optimal rotation R
    U, _, Vt = torch.linalg.svd(covariance)  # U, Vt shapes: (N, 3, 3)
    R = U @ Vt  # Optimal rotation matrix, shape: (N, 3, 3)
    
    # # # Ensure R is a valid rotation matrix (handle reflection cases)
    # det = torch.det(R)
    # R[det < 0] *= -1.0
    
    # Rotate A_edges and compute the ARAP loss
    A_edges_rot = torch.einsum('bij,bkj->bki', R.clone().detach(), A_edges_flat)  # Shape: (N, 6, 3)
    # loss = torch.sum((A_edges_rot - B_edges_flat) ** 2)
    # loss = torch.sum((A_edges_flat - B_edges_flat) ** 2)

    diff = (A_edges_rot - B_edges_flat)**2
    if weights is not None:
        loss = (diff * weights.view(-1, 1, 1)).sum()
        # loss = (diff.sum(dim=-1).sum(dim=-1) * weights).sum()
        # loss = diff.sum()
    else:
        # loss = (diff[diff < diff.quantile(percentage)].sum())
        loss = diff.sum()
    
    return loss

# def uv_mesh_triangles(positions):
#     """
#     Create triangles from UV position map using batch processing.
    
#     Args:
#         positions: Tensor of shape (3, H, W) containing vertex positions
        
#     Returns:
#         vertices: Tensor of shape (N, 3) containing unique vertex positions
#         faces: Tensor of shape (M, 3) containing vertex indices for triangles
#     """
#     _, H, W = positions.shape
    
#     # Create vertex indices grid
#     vertex_indices = torch.arange(H * W).reshape(H, W)
    
#     # Create faces by connecting each pixel with right, right-down, and down neighbors
#     # Exclude right-most column and bottom row as they don't have all neighbors
#     faces_list = []
    
#     # Get vertex indices for all pixels except last row and column
#     v00 = vertex_indices[:-1, :-1].reshape(-1)  # Current pixel
#     v10 = vertex_indices[:-1, 1:].reshape(-1)   # Right neighbor
#     v11 = vertex_indices[1:, 1:].reshape(-1)    # Right-down neighbor
#     v01 = vertex_indices[1:, :-1].reshape(-1)   # Down neighbor
    
#     # Create two triangles for each quad
#     # Triangle 1: (v00, v10, v11)
#     # Triangle 2: (v00, v11, v01)
#     faces = torch.stack([
#         torch.stack([v00, v10, v11], dim=1),
#         torch.stack([v00, v11, v01], dim=1)
#     ], dim=1).reshape(-1, 3)
    
#     # Reshape positions into (N, 3) vertex array
#     vertices = positions.permute(1, 2, 0).reshape(-1, 3)
    
#     return vertices, faces


def uv_mesh_triangles(positions):
    """
    Create triangles from UV position map using batch processing.
    
    Args:
        positions: Tensor of shape (3, H, W) containing vertex positions
        
    Returns:
        vertices: Tensor of shape (3, H, W) containing vertex positions (same as input)
        faces: Tensor of shape (3, 2, H-1, W-1) containing vertex indices for triangles
               where faces[i, 0] gives the i-th vertex index of first triangle
               and faces[i, 1] gives the i-th vertex index of second triangle
               for each pixel position
    """
    _, H, W = positions.shape
    
    # Create vertex indices grid
    vertex_indices = torch.arange(H * W, device=positions.device).reshape(H, W)
    
    # Get vertex indices for all pixels except last row and column
    v00 = vertex_indices[:-1, :-1]  # Current pixel
    v10 = vertex_indices[:-1, 1:]   # Right neighbor
    v11 = vertex_indices[1:, 1:]    # Right-down neighbor
    v01 = vertex_indices[1:, :-1]   # Down neighbor
    
    # Create faces with shape (3, 2, H-1, W-1)
    # First triangle: (v00, v10, v11)
    # Second triangle: (v00, v11, v01)
    faces = torch.stack([
        torch.stack([v00, v10, v11], dim=0),  # First triangle
        torch.stack([v00, v11, v01], dim=0)   # Second triangle
    ], dim=1)
    
    return positions, faces

def save_obj_colorful_point_cloud(vertices, colors, file_path):
    """
    Save a tensor of vertices and corresponding colors to an OBJ file.

    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) representing the vertices.
        colors (torch.Tensor): A tensor of shape (N, 3) representing the RGB colors.
        file_path (str): The path to save the OBJ file.
    """
    assert vertices.shape[0] == colors.shape[0], "Vertices and colors must have the same number of points."
    
    with open(file_path, 'w') as f:
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")

def generate_mesh(in_path, out_path):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)
    ms.compute_normal_for_point_clouds(k=10, flipflag=False)
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.save_current_mesh(out_path, save_face_color=False, save_vertex_color=False)


def k_means_clustering(points, num_clusters, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering on a 3D point cloud.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3), representing the 3D points.
        num_clusters (int): Number of clusters (K).
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        torch.Tensor: Cluster assignments for each point (shape: N).
        torch.Tensor: Cluster centers (shape: K, 3).
    """
    # Initialize cluster centers randomly from the points
    centers = points[torch.randperm(points.size(0))[:num_clusters]]

    for i in tqdm(range(max_iters)):
        # Compute distances to each cluster center
        distances = torch.cdist(points, centers)  # Shape: (N, K)

        # Assign points to the nearest cluster center
        cluster_assignments = torch.argmin(distances, dim=1)  # Shape: (N,)

        # Update cluster centers
        new_centers = torch.stack([
            points[cluster_assignments == k].mean(dim=0) if (cluster_assignments == k).sum() > 0 else centers[k]
            for k in range(num_clusters)
        ])

        # Check for convergence
        if torch.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    # return cluster_assignments, centers

    # Calculate covariance matrices for each cluster
    covariances = []
    for k in range(num_clusters):
        cluster_points = points[cluster_assignments == k]  # Points assigned to cluster k
        if cluster_points.size(0) > 0:
            # Center the points by subtracting the cluster mean
            centered_points = cluster_points - centers[k]
            # Compute covariance matrix
            covariance = (centered_points.T @ centered_points) / cluster_points.size(0)  # Shape: (3, 3)
        else:
            # If no points assigned to the cluster, assign a zero covariance matrix
            covariance = torch.zeros((3, 3), device=points.device)
        covariances.append(covariance)

    covariances = torch.stack(covariances)  # Shape: (K, 3, 3)

    return cluster_assignments, centers, covariances

def Mahalanobis_distance(pts, kpts, cov):
    # # pts:  (1, N, 3)
    # # kpts: (N, K, 3)
    # # cov:  (N, K, 3, 3)
    N, K, _ = kpts.shape
    d = (pts - kpts).view(N*K, 1, 3)     # [N*K, 1, 3]
    AB  = torch.bmm(d, cov.view(N*K, 3, 3))
    ABC = torch.bmm(AB, d.permute(0, 2, 1))    # [N*K, 1]
    # return torch.exp(-ABC).view(N, K)
    return torch.softmax(-ABC.view(N, K), dim=1)

def represent_points_with_keypoints(points, keypoints):
    """
    Represent points relative to their closest keypoints.
    
    Args:
        points: Tensor of shape (1, N, 3), all points.
        keypoints: Tensor of shape (1, K, 3), key points.

    Returns:
        Tensor of shape (1, N, 6), where each point is represented as 
        [closest_keypoint_coords, relative_position].
    """
    # Compute pairwise distances
    diff = points.unsqueeze(2) - keypoints.unsqueeze(1)  # Shape: (1, N, K, 3)
    distances = torch.sum(diff ** 2, dim=-1)  # Shape: (1, N, K)
    
    # Find the closest key points
    closest_indices = torch.argmin(distances, dim=-1)  # Shape: (1, N)
    closest_keypoints = torch.gather(
        keypoints, dim=1, index=closest_indices.unsqueeze(-1).expand(-1, -1, 3)
    )  # Shape: (1, N, 3)
    
    # Compute relative positions
    relative_positions = points - closest_keypoints  # Shape: (1, N, 3)
    
    # Concatenate closest keypoints and relative positions
    representation = torch.cat([closest_keypoints, relative_positions], dim=-1)  # Shape: (1, N, 6)
    
    return representation

def get_lm(idname):
    if idname == 'fox':
        lm3d = torch.tensor([# [-0.295361, -0.143545, 0.270077],
                            [-0.291609, -0.155117, 0.279474],
                            # [-0.273475, -0.011873, 0.224137],
                            [-0.245688, 0.039804, 0.248986],
                            # [-0.125123, -0.003584, 0.282769],
                            [-0.145260, 0.032155, 0.282568],
                            # [-0.115685, -0.123042, 0.335878],
                            [-0.122688, -0.134572, 0.342494],
                            [-0.150973, -0.158550, 0.337818],
                            # [-0.123243, -0.134835, 0.341726],
                            [-0.274369, -0.164682, 0.290243],
                            # [-0.241425, -0.179818, 0.311472],

                            # [0.096214, -0.117382, 0.305678],
                            [0.100570, -0.126827, 0.308411],
                            # [0.135666, 0.005811, 0.275970],
                            [0.159607, 0.052260, 0.294608],
                            # [0.278413, -0.014930, 0.237293],
                            [0.268785, 0.022318, 0.252335],
                            # [0.314624, -0.138155, 0.255241],
                            [0.302743, -0.157758, 0.269509],
                            [0.257274, -0.180689, 0.295596],
                            [0.147665, -0.170481, 0.324246],

                            [-0.159759, -0.333305, 0.331175],
                            # [-0.104847, -0.281906, 0.475293],
                            [-0.149407, -0.252261, 0.398211],
                            # [0.001561, -0.273779, 0.549502],
                            [0.001676, -0.291903, 0.511419],
                            # [0.083646, -0.279211, 0.504950],
                            [0.111574, -0.280619, 0.436139],
                            [0.147374, -0.327541, 0.332902],
                            [0.083231, -0.405308, 0.352601],
                            # [0.008113, -0.481737, 0.399590],
                            [0.016037, -0.484982, 0.399349],
                            [-0.093996, -0.400904, 0.353741],

                            # # pupil
                            [-0.168751, -0.107478, 0.300796],
                            [0.184509, -0.111187, 0.296371]
                            
                            # [-0.474079, 0.593881, 0.089427],
                            # [-0.355025, 0.548469, 0.077247],
                            # [-0.225804, 0.446868, 0.060686],
                            # [0.481302, 0.599203, 0.094475],
                            # [0.363072, 0.549255, 0.082461],
                            # [0.227146, 0.450541, 0.056252],
                            
                            ], dtype=torch.float)
    elif idname =='dog':
        lm3d = torch.tensor([[-0.239596, 0.358125, 0.238073],
                            [-0.205596, 0.401485, 0.242093],
                            [-0.133885, 0.396523, 0.231341],
                            [-0.115559, 0.379710, 0.317507],
                            [-0.151991, 0.344447, 0.308156],
                            [-0.201268, 0.334159, 0.278324],
                            [0.118382, 0.367648, 0.293314],
                            [0.160138, 0.413434, 0.276329],
                            [0.216453, 0.393040, 0.261576],
                            [0.246359, 0.350798, 0.235256],
                            [0.218858, 0.320310, 0.273561],
                            [0.159133, 0.335316, 0.308993],

                            # [-0.011107, 0.312154, 0.677940],

                            # [-0.312411, 0.091391, 0.203624],
                            # [-0.130115, 0.048192, 0.558518],
                            # [-0.006676, 0.090284, 0.604875],
                            # [0.167635, 0.045196, 0.511972],
                            # [0.273182, 0.071753, 0.207461],
                            # [0.145822, -0.090812, 0.348546],
                            # [0.007586, -0.198862, 0.494366],
                            # [-0.153840, -0.099272, 0.370820],
                            [-0.305772, 0.099781, 0.212298],
                            [-0.176932, 0.029998, 0.392268],
                            [-0.002025, 0.073249, 0.575522],
                            [0.182506, 0.044220, 0.393863],
                            [0.274529, 0.065032, 0.203630],
                            
                            [0.148641, -0.130742, 0.326417],
                            [0.002717, -0.230551, 0.498289],
                            [-0.151162, -0.156305, 0.356792],
                            
                            # # pupil
                            # [-0.151851, 0.360686, 0.227295],
                            # [0.163005, 0.361674, 0.290084]
                            [-0.146022, 0.346330, 0.213752],
                            [0.156340, 0.364158, 0.290933]
                            ], dtype=torch.float)
    elif idname =='bear':
        lm3d = torch.tensor([[-0.227490, 0.262953, 0.182310],
                            [-0.201284, 0.301908, 0.184683],
                            [-0.165366, 0.304174, 0.225626],
                            [-0.155021, 0.281601, 0.228660],
                            [-0.171033, 0.250723, 0.206711],
                            [-0.188532, 0.246029, 0.199097],

                            [0.119429, 0.273287, 0.189472],
                            [0.147050, 0.306014, 0.204134],
                            [0.184931, 0.285900, 0.163790],
                            [0.202548, 0.254770, 0.159952],
                            [0.179790, 0.235024, 0.179366],
                            [0.144782, 0.243044, 0.198739],

                            [-0.143791, -0.030206, 0.321801],
                            [-0.089874, -0.006637, 0.518643],
                            [-0.004386, 0.026032, 0.561653],
                            [0.092932, -0.011914, 0.517280],
                            [0.107908, -0.049152, 0.348045],
                            [0.115774, -0.156224, 0.370347],
                            [-0.002765, -0.294301, 0.474141],
                            [-0.126761, -0.139163, 0.357626]], dtype=torch.float)
    elif idname =='cat':
        lm3d = torch.tensor([
                            # [-0.309843, -0.039507, 0.200668],
                            # [-0.241811, 0.069652, 0.220335],
                            # [-0.163371, 0.056416, 0.255746],
                            # [-0.116386, -0.033966, 0.285495],
                            # [-0.168101, -0.106605, 0.298141],
                            # [-0.267788, -0.099777, 0.259538],

                            [-0.303152, -0.065591, 0.231448],
                            # [-0.253456, 0.072506, 0.217901],
                            [-0.308214, 0.051106, 0.190255],
                            [-0.158355, 0.068452, 0.259754],
                            [-0.122018, -0.072444, 0.278824],
                            [-0.172975, -0.100158, 0.282957],
                            # [-0.267162, -0.091744, 0.251863],
                            [-0.307064, -0.078600, 0.239917],

                            [0.114137, -0.065284, 0.237896],
                            [0.150546, 0.056483, 0.227517],
                            [0.219360, 0.067247, 0.218533],
                            [0.298163, -0.047031, 0.173607],
                            [0.250654, -0.112276, 0.240796],
                            [0.174129, -0.107712, 0.269665],

                            # # nose
                            # [-0.004215, -0.148210, 0.473251],

                            [-0.084486, -0.376119, 0.322901],
                            [-0.096416, -0.336090, 0.378019],
                            [0.003100, -0.296536, 0.402401],
                            [0.099795, -0.331894, 0.376553],
                            [0.089713, -0.377833, 0.314865],
                            # [0.091449, -0.423940, 0.324138],
                            [0.087647, -0.444785, 0.344164],
                            [0.008444, -0.473711, 0.400961],
                            # [0.014975, -0.498763, 0.395435],
                            # [-0.076607, -0.416243, 0.341229],
                            [-0.063354, -0.446932, 0.366632],

                        

                            
                            # # pupil
                            [-0.202347, -0.012572, 0.245945],
                            [0.193218, -0.008729, 0.236577]
                            ], dtype=torch.float)
    elif idname =='pigly':
        lm3d = torch.tensor([                            
                            # [-0.284694, 0.032724, 0.247741],
                            # [-0.237990, 0.140937, 0.270676],
                            # [-0.190327, 0.128889, 0.305911],
                            # [-0.176379, 0.036413, 0.328760],
                            # [-0.226271, -0.025192, 0.304922],
                            # [-0.269811, -0.019240, 0.275659],
                            [-0.285476, 0.028700, 0.247882],
                            [-0.247809, 0.127482, 0.265091],
                            [-0.202870, 0.129452, 0.296528],
                            [-0.173053, 0.039769, 0.335493],
                            [-0.228034, 0.014950, 0.303543],
                            [-0.274718, 0.029913, 0.271654],


                            # [0.178712, 0.063714, 0.286321],
                            # [0.194730, 0.115037, 0.295103],
                            # [0.255352, 0.102958, 0.269939],
                            # [0.266664, 0.061309, 0.259804],
                            # [0.261357, 0.041233, 0.300023],
                            # [0.211152, 0.034691, 0.304569],

                            # [0.151725, 0.038334, 0.311311],
                            # [0.189848, 0.133343, 0.301592],
                            # [0.263962, 0.117869, 0.263005],
                            # [0.288292, 0.025878, 0.252767],
                            # [0.266639, -0.024231, 0.290650],
                            # [0.199875, -0.020280, 0.319518],
                            [0.145860, 0.050044, 0.319094],
                            [0.206845, 0.134519, 0.293729],
                            [0.267783, 0.103758, 0.262524],
                            [0.287422, 0.028381, 0.252534],
                            [0.265771, 0.008502, 0.292976],
                            [0.224774, 0.010428, 0.308179],


                            # [0.003461, -0.060614, 0.622794],

                            [-0.168714, -0.212599, 0.401356],
                            [-0.112193, -0.185278, 0.595927],
                            [0.010045, -0.204826, 0.605507],
                            [0.119914, -0.195916, 0.600247],
                            [0.161747, -0.234035, 0.411962],
                            [0.123573, -0.335028, 0.423594],
                            [0.012743, -0.430370, 0.524627],
                            [-0.098126, -0.344767, 0.437041],

                            # # pupil
                            [-0.236616, 0.068862, 0.278243],
                            [0.226661, 0.076052, 0.278870]
                            
                            # [-0.689949, 0.527824, 0.012599],
                            # [-0.489298, 0.599700, 0.066863],
                            # [-0.303058, 0.492846, 0.087037],
                            # [0.645804,  0.619476, 0.056419],
                            # [0.472674,  0.595371, 0.053327],
                            # [0.267774,  0.447607, 0.093999],

                            ], dtype=torch.float)




    return lm3d

class DualQuaternion:
    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Multiplies two batches of quaternions q1 and q2.
        q1, q2: Tensors of shape (N, 4) representing quaternions [w, x, y, z].
        Returns: Tensor of shape (N, 4) representing the product.
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def quaternion_inverse(q):
        """
        Computes the inverse of a batch of quaternions.
        q: Tensor of shape (N, 4) [w, x, y, z].
        Returns: Tensor of shape (N, 4) representing the inverse quaternions.
        """
        norm_squared = torch.sum(q**2, dim=-1, keepdim=True)
        return q * torch.tensor([1, -1, -1, -1], device=q.device).view(1, 4) / norm_squared

    @staticmethod
    def translation_to_dual_part(translation, rotation):
        """
        Computes the dual part for a batch of dual quaternions.
        translation: Tensor of shape (N, 3) [t_x, t_y, t_z].
        rotation: Tensor of shape (N, 4) [w, x, y, z].
        Returns: Tensor of shape (N, 4) representing the dual parts.
        """
        # Convert translation to pure quaternion (N, 4)
        t_quaternion = torch.cat([torch.zeros(translation.size(0), 1, device=translation.device), translation], dim=-1)
        dual_part = 0.5 * DualQuaternion.quaternion_multiply(t_quaternion, rotation)
        return dual_part

    @staticmethod
    def from_rotation_translation(rotation, translation):
        """
        Creates a batch of dual quaternions from rotation and translation.
        rotation: Tensor of shape (N, 4) [w, x, y, z].
        translation: Tensor of shape (N, 3) [t_x, t_y, t_z].
        Returns: Tuple of tensors (real_part, dual_part), each of shape (N, 4).
        """
        real_part = rotation
        dual_part = DualQuaternion.translation_to_dual_part(translation, rotation)
        return real_part, dual_part

    @staticmethod
    def to_rotation_translation(dual_quaternion):
        """
        Extracts rotation and translation from a batch of dual quaternions.
        dual_quaternion: Tuple of tensors (real_part, dual_part), each of shape (N, 4).
        Returns: Tuple (rotation, translation).
            rotation: Tensor of shape (N, 4) [w, x, y, z].
            translation: Tensor of shape (N, 3) [t_x, t_y, t_z].
        """
        real_part, dual_part = dual_quaternion
        rotation = real_part
        rotation_inverse = DualQuaternion.quaternion_inverse(rotation)
        pure_translation_quaternion = 2 * dual_part
        p = DualQuaternion.quaternion_multiply(pure_translation_quaternion, rotation_inverse)
        translation = p[:, 1:]  # Extract vector part (t_x, t_y, t_z)
        return rotation, translation

    @staticmethod
    def blend_dual_quaternions(weights, dual_quaternions):
        """
        Blends a batch of dual quaternions.
        weights: Tensor of shape (N,) representing the blending weights.
        dual_quaternions: Tuple of tensors (real_parts, dual_parts), each of shape (N, 4).
        Returns: Tuple (blended_real, blended_dual), each of shape (N, 4).
        """
        real_parts, dual_parts = dual_quaternions
        blended_real = torch.sum(weights[:, None] * real_parts, dim=0)
        blended_dual = torch.sum(weights[:, None] * dual_parts, dim=0)

        # Normalize the blended real part
        norm = torch.norm(blended_real, dim=-1, keepdim=True)
        blended_real /= norm
        blended_dual /= norm

        return blended_real, blended_dual




class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        self.raster_settings_dict = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        self.raster_settings = dict2obj(self.raster_settings_dict)

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)

        return pixel_vals, pix_to_face, bary_coords #, vismask

    def extra_repr(self):
        return '{image_size}px, blur_radius={blur_radius}, faces_per_pixel={faces_per_pixel}'.format(
            **self.raster_settings_dict)


class Embedder(nn.Module):
    def __init__(self, N_freqs, input_dims=3, include_input=True) -> None:
        super().__init__()
        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        embed_fns = []
        if self.include_input:
            embed_fns.append(lambda x: x)

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns
        self.dim_embeded = self.input_dims*len(self.embed_fns)

    def forward(self, inputs, alpha = 10.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], 2)
        start = 0
        # print(alpha)
        # if self.include_input:
        #     start = 1
        # for i in range(output.shape[1]//2):
        #     output[:, (2*i+start)*self.input_dims:(2*(i+1)+start)*self.input_dims] *= (1-math.cos(math.pi*(max(min(alpha-i, 1.), 0.))))*.5
        return output