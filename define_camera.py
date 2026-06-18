import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures
)
from pytorch3d.io import load_objs_as_meshes, load_obj
from torchvision.utils import save_image
import torchvision
import os, sys 
import argparse
import json
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description="Training script parameters")
parser.add_argument('--idname', type=str, default='dog', help='id name')
parser.add_argument('--cam_dist', type=float, default=2.25, help='id name')
parser.add_argument('--offsetH', type=float, default=0.0, help='id name')
parser.add_argument('--offsetV', type=float, default=0.0, help='id name')
parser.add_argument('--s', type=float, default=1.0)
parser.add_argument('--ry', type=float, default=0.0)
parser.add_argument('--tx', type=float, default=0.0)
parser.add_argument('--ty', type=float, default=0.0)
parser.add_argument('--tz', type=float, default=0.0)
parser.add_argument('--normalize_mesh', action='store_true')
args = parser.parse_args(sys.argv[1:])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory to save images and matrices
cam_dir = os.path.join('cameras')
os.makedirs(cam_dir, exist_ok=True)

# Generate viewpoints over a sphere
cam_dist = args.cam_dist
video_len  = 25

# viewpoints = [(args.offsetV, azim+args.offsetH) for azim in np.linspace(-60, 60, video_len)] + [(elev+args.offsetV, args.offsetH) for elev in np.linspace(-60, 60, video_len)]   # 25 horizontal + 25 vertical
viewpoints = [(args.offsetV, azim+args.offsetH) for azim in np.linspace(-90, 90, video_len)] + [(elev+args.offsetV, args.offsetH) for elev in np.linspace(-90, 90, video_len)]   # 25 horizontal + 25 vertical
print(viewpoints)

frames = torch.zeros(len(viewpoints), 512, 512, 3)

for i, (elev, azim) in enumerate(viewpoints):
    R, T = look_at_view_transform(cam_dist, elev, azim)  # distance, elevation, azimuth
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Get camera matrices
    intrinsic_matrix = cameras.get_projection_transform().get_matrix().squeeze(0)
    
    extrinsic_matrix = cameras.get_world_to_view_transform().get_matrix().squeeze(0)
    full_projection_matrix = cameras.get_full_projection_transform().get_matrix().squeeze(0)

    matrices = {
        "fov": cameras.fov,
        "intrinsic_matrix": intrinsic_matrix,
        "extrinsic_matrix": extrinsic_matrix,
        "full_projection_matrix": full_projection_matrix,
        "elev": elev,
        "azim": azim,
        "cam_dist": cam_dist
    }
    torch.save(matrices, os.path.join(cam_dir, f'view_{i:04d}.pt'))
