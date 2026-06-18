import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm
import numpy as np

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor, cv2toTensor, load_p3d_camera, process_p3d_camera, batch_temporal_positional_encoding, positional_encoding, load_video_frames
from utils.graphics_utils import focal2fov


class Scene:
    def __init__(self, camera_folder, device='cuda:0', video_len=13, n_views=25, d_pe=64):
        
        self.device = device
        self.cameras = [[None] * n_views for _ in range(video_len)]
        self.d_pe = d_pe
        ts_pe = batch_temporal_positional_encoding(video_len, self.d_pe, self.device)

        cam_list = []
        for v in range(n_views):
            cam_list.append(f'view_{v:04}.pt')

        for v in range(n_views):

            cam_path = os.path.join(camera_folder, cam_list[v])

            elev, azim, cam_dist, extrinsic, intrinsic, FoVx, FoVy = load_p3d_camera(cam_path)
            elev_pe = positional_encoding(elev, self.d_pe, self.device)
            azim_pe = positional_encoding(azim, self.d_pe, self.device)
            cam_dist_pe = positional_encoding(cam_dist, self.d_pe, self.device)

            for f in range(len(self.cameras)):
                t_pe = ts_pe[f:f+1]
                pe = torch.cat((t_pe, elev_pe, azim_pe, cam_dist_pe), dim=1)

                camera_indiv = Camera(colmap_id=None, image=None, 
                                    FoVx=FoVx, FoVy=FoVy,
                                    proj=intrinsic, w2v=extrinsic, full_proj=None, 
                                    head_mask=None, mouth_mask=None,
                                    image_name=None, uid=f, uid_pe=pe,
                                    cam_dist=cam_dist, elev=elev, azim=azim,
                                    data_device=self.device
                                    )
                self.cameras[f][v] = camera_indiv
    
    def getCameras(self):
        return self.cameras





    
