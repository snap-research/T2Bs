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
from utils.general_utils import PILtoTensor, load_p3d_camera, process_p3d_camera, batch_temporal_positional_encoding, positional_encoding
from utils.graphics_utils import focal2fov


class Scene_mica:
    def __init__(self, image_folder, camera_folder, mask_folder, white_background, device, camera_test=None, video_len=1, n_views=180, d_pe=64):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        
        self.device = device

        self.N_frames = len(os.listdir(camera_folder))
        self.cameras = [[None] * n_views for _ in range(video_len)]

        self.d_pe = d_pe
        ts_pe = batch_temporal_positional_encoding(video_len, self.d_pe, self.device)


        for frame_id in tqdm(range(n_views)):
            image_name_ori = 'image_' + str(frame_id).zfill(4)
            if camera_test is not None:
                image_name_cam = 'matrices_' + str(camera_test + video_len*6).zfill(4)
            else:
                image_name_cam = 'matrices_' + str(frame_id).zfill(4)
            cam_path = os.path.join(camera_folder, image_name_cam+'.pt')
            elev, azim, cam_dist, extrinsic, intrinsic, FoVx, FoVy = load_p3d_camera(cam_path)



            image_path = os.path.join(image_folder, image_name_ori+'.jpg')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            # alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            # alpha = Image.open(alpha_path)
            # alpha = PILtoTensor(alpha)
            # alpha = mouth_mask = torch.ones((1, 512, 512), dtype=torch.float, device=device)

            # # if add head mask
            if mask_folder is None:
                head_mask = torch.ones((1, gt_image.shape[1], gt_image.shape[1]), dtype=torch.float, device=self.device)
            else:
                head_mask_path = os.path.join(mask_folder, image_name_ori+'.jpg')
                head_mask = Image.open(head_mask_path)
                head_mask = PILtoTensor(head_mask)
                head_mask = head_mask[:3, ...]
            # gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            # gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            # mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            # mouth_mask = Image.open(mouth_mask_path)
            # mouth_mask = PILtoTensor(mouth_mask)
            mouth_mask = torch.ones((1, gt_image.shape[1], gt_image.shape[1]), dtype=torch.float, device=self.device)

            # # position encoding for uid
            
            # def batch_positional_encoding(N, d):
            #     position = torch.arange(0, N, dtype=torch.float).unsqueeze(1)
            #     div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            #     pe = torch.zeros(N, d)
            #     pe[:, 0::2] = torch.sin(position * div_term)
            #     pe[:, 1::2] = torch.cos(position * div_term)
            #     return pe
            # N = video_len  # Total number of time steps
            # d = 128   # Dimensionality of positional encoding
            # pe_matrix = batch_positional_encoding(N, d)

            uid = 0
            vid = frame_id

            t_pe = ts_pe[uid:uid+1]
            elev_pe = positional_encoding(elev, self.d_pe, self.device)
            azim_pe = positional_encoding(azim, self.d_pe, self.device)
            cam_dist_pe = positional_encoding(cam_dist, self.d_pe, self.device)
            pe = torch.cat((t_pe, elev_pe, azim_pe, cam_dist_pe), dim=1)


            camera_indiv = Camera(colmap_id=None, image=gt_image, 
                                FoVx=FoVx, FoVy=FoVy,
                                proj=intrinsic, w2v=extrinsic, full_proj=None, 
                                head_mask=head_mask, mouth_mask=mouth_mask,
                                image_name=image_name_cam, uid=uid, uid_pe=pe,
                                data_device=self.device
                                )
            # self.cameras.append(camera_indiv)
            self.cameras[uid][vid] = camera_indiv
    
    def getCameras(self):
        return self.cameras





    
