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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, quatProduct_batch


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, device='cuda'):
        self.device = device
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling_base = torch.empty(0)
        self._scaling = self._scaling_base
        self._rotation_base = torch.empty(0)
        self._rotation = self._rotation_base
        self._opacity = torch.empty(0)
        self._xyz_d = torch.empty(0)
        self._features_dc_d = torch.empty(0)
        self._features_rest_d = torch.empty(0)
        self._scaling_base_d = torch.empty(0)
        self._scaling_d = self._scaling_base
        self._rotation_base_d = torch.empty(0)
        self._rotation_d = self._rotation_base
        self._opacity_d = torch.empty(0)
        self._features_rest0 = torch.empty(0)
        self._opacity0 = torch.empty(0)
        self.d_idx = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
    
    def capture(self, deform=False):
        if deform:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling_base,
                self._rotation_base,
                self._opacity,
                self._xyz_d,
                self._features_dc_d,
                self._features_rest_d,
                self._scaling_base_d,
                self._rotation_base_d,
                self._opacity_d,
                self._features_rest0,
                self._opacity0,
                self.d_idx,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling_base,
                self._rotation_base,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
    
    def restore(self, model_args, training_args, deform=False, test=False):
        if test:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling_base, 
            self._rotation_base, 
            self._opacity,
            self._xyz_d,
            self._features_dc_d,
            self._features_rest_d,
            self._scaling_base_d,
            self._rotation_base_d,
            self._opacity_d,
            self._features_rest0,
            self._opacity0,
            self.d_idx,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        else:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling_base, 
            self._rotation_base, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        if deform:
            self.training_setup_deform(training_args)
        else:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        # try:
        #     return (self.opacity_activation(self._opacity) - self.opacity_deform).clamp(0, 1)
        # except:
        #     print('run without self.opacity_deform')
        #     return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_verts(self, points, features_dc):
        features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        dist2 = torch.clamp_min(knn_points(points[None], points[None], K=2).dists[..., 1].squeeze().sqrt(), 0.0000001)
        scales = torch.log(dist2)[...,None].repeat(1, 3)
        # dist2 = torch.zeros((points.shape[0], 3), device=self.device) + 0.015
        # scales = torch.log(dist2)
        # scales = torch.log(torch.ones_like(points))
        # dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((points.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        # opacities = inverse_sigmoid(torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = torch.ones((points.shape[0], 1), dtype=torch.float, device=self.device)
        
        self._xyz = points
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc = features_dc[:, 0:1, :]
        # self._features_rest = features[:, 1:, :]
        self._scaling_base = nn.Parameter(scales.requires_grad_(True))
        self._scaling = self._scaling_base
        self._rotation_base = nn.Parameter(rots.requires_grad_(True))
        self._rotation = self._rotation_base
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    # def create_from_verts_deform(self, points, features_dc, scale_base, rotation_base):
    #     features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     opacities = inverse_sigmoid(torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda"))
        
    #     self._xyz = points
    #     # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = features[:,:,1:].transpose(1, 2)
    #     self._features_dc = features_dc[:, 0:1, :]
    #     self._scaling_base = scale_base
    #     self._scaling = self._scaling_base
    #     self._rotation_base = rotation_base
    #     self._rotation = self._rotation_base
    #     self._opacity = opacities
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))
    def create_from_verts_deform(self, idx):
        points = self._xyz[idx]
        features_dc = self._features_dc[idx]
        features_rest = self._features_rest[idx]
        scaling_base = self._scaling_base[idx]
        rotation_base = self._rotation_base[idx]
        opacity = torch.zeros_like(self._opacity[idx])

        self._xyz_d = nn.Parameter(points.requires_grad_(True))
        self._features_dc_d = nn.Parameter(features_dc.contiguous().requires_grad_(True))
        self._features_rest_d = nn.Parameter(features_rest.contiguous().requires_grad_(True))
        self._scaling_base_d = nn.Parameter(scaling_base.requires_grad_(True))
        self._scaling_d = self._scaling_base
        self._rotation_base_d = nn.Parameter(rotation_base.requires_grad_(True))
        self._rotation_d = self._rotation_base
        self._opacity_d = nn.Parameter(opacity.requires_grad_(True))
        # self._opacity = nn.Parameter(self._opacity.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self._xyz_d.shape[0]), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._features_rest0 = self._features_rest.clone()
        self._opacity0 = self._opacity.clone()
        self.d_idx = idx.to(self._opacity0.device)

    
    def update_xyz_rot_scale(self, points, rot_delta, scale_coeff):
        self._xyz = points
        self._rotation = quatProduct_batch(self._rotation_base, rot_delta)
        self._scaling = self._scaling_base * scale_coeff

    def update_xyz(self, points):
        self._xyz = points
        self._rotation = self._rotation_base
        self._scaling = self._scaling_base

    def update_xyz_feature(self, points, features):
        self._xyz = points
        self._features_dc = features
        self._rotation = self._rotation_base
        self._scaling = self._scaling_base

    # def update_xyz_feature_pc(self, points, features):
    #     self._xyz = points
    #     self._features_dc = features
    #     self._rotation = self._rotation_base
    #     dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
    #     self._scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     self._opacity = torch.ones_like(self._opacity)

    def update_xyz_feature_pc(self, points, features, mask=None, mask1=None):
        if mask is None:
            self._xyz = points
            self._features_dc = features
            self._features_rest = torch.zeros_like(self._features_rest0)
            self._rotation = self._rotation_base
            dist2 = torch.clamp_min(knn_points(points[None,...], points[None,...], K=1).dists.squeeze()**2, 0.0000001)
            # dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
            self._scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            self._opacity = torch.ones_like(self._opacity)
        else:
            if mask1 is None:
                self._xyz = points[mask]
                self._features_dc = features[mask]
                self._features_rest = torch.zeros_like(self._features_rest0)[mask]
                self._rotation = self._rotation_base[mask]
                dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
                self._scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)[mask]
                self._opacity = torch.ones_like(self._opacity)[mask]
            else:
                self._xyz = points[mask][mask1]
                self._features_dc = features[mask][mask1]
                self._features_rest = torch.zeros_like(self._features_rest0)[mask][mask1]
                self._rotation = self._rotation_base[mask][mask1]
                dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
                self._scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)[mask][mask1]
                self._opacity = torch.ones_like(self._opacity)[mask][mask1]

    def update_xyz_rot_scale_feature(self, points, rot_delta, scale_coeff, features):
        self._xyz = points
        self._features_dc = features
        self._rotation = quatProduct_batch(self._rotation_base, rot_delta)
        self._scaling = self._scaling_base * scale_coeff

    # def update_everything_cat(self, points, rot_delta, scale_coeff, features):
    #     xyz_d = self._xyz_d + points[self.d_idx.squeeze()]    # # potentially do it for the rest
    #     self._xyz = torch.cat((points, xyz_d), dim=0)
    #     self._features_dc = torch.cat((features, self._features_dc_d), dim=0)
    #     self._rotation = torch.cat((quatProduct_batch(self._rotation_base, rot_delta), self._rotation_base_d), dim=0)
    #     self._scaling = torch.cat((self._scaling_base * scale_coeff, self._scaling_base_d), dim=0)
    #     self._features_rest = torch.cat((self._features_rest0, self._features_rest_d), dim=0)
    #     self._opacity = torch.cat((self._opacity0, self._opacity_d), dim=0)

    # def update_everything_cat(self, points, rot_delta, scale_coeff, features, features_rest_deform):
    #     self.N0 = points.shape[0]
    #     xyz_d = self._xyz_d + points[self.d_idx.squeeze()]
    #     # feature_dc_d = self._features_dc_d + self._features_dc[:self.N0][self.d_idx.squeeze()]
    #     features_dc_d = self._features_dc_d + features[self.d_idx.squeeze()]
    #     features_rest_d = self._features_rest_d + self._features_rest[:self.N0].clone().detach()[self.d_idx.squeeze()]

    #     scaling_base = self._scaling_base.clone().detach() * scale_coeff
    #     scaling_base_d = self._scaling_base_d + scaling_base[self.d_idx.squeeze()]         # # plus?
        
    #     rotation_base = quatProduct_batch(self._rotation_base.clone().detach(), rot_delta)
    #     rotation_base_d = self._rotation_base_d + rotation_base[self.d_idx.squeeze()]      # # plus?
        
    #     self._xyz = torch.cat((points, xyz_d), dim=0)
    #     self._features_dc = torch.cat((features, features_dc_d), dim=0)
    #     self._features_rest = torch.cat((self._features_rest[:self.N0], features_rest_d), dim=0)
    #     self._rotation = self._rotation = torch.cat((rotation_base, rotation_base_d), dim=0)
    #     self._scaling = self._scaling = torch.cat((scaling_base, scaling_base_d), dim=0)
    #     self._opacity = torch.cat((self._opacity0, self._opacity_d), dim=0)

    # def update_everything_cat(self, points, rot_delta, scale_coeff, features, features_rest_deform, opacity, SR=True):

    #     self.N0 = points.shape[0]
    #     xyz_d = self._xyz_d + points[self.d_idx.squeeze()]
    #     features_dc_d = self._features_dc_d + features[self.d_idx.squeeze()]
    #     opacity_d = self._opacity_d + opacity[self.d_idx.squeeze()]

    #     features_rest = self._features_rest0[:self.N0].clone().detach()
    #     # features_rest = self._features_rest[:self.N0].clone().detach() + features_rest_deform
    #     # features_rest = torch.zeros_like(self._features_rest[:self.N0]) + features_rest_deform * 0.1
    #     features_rest_d = self._features_rest_d + features_rest[self.d_idx.squeeze()]

    #     if SR:
    #         scaling_base = self._scaling_base.clone().detach() * scale_coeff
    #     else:
    #         scaling_base = self._scaling_base.clone().detach()
    #     scaling_base_d = self._scaling_base_d + scaling_base[self.d_idx.squeeze()]         # # plus?
        
    #     if SR:
    #         rotation_base = quatProduct_batch(self._rotation_base.clone().detach(), rot_delta)
    #     else:
    #         rotation_base = self._rotation_base.clone().detach()
    #     rotation_base_d = self._rotation_base_d + rotation_base[self.d_idx.squeeze()]      # # plus?

    #     # opacity = self._opacity[:self.N0].clone().detach()
    #     # opacity_d = self._opacity_d + opacity[self.d_idx.squeeze()]

    #     # opacity = self._opacity[:self.N0].clone().detach()
    #     # opacity_d = opacity[self.d_idx.squeeze()]

    #     # # opacity_deform_d = self._opacity_d + opacity_deform[self.d_idx.squeeze()]
    #     # opacity_deform_d = opacity_deform[self.d_idx.squeeze()]
        
    #     self._xyz = torch.cat((points, xyz_d), dim=0)
    #     self._features_dc = torch.cat((features, features_dc_d), dim=0)
    #     self._features_rest = torch.cat((features_rest, features_rest_d), dim=0)
    #     self._rotation = torch.cat((rotation_base, rotation_base_d), dim=0)
    #     self._scaling = torch.cat((scaling_base, scaling_base_d), dim=0)
    #     self._opacity = torch.cat((opacity, opacity_d), dim=0)
    #     # self.opacity_deform = torch.cat((opacity_deform, opacity_deform_d), dim=0)
    #     # # print(self.opacity_deform)

    def update_everything_cat(self, points, rot_delta, scale_coeff, features, features_rest_deform, opacity, SR=True):

        # features_rest = self._features_rest0[:self.N0].clone().detach().to(points.device)
        features_rest = self._features_rest0[:points.shape[0]].clone().detach().to(points.device)

        scaling_base = self._scaling_base[:points.shape[0]].clone().detach() * scale_coeff
        rotation_base = quatProduct_batch(self._rotation_base[:points.shape[0]].clone().detach(), rot_delta)
        # scaling_base = self._scaling_base.clone().detach()[:points.shape[0]].clone().detach() * scale_coeff
        # rotation_base = quatProduct_batch(self._rotation_base.clone().detach()[:points.shape[0]].clone().detach(), rot_delta)
        # scaling_base = self._scaling_base.clone().detach()
        # rotation_base = self._rotation_base.clone().detach()
        
        self._xyz = points
        self._features_dc = features
        self._features_rest = features_rest
        self._rotation = rotation_base
        self._scaling = scaling_base
        self._opacity = opacity

    # def update_everything_cat_dense(self, seg_mask, points, rot_delta, scale_coeff, features, features_rest_deform, opacity, SR=True):

    #     features_rest = self._features_rest0.clone().detach().to(points.device)

    #     # scaling_base = self._scaling_base
    #     # scaling_base_d = self._scaling_base_d
        
    #     # rotation_base = self._rotation_base
    #     # rotation_base_d = self._rotation_base_d
    #     if SR:
    #         scaling_base = self._scaling_base * scale_coeff
    #     else:
    #         scaling_base = self._scaling_base.clone().detach()
        
    #     if SR:
    #         rotation_base = quatProduct_batch(self._rotation_base, rot_delta)
    #     else:
    #         rotation_base = self._rotation_base.clone().detach()

    #     # self._xyz = torch.cat((points, xyz_d), dim=0)
    #     # self._features_dc = torch.cat((features, features_dc_d), dim=0)
    #     # self._features_rest = torch.cat((features_rest, features_rest_d), dim=0)
    #     # self._xyz = points
    #     # self._features_dc = features
    #     self._xyz = torch.cat((points, points_d), dim=0)
    #     self._features_dc = torch.cat((features, self._features_dc_d), dim=0)
    #     self._features_rest = features_rest
    #     if SR:
    #         self._rotation = quatProduct_batch(torch.cat((rotation_base, rotation_base_d), dim=0), rot_delta)
    #         self._scaling = torch.cat((scaling_base, scaling_base_d), dim=0) * scale_coeff
    #     else:
    #         self._rotation = torch.cat((rotation_base.clone().detach(), rotation_base_d), dim=0)
    #         self._scaling = torch.cat((scaling_base.clone().detach(), scaling_base_d), dim=0)
    #     # self._opacity = torch.cat((opacity, opacity_d), dim=0)
    #     self._opacity = opacity

    def update_everything_cat_dense(self, seg_mask, points, rot_delta, scale_coeff, features, features_rest_deform, opacity, SR=True):

        features_rest = self._features_rest0.clone().detach().to(points.device)
        points_d  = points[seg_mask]
        opacity_d = opacity[seg_mask]

        if SR:
            scaling_base = self._scaling_base * scale_coeff
            rotation_base = quatProduct_batch(self._rotation_base, rot_delta)
        else:
            scaling_base = self._scaling_base.clone().detach()
            rotation_base = self._rotation_base.clone().detach()

        # scale_coeff_d = scale_coeff[seg_mask]
        # rot_delta_d = rot_delta[seg_mask]
        # scaling_base_d = self._scaling_base_d * scale_coeff_d
        # rotation_base_d = quatProduct_batch(self._rotation_base_d, rot_delta_d)
        scaling_base_d = self._scaling_base_d
        rotation_base_d = self._rotation_base_d
            
        self._xyz = torch.cat((points, points_d), dim=0)
        self._features_dc = torch.cat((features, self._features_dc_d), dim=0)
        self._features_rest = features_rest
        self._rotation = torch.cat((rotation_base, rotation_base_d), dim=0)
        self._scaling = torch.cat((scaling_base, scaling_base_d), dim=0)
        self._opacity = torch.cat((opacity, opacity_d), dim=0)

    def update_dense_cat_dense(self, seg_mask, points, rot_delta, scale_coeff, features, features_rest_deform, opacity, SR=True):

        features_rest = torch.cat((self._features_rest0[:points.shape[0]][seg_mask].clone().detach().to(points.device), self._features_rest0[:points.shape[0]][seg_mask].clone().detach().to(points.device)), dim=0)
        points_d  = points[seg_mask]
        opacity_d = opacity[seg_mask]

        if SR:
            scaling_base = self._scaling_base * scale_coeff
            rotation_base = quatProduct_batch(self._rotation_base, rot_delta)
        else:
            scaling_base = self._scaling_base.clone().detach()
            rotation_base = self._rotation_base.clone().detach()

        # scale_coeff_d = scale_coeff[seg_mask]
        # rot_delta_d = rot_delta[seg_mask]
        # scaling_base_d = self._scaling_base_d * scale_coeff_d
        # rotation_base_d = quatProduct_batch(self._rotation_base_d, rot_delta_d)
        scaling_base_d = self._scaling_base_d
        rotation_base_d = self._rotation_base_d
            
        self._xyz = torch.cat((points[seg_mask], points_d), dim=0)
        self._features_dc = torch.cat((features[seg_mask], self._features_dc_d), dim=0)
        self._features_rest = features_rest
        self._rotation = torch.cat((rotation_base[seg_mask], rotation_base_d), dim=0)
        self._scaling = torch.cat((scaling_base[seg_mask], scaling_base_d), dim=0)
        self._opacity = torch.cat((opacity[seg_mask], opacity_d), dim=0)

    def update_dense_only(self, points):
        self._xyz = points
        self._features_dc = self._features_dc_d
        self._features_rest = self._features_rest0[-points.shape[0]:]
        self._rotation = self._rotation_base_d
        self._scaling = self._scaling_base_d
        self._opacity = self._opacity[-points.shape[0]:]
        


    def update_everything_partial(self, points, rot_delta, scale_coeff, features, features_rest_deform, opacity, mask):

        self.N0 = points.shape[0]
        features_rest = self._features_rest0[:self.N0].clone().detach()
        scaling_base = self._scaling_base.clone().detach() * scale_coeff
        rotation_base = quatProduct_batch(self._rotation_base.clone().detach(), rot_delta)
        
        self._xyz = points[mask]
        self._features_dc = features[mask]
        self._features_rest = features_rest[mask]
        self._rotation = rotation_base[mask]
        self._scaling = scaling_base[mask]
        self._opacity = opacity[mask]

    def update_everything(self, points, rot_delta, scale_coeff, features, _scaling_base, _rotation_base, _features_rest, _opacity):
        self._xyz = points
        self._features_dc = features
        self._rotation = quatProduct_batch(_rotation_base, rot_delta)
        self._scaling = _scaling_base * scale_coeff
        self._features_rest = _features_rest
        self._opacity = _opacity

    def update_sh(self, Dmodel):
        # self._features_dc = torch.rand_like(self._features_dc)
        self._features_rest = torch.rand_like(self._features_rest)

        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._features_dc[Dmodel.uv_nose_idx] = 0.1
        self._features_rest[Dmodel.uv_nose_idx] = 0.1

        self._features_dc[Dmodel.uv_lips_idx] = 0.5
        self._features_rest[Dmodel.uv_lips_idx] = 0.5
        
    def training_setup(self, training_args):
        self.N0 = self._xyz.shape[0]
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.N0, 1), device=self.device)
        self.denom = torch.zeros((self.N0, 1), device=self.device)

        l = [
            {'params': [self._scaling_base], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation_base], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def training_setup_deform(self, training_args, test=False):
        self.N0 = self._xyz.shape[0]
        self.percent_dense = training_args.percent_dense
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.xyz_gradient_accum = torch.zeros((self._xyz_d.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.N0, 1), device="cuda")
        self.denom = torch.zeros((self.N0, 1), device="cuda")
        # l = [{'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"}]
        # l = [{'params': [self.dummy_param], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"}]
        l = [
            {'params': [self._xyz_d], 'lr': training_args.position_lr_init, "name": "xyz"},
            {'params': [self._features_dc_d], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest_d], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity_d], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity_base"},
            {'params': [self._scaling_base_d], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation_base_d], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._scaling_base], 'lr': training_args.scaling_lr, "name": "scaling_base"},
            {'params': [self._rotation_base], 'lr': training_args.rotation_lr, "name": "rotation_base"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def reset_opacity(self):
    #     opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]

    def reset_opacity(self):
        opacity   = self.opacity_activation(self._opacity[:self.N0])
        opacity_d = self.opacity_activation(self._opacity_d)

        opacities_d_new = inverse_sigmoid(torch.min(opacity_d, torch.ones_like(self._opacity_d)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_d_new, "opacity")
        self._opacity_d = optimizable_tensors["opacity"]

        opacities_new = inverse_sigmoid(torch.min(opacity, torch.ones_like(self._opacity[:self.N0])*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity_base")
        self._opacity = optimizable_tensors["opacity_base"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] not in ['scaling_base', 'rotation_base', 'opacity_base', 'rigid_R', 'rigid_S', 'rigid_T']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]
        self._xyz_d = optimizable_tensors["xyz"]
        self._features_dc_d = optimizable_tensors["f_dc"]
        self._features_rest_d = optimizable_tensors["f_rest"]
        self._opacity_d = optimizable_tensors["opacity"]
        self._scaling_d = optimizable_tensors["scaling"]
        self._rotation_d = optimizable_tensors["rotation"]

        # # since i dont prune base points
        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.denom = self.denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] not in ['scaling_base', 'rotation_base', 'opacity_base', 'rigid_R', 'rigid_S', 'rigid_T']:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        self._xyz_d = optimizable_tensors["xyz"]
        self._features_dc_d = optimizable_tensors["f_dc"]
        self._features_rest_d = optimizable_tensors["f_rest"]
        self._opacity_d = optimizable_tensors["opacity"]
        self._scaling_base_d = optimizable_tensors["scaling"]
        self._rotation_base_d = optimizable_tensors["rotation"]

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        N = self.N0
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self.max_radii2D = torch.zeros((N), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # new_xyz = self._xyz[:self.N0][selected_pts_mask]
        new_xyz = torch.zeros_like(self._xyz[:self.N0][selected_pts_mask])
        
        # new_features_dc = self._features_dc[:self.N0][selected_pts_mask]
        # new_features_rest = self._features_rest[:self.N0][selected_pts_mask]
        # new_scaling = self._scaling[:self.N0][selected_pts_mask]
        # new_rotation = self._rotation[:self.N0][selected_pts_mask]
        new_features_dc   = torch.zeros_like(self._features_dc[:self.N0][selected_pts_mask])
        new_features_rest = torch.zeros_like(self._features_rest[:self.N0][selected_pts_mask])
        new_scaling       = torch.zeros_like(self._scaling[:self.N0][selected_pts_mask])
        new_rotation      = torch.zeros_like(self._rotation[:self.N0][selected_pts_mask])
        new_opacities     = torch.zeros_like(self._opacity[:self.N0][selected_pts_mask])
        # new_opacities     = self._opacity[:self.N0][selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self.d_idx = torch.cat((self.d_idx, torch.nonzero(selected_pts_mask)), dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        # print(self._xyz_d.shape)
        # print(self._features_dc_d.shape)
        # print(self._features_rest_d.shape)
        # print(self._scaling_base_d.shape)
        # print(self._rotation_base_d.shape)
        # print(self._opacity_d.shape)

        # # opacity = self.opacity_activation(torch.cat((self._opacity, self._opacity_d), dim=0))
        # # print(opacity.shape, self._opacity.shape, self._xyz.shape, self._features_dc.shape)
        # # prune_mask = (opacity < min_opacity).squeeze()[self.N0:]
        # # # prune_mask[:self.N0] = False
        # print(self._opacity_d.shape, self._opacity[self.N0:].shape)
        # opacity = self._opacity[:self.N0].clone().detach()
        # opacity_d = self.opacity_activation(self._opacity_d + opacity[self.d_idx.squeeze()])
        # print(self._opacity_d, self._opacity)
        # print(opacity_d, self.opacity_activation(self._opacity))
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        # print('prune: ', self.opacity_deform.quantile(0.5), self.opacity_deform.min(), self.opacity_deform.max(), prune_mask.sum(), prune_mask.shape)

        # # if max_screen_size:
        # #     big_points_vs = self.max_radii2D > max_screen_size
        # #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print(viewspace_point_tensor.grad.shape)
        # print(torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True))
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:self.N0][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_dist(self, dist):
        self.xyz_gradient_accum += dist[:self.N0]
        self.denom += 1

    def add_densification_stats_MLP(self, v, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(v.grad.squeeze()[:self.N0][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1