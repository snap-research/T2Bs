import os, sys
import pickle
import torch
import numpy as np
from torch import nn
import math
import trimesh
import torch.nn.functional as F
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesVertex
import torchvision
from torchvision import transforms
import pytorch3d

# from flame import FLAME_mica, parse_args
from utils.general_utils import Pytorch3dRasterizer, Embedder, load_binary_pickle, a_in_b_torch, face_vertices_gen, build_rotation, rgb_to_hsv, hsv_to_rgb, uv_mesh_triangles, generate_mesh, represent_points_with_keypoints, k_means_clustering, save_obj_colorful_point_cloud, get_lm, Mahalanobis_distance
from utils.general_utils import DualQuaternion as DQB

class Deform_Model(nn.Module):
    def __init__(self, device, mesh_dir, k=10, num_clusters=5000, d_pe=64, expre_dir=None, eigen_num=None, normalize_mesh=False, normalize_scale=False, s=1, tx=0, ty=0, tz=0):
        super().__init__()
        self.device = device
        self.mesh_dir = mesh_dir
        self.k = k
        self.num_clusters = num_clusters 
        self.d_pe = d_pe
        
        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)
        
        mesh_path = os.path.join(self.mesh_dir, 'textured.obj')         # # name of the obj file
        verts, faces, aux = load_obj(mesh_path, load_textures=True)
        self.mesh = load_objs_as_meshes([mesh_path], device=self.device)

        if normalize_mesh:
            # verts = verts - verts.mean(dim=0, keepdim=True)
            # max_dist = torch.cdist(verts, verts).max()
            # if normalize_scale:
            #     verts = verts / max_dist * s
            # else:
            #     verts = verts * s
            # translation = torch.tensor([tx, ty, tz])
            # verts = verts + translation
            # self.mesh = self.mesh.update_padded(verts.unsqueeze(0).to(self.mesh.device))

            verts = self.normalize_like_trimesh_batched(verts[None, None])[0, 0]
            self.mesh = self.mesh.update_padded(verts.unsqueeze(0).to(self.mesh.device))

        self.geometry_shape = verts.to(self.device)
        self.faces_idx = faces.verts_idx.to(self.device)
        self.texture_map_raw = next(iter(aux.texture_images.values())).to(self.device)
        self.texture_map = transforms.Resize((512, 512))(self.texture_map_raw.permute(2, 0, 1))
        torchvision.utils.save_image(self.texture_map, 'texture.jpg')       # # # # name of the texture map
        uv_coords = aux.verts_uvs[None, ...]
        uv_coords = uv_coords * 2 - 1
        uv_coords[..., 1] = - uv_coords[..., 1]
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)
        self.uvfaces = faces.textures_idx[None, ...].to(self.device)
        self.tri_faces = faces.verts_idx[None, ...].to(self.device)

        uv = aux.verts_uvs[None, ...].to(self.device) * 2 - 1
        uv[..., 1] = -uv[..., 1]
        texture = self.texture_map_raw.permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        grid = uv.unsqueeze(2)  # [1, V, 1, 2]
        colors = F.grid_sample(texture, grid, align_corners=False)  # [1, 3, V, 1]
        self.verts_colors = colors.squeeze().T

        self.init_networks()

    def normalize_like_trimesh_batched(self, verts: torch.Tensor, eps: float = 1e-12, return_c_s=False):
        """
        verts: [B, T, N, 3] (e.g. [1,1,N,3])
        normalize each [N,3] independently per (B,T).
        """
        min_bound = verts.amin(dim=2, keepdim=True)   # [B,T,1,3]
        max_bound = verts.amax(dim=2, keepdim=True)   # [B,T,1,3]

        center = (min_bound + max_bound) * 0.5        # [B,T,1,3]
        scale  = (max_bound - min_bound).amax(dim=-1, keepdim=True)  # [B,T,1,1] max over xyz

        verts_norm = (verts - center) / (scale + eps)

        if return_c_s:
            return verts_norm, center, scale
        else:
            return verts_norm

    def init_networks(self, eigen_num=1):
        kp_dim = 3 * eigen_num
        self.KPdeformNet = MLP(
            input_dim=kp_dim*len(self.pts_embedder.embed_fns)+4*self.d_pe, 
            output_dim=10, 
            hidden_dim=256,  
            hidden_layers=6 
        )
        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded+4*self.d_pe,
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6    # 1
        )
        self.deformNet_features = MLP(
            input_dim=self.pts_embedder.dim_embeded+4*self.d_pe,
            output_dim=3,
            hidden_dim=256,
            hidden_layers=6
        )
        self.rigidNet = MLP(
            input_dim=256,
            output_dim=8,
            hidden_dim=256,
            hidden_layers=6
        )

        _cov_r = torch.zeros((self.num_clusters, 4), dtype=torch.float, device=self.device)
        _cov_r[:, 0] = 1
        self._cov_r = torch.nn.Parameter(_cov_r.requires_grad_(True))

        _cov_s = torch.ones((self.num_clusters, 3), dtype=torch.float, device=self.device)
        self._cov_s = torch.nn.Parameter(_cov_s.requires_grad_(True))

        
    def example_init(self):

        self.uv_vertices_shape = self.geometry_shape[None].detach().clone()
        self.uv_vertices_shape_embeded = self.pts_embedder(self.uv_vertices_shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]

        self.uv_features_dc = self.verts_colors[None].detach().clone()
        self.uv_features_dc_embeded = self.pts_embedder(self.uv_features_dc)

        self.rigid_S_base = torch.ones((1), dtype=torch.float, device=self.device)[None, ...]
        self.rigid_R_base = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device)[None, ...]
        self.rigid_T_base = torch.zeros((3), dtype=torch.float, device=self.device)[None, ...]

        try:
            kpts_data = torch.load(os.path.join(self.mesh_dir, f'kpts_{self.k}_{self.num_clusters}.pt'))
            self.kpts_idx     = kpts_data['kpts_idx'].to(self.device)
            self.kpts         = kpts_data['kpts'].to(self.device)
            self.kpts_embeded = kpts_data['kpts_embeded'].to(self.device)
            self.knn_idx      = kpts_data['knn_idx'].to(self.device)
            self.W            = kpts_data['W'].to(self.device)
            self.cov          = kpts_data['cov'].to(self.device)
            self.cov_inv      = kpts_data['cov_inv'].to(self.device)
            self.knn_cov_inv  = kpts_data['knn_cov_inv'].to(self.device)
            print('loaded kpts data.')
        except:
            print('k-mean clustering...')
            kpts_assignments, key_points, covariance = k_means_clustering(self.uv_vertices_shape[0], self.num_clusters)

            epsilon = 1e-6
            cov_inv = torch.linalg.inv(covariance + epsilon * torch.eye(3, device=covariance.device)[None])

            self.kpts = key_points[None,...]
            self.kpts_embeded = self.pts_embedder(self.kpts)

            self.kpts_idx = knn_points(key_points[None,...], self.uv_vertices_shape, K=1).idx   # # 1, num_clusters, 1

            self.knn_idx = knn_points(self.uv_vertices_shape, key_points[None,...], K=self.k).idx
            knn_kpts = key_points[self.knn_idx][0]            # # [N, k, 3]
            knn_cov_inv = cov_inv[self.knn_idx][0]
            W = Mahalanobis_distance(self.uv_vertices_shape.permute(1, 0, 2), knn_kpts, knn_cov_inv)[None,...]
            self.W = W / W.sum(dim=2, keepdim=True)
            self.cov = covariance
            self.cov_inv = cov_inv
            self.knn_cov_inv = knn_cov_inv

            torch.save({
                'kpts_idx': self.kpts_idx,
                'kpts': self.kpts,
                'kpts_embeded': self.kpts_embeded,
                'knn_idx': self.knn_idx,
                'W': self.W,
                'cov': self.cov,
                'cov_inv': self.cov_inv,
                'knn_cov_inv': self.knn_cov_inv
            }, os.path.join(self.mesh_dir, f'kpts_{self.k}_{self.num_clusters}.pt'))

        pytorch3d.io.save_ply(os.path.join(self.mesh_dir, f'kpts_{self.num_clusters}.ply'), self.kpts[0])


    def rigid_transform(self, condition, verts=None):
        if verts is None:
            verts = self.uv_vertices_shape
        rigid = self.rigidNet(condition.unsqueeze(1))
        R = rigid[:, 0, :4]  + self.rigid_R_base
        S = rigid[:, 0, 4:5] + self.rigid_S_base
        T = rigid[:, 0, 5:]  + self.rigid_T_base
        rot = build_rotation(R)
        return S * torch.bmm(verts, rot).squeeze(-1) + T

    
    def decode(self, condition, predict_color_offset=False, test=False, condition1=None, use_LBS=True):

        # # # shape MLP
        condition_verts = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition_verts), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)

        uv_vertices_deforms = deforms[..., :3]
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)


        if use_LBS:
            condition_kpts = condition.unsqueeze(1).repeat(1, self.kpts.shape[1], 1)
            kpts_embeded_condition = torch.cat((self.kpts_embeded, condition_kpts), dim=2)
            deforms_kpts = self.KPdeformNet(kpts_embeded_condition)
            deforms_kpts = torch.tanh(deforms_kpts)

            t_kpts = deforms_kpts[..., :3]
            r_kpts_0 = deforms_kpts[..., 3:7]
            r_kpts_r = torch.exp(r_kpts_0[..., 0]).unsqueeze(-1)
            r_kpts_v = r_kpts_0[..., 1:]
            r_kpts = torch.cat((r_kpts_r, r_kpts_v), dim=-1)
            s_kpts = deforms_kpts[..., 7:]
            s_kpts = torch.exp(s_kpts)
            rot_kpts = build_rotation(r_kpts[0])
            
            kpts_deformed = s_kpts * torch.bmm(self.kpts.permute(1, 0, 2), rot_kpts).permute(1, 0, 2) + t_kpts
            self.kpts_final = self.rigid_transform(condition, kpts_deformed)  # # for visualization

            # # per-gaussian transformation for LBS and DQB
            s = s_kpts[0][self.knn_idx][0]     # # N, K, 3
            t = t_kpts[0][self.knn_idx][0]
            r = r_kpts[0][self.knn_idx][0]
            rot = rot_kpts[self.knn_idx][0]
            verts0_knn = self.uv_vertices_shape.view(-1, 1, 1, 3).repeat(1, rot.shape[1], 1, 1)   # # N, K, 1, 3
            # # LBS
            verts_knn = s * torch.bmm(verts0_knn.view(-1, 1, 3), rot.view(-1, 3, 3)).view(-1, rot.shape[1], 3) + t   # # N, K, 3
            # verts_knn = verts0_knn[:, :, 0] + t   # # N, K, 3

            # # blending weights
            # cov_s = torch.zeros((self.num_clusters, 3, 3), dtype=torch.float, device=self.device)
            # for i in range(3):
            #     cov_s[:, i, i] = self._cov_s[:, i]
            # cov_r = build_rotation(self._cov_r)
            # cov_res = torch.bmm(cov_r, cov_s)
            # cov_inv = cov_res
            # cov_inv = torch.bmm(torch.bmm(rot_kpts, cov_inv), rot_kpts.permute(0, 2, 1))
            # knn_cov_inv = cov_inv[self.knn_idx][0]
            # W = Mahalanobis_distance(self.uv_vertices_shape.permute(1, 0, 2), kpts_deformed[0][self.knn_idx][0], knn_cov_inv)[None]
            # W = W / W.sum(dim=2, keepdim=True)

            W = self.W
            verts_deformed = torch.bmm(W.permute(1, 0, 2), verts_knn).permute(1, 0, 2)

        else:
            verts_deformed = self.uv_vertices_shape + uv_vertices_deforms

        # verts_final = self.rigid_transform(condition, verts_deformed)
        verts_final = verts_deformed

        if predict_color_offset:
            uv_vertices_features_dc_embeded_condition = torch.cat((self.uv_features_dc_embeded, condition_verts), dim=2)
            deforms_features = self.deformNet_features(uv_vertices_features_dc_embeded_condition)
            deforms_features = 0.5 * torch.tanh(deforms_features)
            uv_features_deforms_vis = deforms_features[..., :3]
            features_dc_final = self.uv_features_dc + uv_features_deforms_vis
        else:
            features_dc_final = self.uv_features_dc
            
        opacity_final = torch.ones_like(verts_final)[:, :, :1]

        return verts_final, rot_delta, scale_coef, features_dc_final, verts_deformed, opacity_final

    
    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.deformNet_features.state_dict(),
            self.rigidNet.state_dict(),
            self.KPdeformNet.state_dict(),
            self.optimizer.state_dict(),
            self._cov_r,
            self._cov_s
        )

    def restore_train(self, model_args):
        try:
            try:
                (net_dict,
                net_dict_features,
                net_dict_rigid,
                opt_dict) = model_args
            except:
                (net_dict,
                net_dict_features,
                net_dict_rigid,
                net_dict_KP,
                net_dict_offset,
                net_dict_feature_vis,
                opt_dict) = model_args[:7]
        except:
            (net_dict, net_dict_features) = model_args[:2]
        self.rigidNet.load_state_dict(net_dict_rigid)
        self.training_setup()

    def restore(self, model_args):

        (net_dict,
        net_dict_features,
        net_dict_rigid,
        net_dict_KP,
        net_dict_offset,
        net_dict_feature_vis,
        opt_dict) = model_args[:7]

        try:
            self._cov_r = model_args[7]
            self._cov_s = model_args[8]
        except:
            pass

        self.deformNet.load_state_dict(net_dict)
        self.deformNet_features.load_state_dict(net_dict_features)
        self.rigidNet.load_state_dict(net_dict_rigid)
        self.KPdeformNet.load_state_dict(net_dict_KP)
        self.training_setup()

    
    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
            {'params': self.deformNet_features.parameters(), 'lr': 1e-4},   # # 1e-4
            {'params': self.KPdeformNet.parameters(), 'lr': 1e-4},
            {'params': self._cov_r, 'lr': 1e-4},
            {'params': self._cov_s, 'lr': 1e-4},
            {'params': self.rigidNet.parameters(), 'lr': 1e-4},
        ]
        # self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999), weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    # def training_setup(self):
    #     params_group = [
    #         {'params': self.deformNet.parameters(), 'lr': 1e-5},
    #         {'params': self.deformNet_features.parameters(), 'lr': 1e-5},   # # 1e-4
    #         {'params': self.KPdeformNet.parameters(), 'lr': 1e-5},
    #         {'params': self._cov_r, 'lr': 1e-5},
    #         {'params': self._cov_s, 'lr': 1e-5},
    #         {'params': self.rigidNet.parameters(), 'lr': 1e-5},
    #     ]
    #     # self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999), weight_decay=1e-4)
    #     self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    def training_setup_rigid(self):
        params_group = [
            {'params': self.rigidNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999), weight_decay=1e-4)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

        # nn.init.constant_(self.output_linear.weight, 1e-4)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output


