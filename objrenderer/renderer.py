import torch
from torch import nn
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    AmbientLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
    BlendParams
)
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
from pytorch3d.io import load_objs_as_meshes, load_obj
import torchvision
from torchvision.utils import save_image
import os
import json
from PIL import Image
import random
from tqdm import tqdm
from scene.cameras import Camera
from utils.general_utils import process_p3d_camera, positional_encoding, build_rotation
from utils.loss_utils import huber_loss

from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
)



class OBJRenderer:
    def __init__(self, device, mesh_path=None, img_res=512, dist=2.25, d_pe=64, normalize_mesh=False, normalize_scale=False, s=1, tx=0, ty=0, tz=0):
        self.device = device
        self.img_res = img_res
        self.d_pe = d_pe
        if mesh_path is not None:
            self.mesh = load_objs_as_meshes([mesh_path], device=self.device)
        else:
            self.mesh = None
        self.raster_settings = RasterizationSettings(
            image_size=self.img_res,
            blur_radius=0.0,
            # blur_radius=1e-4,
            faces_per_pixel=1,
            # faces_per_pixel=10,
            # max_faces_per_bin=10000,
            # bin_size=0,
        )
        self.lights = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))
        self.dist = dist

        if normalize_mesh:
            verts = self.mesh.verts_packed()
            verts = verts - verts.mean(dim=0, keepdim=True)
            max_dist = torch.cdist(verts, verts).max()
            if normalize_scale:
                verts = verts / max_dist * s
            else:
                verts = verts * s
            translation = torch.tensor([tx, ty, tz], device=self.device)
            verts = verts + translation
            self.mesh = self.mesh.update_padded(verts.unsqueeze(0).to(self.mesh.device))

    def batch_positional_encoding(self, N, d):
        position = torch.arange(0, N, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(N, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def random_position(self):
        # cam_dist = random.uniform(self.dist*0.5, self.dist*1)
        cam_dist = 2.25
        elev = random.uniform(-90, 90)
        azim = random.uniform(0, 360)
        return cam_dist, elev, azim

    def directional_light_from_camera(self, R):
        cam_dir = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)  # camera looks along -Z
        return torch.bmm(R, cam_dir[..., None])[..., 0]    # # (1, 3, 3) @ (1, 3, 1)

    def render_mesh(self, mesh, background, cam_dist=None, elev=None, azim=None, light_type='ambient', light_dir=[1.0, 1.0, 1.0], ambient=[0.2, 0.2, 0.2], diffuse=[0.8, 0.8, 0.8], specular=[0.0, 0.0, 0.0]):
        if cam_dist is not None and elev is not None and azim is not None:
            pass
        else:    
            cam_dist, elev, azim = self.random_position()
        R, T = look_at_view_transform(cam_dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        if light_type == 'directional':
            light_direction = self.directional_light_from_camera(R.to(self.device))
            # light_direction = torch.tensor([light_dir], device=self.device)  # Example direction
            # light_direction = torch.tensor([[-1.0, -1.0, -1.0]], device=self.device)  # Example direction
            ambient_color = torch.tensor([ambient], device=self.device)  # Soft ambient light
            diffuse_color = torch.tensor([diffuse], device=self.device)  # Bright diffuse light
            specular_color = torch.tensor([specular], device=self.device)  # Subtle specular highlights
            light = DirectionalLights(
                device=self.device,
                direction=light_direction,
                ambient_color=ambient_color,
                diffuse_color=diffuse_color,
                specular_color=specular_color
            )
        else:
            light = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))

        # raster_settings = RasterizationSettings(
        #     image_size=self.img_res*4,
        #     blur_radius=1e-4,
        #     faces_per_pixel=10
        # )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
                # raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=light,
                blend_params=BlendParams(background_color=background)
                # blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=background)
            )
        )
        return renderer(mesh)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()

        # image = renderer(mesh)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()
        # return torchvision.transforms.Resize((512, 512))(image)

    def render_mesh_orth(self, mesh, background, cam_dist=None, elev=None, azim=None, light_type='ambient', light_dir=[1.0, 1.0, 1.0], ambient=[0.2, 0.2, 0.2], diffuse=[0.8, 0.8, 0.8], specular=[0.0, 0.0, 0.0]):
        if cam_dist is not None and elev is not None and azim is not None:
            pass
        else:    
            cam_dist, elev, azim = self.random_position()
        R, T = look_at_view_transform(cam_dist, elev, azim)
        cameras = OrthographicCameras(device=self.device, R=R, T=T)

        light = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
                # raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=light,
                blend_params=BlendParams(background_color=background)
                # blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=background)
            )
        )
        return renderer(mesh)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()

    def render_pc(self, points, features, background, cam_dist=None, elev=None, azim=None):
        if cam_dist is not None and elev is not None and azim is not None:
            pass
        else:    
            cam_dist, elev, azim = self.random_position()
        R, T = look_at_view_transform(cam_dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        point_cloud = Pointclouds(points=points, features=features)

        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.01,   # Radius of each point in the image plane
            points_per_pixel=10,
        )
        # The renderer
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        compositor = AlphaCompositor(background_color=background)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
        # images = renderer(point_cloud)
        return renderer(point_cloud)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()
    
    def render(self, background, t=0, cam_dist=None, elev=None, azim=None, light_type='ambient', ambient=[0.2, 0.2, 0.2], diffuse=[0.8, 0.8, 0.8], specular=[0.0, 0.0, 0.0]):
        if cam_dist is not None and elev is not None and azim is not None:
            pass
        elif cam_dist is not None:
            _, elev, azim = self.random_position()
        else:    
            cam_dist, elev, azim = self.random_position()
        R, T = look_at_view_transform(cam_dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # x = torch.randn(3)
        # # x = torch.rand(3) * 2 - 1
        # light_direction = (x / x.norm())[None].to(self.device)
        light_direction = self.directional_light_from_camera(R.to(self.device))
        self.light_dir = light_direction

        if light_type == 'directional':
            # light_direction = torch.tensor([light_dir], device=self.device)  # Example direction
            # light_direction = torch.tensor([[-1.0, -1.0, -1.0]], device=self.device)  # Example direction
            ambient_color = torch.tensor([ambient], device=self.device)  # Soft ambient light
            diffuse_color = torch.tensor([diffuse], device=self.device)  # Bright diffuse light
            specular_color = torch.tensor([specular], device=self.device)  # Subtle specular highlights
            light = DirectionalLights(
                device=self.device,
                direction=light_direction,
                ambient_color=ambient_color,
                diffuse_color=diffuse_color,
                specular_color=specular_color
            )
        else:
            light = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                # lights=self.lights,
                lights=light,
                blend_params=BlendParams(background_color=background)
            )
        )
        image = renderer(self.mesh)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()

        fov = cameras.fov
        intrinsic_matrix = cameras.get_projection_transform().get_matrix().squeeze(0)
        extrinsic_matrix = cameras.get_world_to_view_transform().get_matrix().squeeze(0)
        intrinsic, FoVx, FoVy = process_p3d_camera(fov, intrinsic_matrix)

        head_mask = torch.zeros((1, self.img_res, self.img_res), dtype=torch.float, device=self.device)
        mouth_mask = torch.zeros((1, self.img_res, self.img_res), dtype=torch.float, device=self.device)

        t_pe = positional_encoding(t, self.d_pe, self.device)
        elev_pe = positional_encoding(elev, self.d_pe, self.device)
        azim_pe = positional_encoding(azim, self.d_pe, self.device)
        cam_dist_pe = positional_encoding(cam_dist, self.d_pe, self.device)
        pe = torch.cat((t_pe, elev_pe, azim_pe, cam_dist_pe), dim=1)

        camera = Camera(colmap_id=None, image=image, 
                        FoVx=FoVx, FoVy=FoVy,
                        proj=intrinsic, w2v=extrinsic_matrix, full_proj=None, 
                        head_mask=head_mask, mouth_mask=mouth_mask,
                        image_name=None, uid=0, uid_pe=pe,
                        data_device=self.device,
                        cam_dist=cam_dist, elev=elev, azim=azim
                        )
        return camera

        
    def render_mesh_with_cameras_from_opencv(self, mesh, background, K, R, T, light_type='ambient', light_dir=[1.0, 1.0, 1.0], ambient=[0.2, 0.2, 0.2], diffuse=[0.8, 0.8, 0.8], specular=[0.0, 0.0, 0.0]):
        """
        Convert OpenCV camera intrinsics (K) and extrinsics (RT) back to PyTorch3D PerspectiveCameras.
        Assumes K is (B, 3, 3) and RT is (B, 3, 4).
        """
        # # Extract rotation (R) and translation (T)
        # R_opencv = RT[:, :3, :3]  # (B, 3, 3)
        # T_opencv = RT[:, :3, 3]   # (B, 3)

        R_opencv = R
        T_opencv = T

        # Convert OpenCV camera rotation to PyTorch3D format (if needed)
        R_pytorch3d = R_opencv.transpose(1, 2)  # OpenCV to PyTorch3D conversion
        # R_pytorch3d = R_opencv

        # Convert OpenCV translation (world-to-camera) to PyTorch3D (camera-to-world)
        T_pytorch3d = -torch.bmm(R_pytorch3d, T_opencv.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # Extract focal lengths
        focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1)  # (B, 2)

        # Extract principal points
        principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=-1)  # (B, 2)

        # Create PyTorch3D cameras
        cameras = PerspectiveCameras(
            R=R_pytorch3d, T=T_opencv, focal_length=focal_length, principal_point=principal_point
        ).to(self.device)

        if light_type == 'directional':
            light_direction = torch.tensor([light_dir], device=self.device)  # Example direction
            ambient_color = torch.tensor([ambient], device=self.device)  # Soft ambient light
            diffuse_color = torch.tensor([diffuse], device=self.device)  # Bright diffuse light
            specular_color = torch.tensor([specular], device=self.device)  # Subtle specular highlights
            light = DirectionalLights(
                device=self.device,
                direction=light_direction,
                ambient_color=ambient_color,
                diffuse_color=diffuse_color,
                specular_color=specular_color)
        else:
            light = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=light,
                blend_params=BlendParams(background_color=background)
            )
        )
        return renderer(mesh)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()


    def render_mesh_mica(self, mesh, background, R, t, fl, pp, light_type='ambient', light_dir=[1.0, 1.0, 1.0], ambient=[0.2, 0.2, 0.2], diffuse=[0.8, 0.8, 0.8], specular=[0.0, 0.0, 0.0]):

        cameras = PerspectiveCameras(
                device=self.device,
                principal_point=pp,
                focal_length=fl,
                R=rotation_6d_to_matrix(R), T=t,
                image_size=self.img_res
            )

        if light_type == 'directional':
            light_direction = torch.tensor([light_dir], device=self.device)  # Example direction
            ambient_color = torch.tensor([ambient], device=self.device)  # Soft ambient light
            diffuse_color = torch.tensor([diffuse], device=self.device)  # Bright diffuse light
            specular_color = torch.tensor([specular], device=self.device)  # Subtle specular highlights
            light = DirectionalLights(
                device=self.device,
                direction=light_direction,
                ambient_color=ambient_color,
                diffuse_color=diffuse_color,
                specular_color=specular_color)
        else:
            light = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=light,
                blend_params=BlendParams(background_color=background)
            )
        )
        return renderer(mesh)[:, :, :, :3].permute(0, 3, 1, 2).squeeze()


    def centralize_mesh_vertices(self, mesh=None):
        if mesh is None:
            vertices = self.mesh.verts_packed()
            transformed_vertices = vertices - vertices.mean(dim=0, keepdim=True)
            self.mesh = self.mesh.update_padded(transformed_vertices.unsqueeze(0))
        else:
            vertices = mesh.verts_packed()
            transformed_vertices = vertices - vertices.mean(dim=0, keepdim=True)
            return mesh.update_padded(transformed_vertices.unsqueeze(0))


    def optimize_camera(self, target_image, mask):
        cam_dist = torch.tensor([2.25], dtype=torch.float, requires_grad=True, device=self.device)
        elev = torch.tensor([0.0], dtype=torch.float, requires_grad=True, device=self.device)
        azim = torch.tensor([0.0], dtype=torch.float, requires_grad=True, device=self.device)
        at = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float, requires_grad=True, device=self.device)
        fov = torch.tensor([60.0], dtype=torch.float, requires_grad=True, device=self.device)

        # optimizer = torch.optim.Adam([cam_dist, elev, azim, at], lr=0.1)
        # optimizer = torch.optim.Adam([cam_dist, fov], lr=0.1)
        optimizer = torch.optim.Adam([
            {'params': [cam_dist], 'lr': 0.001},  # smaller learning rate for cam_dist
            {'params': [fov], 'lr': 0.01}         # larger learning rate for fov
        ])


        num_iters = 10000

        for i in tqdm(range(num_iters)):
            optimizer.zero_grad()

            # Recompute R, T and update camera and renderer
            R, T = look_at_view_transform(dist=cam_dist, elev=elev, azim=azim, at=at)
            # cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
            cameras = OrthographicCameras(device=self.device, R=R, T=T)

            background = torch.rand(3, dtype=torch.float, device=self.device)
            
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=self.raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=self.lights,
                    blend_params=BlendParams(background_color=background)
                )
            )

            images = renderer(self.mesh)[:, 112:400, :, :3].permute(0, 3, 1, 2).squeeze()

            # gt = target_image*(1-mask)+background.view(3, 1, 1)*mask
            gt = target_image*mask+background.view(3, 1, 1)*(1-mask)

            # images = images * mask
            # gt = target_image * mask
            
            # loss = huber_loss(images, gt, 0.1)
            loss = (images - gt).abs().mean()
            loss.backward()
            optimizer.step()

            os.makedirs('opt-cam', exist_ok=True)

            if i % 100 == 0:
            # if True:
                print(f"Iter {i:03d} | Loss: {loss.item():.4f} | Dist: {cam_dist.item():.2f} | "
                    f"Elev: {elev.item():.2f} | Azim: {azim.item():.2f} | "
                    f"FOV: {fov.item():.2f} | LookAt: {at.tolist()[0]}")
                out = torch.cat((gt, images), dim=1)
                torchvision.utils.save_image(out, f"opt-cam/{i}.jpg")

    def create_fov_renderer(self, cam_dist, elev, azim):
        R, T = look_at_view_transform(cam_dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=60)

        background = torch.rand(3, dtype=torch.float, device=self.device)
            
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights,
                blend_params=BlendParams(background_color=background)
            )
        )
        return renderer

    def rigid_align_2_meshes(self, mesh0, mesh1, random_cam=True, cam_dist=None, elev=None, azim=None):
        
        r = nn.Parameter(torch.tensor([[1, 0, 0, 0]], dtype=torch.float, requires_grad=True, device=self.device))
        t = nn.Parameter(torch.tensor([[0, 0, 0]], dtype=torch.float, requires_grad=True, device=self.device))

        optimizer = torch.optim.Adam([
            {'params': [r, t], 'lr': 0.001},
        ])

        num_iters = 2000

        mesh = mesh1.clone().detach()

        for i in tqdm(range(num_iters)):
            optimizer.zero_grad()

            if random_cam:
                cam_dist, elev, azim = self.random_position()
            renderer = self.create_fov_renderer(cam_dist, elev, azim)

            img0 = renderer(mesh0)[..., :3].permute(0, 3, 1, 2).squeeze()

            rot = build_rotation(r)    # 1, 3, 3

            vertices = mesh1.verts_packed()
            transformed_vertices = vertices @ rot[0] + t
            # torch.bmm(self.kpts.permute(1, 0, 2), rot_kpts).permute(1, 0, 2)
            mesh = mesh.update_padded(transformed_vertices.unsqueeze(0))
            
            img = renderer(mesh)[..., :3].permute(0, 3, 1, 2).squeeze()

            # # gt = target_image*(1-mask)+background.view(3, 1, 1)*mask
            # gt = target_image*mask+background.view(3, 1, 1)*(1-mask)

            # # images = images * mask
            # # gt = target_image * mask
            
            # # loss = huber_loss(images, gt, 0.1)
            loss = (img0 - img).abs().mean()
            loss.backward()
            optimizer.step()

            os.makedirs('opt-rigid', exist_ok=True)

            if i % 100 == 0:
            # if True:
                # print(f"Iter {i:03d} | Loss: {loss.item():.4f} | Dist: {cam_dist.item():.2f} | "
                #     f"Elev: {elev.item():.2f} | Azim: {azim.item():.2f} | "
                #     f"FOV: {fov.item():.2f} | LookAt: {at.tolist()[0]}")
                with torch.no_grad():
                    # renderer = self.create_fov_renderer(4, 0, 0)
                    img0 = renderer(mesh0)[..., :3].permute(0, 3, 1, 2).squeeze()
                    img1 = renderer(mesh1)[..., :3].permute(0, 3, 1, 2).squeeze()
                    img  = renderer(mesh)[..., :3].permute(0, 3, 1, 2).squeeze()
                    out = torch.cat((img0, img, img1), dim=2)
                    torchvision.utils.save_image(out, f"opt-rigid/{i}.jpg")