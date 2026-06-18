import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import cv2
import lpips
from torchvision import transforms
import torchvision
import pytorch3d
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing, mesh_edge_loss, chamfer_distance
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_objs_as_meshes, load_obj

from scene import GaussianModel, Scene
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep, verts2D, verts2D_visu, verts2D_img, arap_loss, save_obj_colorful_point_cloud
from utils.sh_utils import RGB2SH
from objrenderer.renderer import OBJRenderer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torchrun_env():
    """Works for both torchrun and normal python."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def list_identities(data_root):
    ids = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if not os.path.isdir(p):
            continue
        # only keep identities that have obj folder
        if os.path.isdir(os.path.join(p, "obj")):
            ids.append(name)

    # ids = [identity for identity in ids if identity in os.listdir('/nfs/usr/jluo2/dev2-root/out_eval-all-t9')]

    import socket
    if socket.gethostname() == 'jiahao-dev-4node-worker-0':
        ids = ids[0::4]
    elif socket.gethostname() == 'jiahao-dev-4node-worker-1':
        ids = ids[1::4]
    elif socket.gethostname() == 'jiahao-dev-4node-worker-2':
        ids = ids[2::4]
    elif socket.gethostname() == 'jiahao-dev-4node-worker-3':
        ids = ids[3::4]

    return ids


def feature_loss(image, gt_image):
    image_percep = normalize_for_percep(image)
    gt_image_percep = normalize_for_percep(gt_image)
    return torch.mean(percep_module.forward(image_percep, gt_image_percep))


def train_one_identity(args, lp, op, pp, percep_module, idname, rank=0):
    # ---- per-identity args ----
    args.idname = idname

    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    set_random_seed(args.seed + rank)  # slightly different per process

    # # dataloader
    data_dir  = os.path.join(args.data_root, args.idname)
    asset_dir = os.path.join(data_dir, 'obj')
    camera_folder = os.path.join('cameras')
    log_dir = os.path.join(data_dir, 'runs', args.run_log)
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')

    print(f"[rank {rank}] Start: {idname}")
    print(f"[rank {rank}] log_dir: {log_dir}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # # load template mesh
    mesh_dir = os.path.join(asset_dir, args.neutral)
    if not os.path.isdir(mesh_dir):
        print(f"[rank {rank}] Skip {idname}: neutral mesh folder not found: {mesh_dir}")
        return

    # # all meshes
    frame_ids = sorted([x for x in os.listdir(asset_dir) if os.path.isdir(os.path.join(asset_dir, x))])
    if len(frame_ids) == 0:
        print(f"[rank {rank}] Skip {idname}: no frames in {asset_dir}")
        return

    frame_ids_dict = {name: idx for idx, name in enumerate(frame_ids)}

    # # create scene/cameras
    scene = Scene(camera_folder, device=args.device, video_len=len(frame_ids), n_views=args.n_views)

    # # deform model
    DeformModel = Deform_Model(
        args.device, mesh_dir, k=args.k, num_clusters=args.num_clusters,
        normalize_mesh=args.normalize_mesh, normalize_scale=True,
        s=args.s, tx=args.tx, ty=args.ty, tz=args.tz
    ).to(args.device)
    DeformModel.training_setup_rigid()
    DeformModel.example_init()

    # # Gaussians
    gaussians = GaussianModel(lpt.sh_degree, args.device)
    gaussians.create_from_verts(DeformModel.uv_vertices_shape[0], RGB2SH(DeformModel.uv_features_dc.permute(1, 0, 2)))
    gaussians.training_setup(opt)

    # # initialize the mesh renderer with neutral mesh from ./obj
    mesh_path = os.path.join(mesh_dir, 'textured.obj')
    renderer = OBJRenderer(args.device, None, args.image_res)

    first_iter = 0
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    # # cameras
    viewpoint_stack = scene.getCameras()

    # # folder to save the registered mesh
    obj_dir = f'{log_dir}/mesh_captures'
    os.makedirs(obj_dir, exist_ok=True)

    # # render each mesh each view as multi-view videos, including color and normal
    meshes = []
    for f in range(len(frame_ids)):
        meshf0 = load_objs_as_meshes([os.path.join(asset_dir, frame_ids[f], 'textured.obj')], device=args.device)

        if args.normalize_mesh:
            # verts = meshf0.verts_packed()
            # verts = verts - verts.mean(dim=0, keepdim=True)
            # max_dist = torch.cdist(verts, verts).max()
            # verts = verts / max_dist * args.s
            # translation = torch.tensor([args.tx, args.ty, args.tz]).to(args.device)
            # verts = verts + translation
            # meshf0 = meshf0.update_padded(verts.unsqueeze(0).to(args.device))

            verts = DeformModel.normalize_like_trimesh_batched(meshf0.verts_packed()[None, None])
            meshf0 = meshf0.update_padded(verts[0].to(meshf0.device))

        meshes.append(meshf0)

        mesh_nv = Meshes(
            verts=meshf0.verts_packed()[None],
            faces=meshf0.faces_packed()[None],
            textures=TexturesVertex(
                verts_features=(meshf0.verts_normals_packed() / 2 + 0.5)[None]
            )
        )
        for i in range(args.n_views):
            viewpoint_stack[f][i].original_image = renderer.render_mesh(
                meshf0, background, viewpoint_stack[f][i].cam_dist, viewpoint_stack[f][i].elev, viewpoint_stack[f][i].azim
            )
            viewpoint_stack[f][i].normal_image = renderer.render_mesh(
                mesh_nv, background, viewpoint_stack[f][i].cam_dist, viewpoint_stack[f][i].elev, viewpoint_stack[f][i].azim
            )

    rigid_fit_steps = 5000
    # gaussians._scaling_base.requires_grad = False
    # gaussians._rotation_base.requires_grad = False

    gaussians._scaling_base.requires_grad = True
    gaussians._rotation_base.requires_grad = True
    DeformModel.training_setup()

    # # start deformation
    for iteration in tqdm(range(first_iter, 20001), desc=f"[rank {rank}] {idname}"):
        frame = random.randint(0, len(frame_ids) - 1)
        view = random.randint(0, args.n_views - 1)
        # view = random.randint(7, 17)
        # view = 12
        viewpoint_cam = viewpoint_stack[frame][view]
        condition = viewpoint_cam.uid_pe
        if args.view_independent:
            condition[:, 64:] = 0


        verts_final, rot_delta, scale_coef, features_dc_final, verts_deformed, opacity_final = DeformModel.decode(
            condition, args.deform_fc, viewpoint_stack[0][7].visibility_mask, use_LBS=args.noLBS
        )
        gaussians.update_everything_cat(
            verts_final[0], rot_delta[0], scale_coef[0],
            RGB2SH(features_dc_final.permute(1, 0, 2)),
            None, opacity_final[0]
        )

        render_pkg = render(viewpoint_cam, gaussians, ppt, background)

        image = render_pkg["render"]

        # mesh = Meshes(verts=verts_final[:, :-DeformModel.num_samples], faces=DeformModel.faces_idx[None],
        #             textures=TexturesVertex(verts_features=DeformModel.uv_features_dc[:, :-DeformModel.num_samples]))
        # mesh_image = renderer.render_mesh(mesh, background, viewpoint_cam.cam_dist, viewpoint_cam.elev, viewpoint_cam.azim)

        # loss_deform = huber_loss(image, viewpoint_cam.original_image, 0.1) + 0.05 * feature_loss(image, viewpoint_cam.original_image) \
        #             + huber_loss(image, mesh_image, 0.1)
        loss_deform = huber_loss(image, viewpoint_cam.original_image, 0.1) + 0.05 * feature_loss(image, viewpoint_cam.original_image)
        # loss_deform = huber_loss(mesh_image, mesh_image, 0.1)

        # laplace_smooth = pytorch3d.loss.mesh_laplacian_smoothing(mesh)
        # loss_reg = laplace_smooth
        loss_reg = 0

        if args.use_loss_n:
            if args.inverse_n:
                nv = -mesh.verts_normals_packed()
            else:
                nv = mesh.verts_normals_packed()
                
            meshn = Meshes(verts=verts_final, faces=DeformModel.faces_idx[None, ...],
                        textures=TexturesVertex(verts_features=nv[None]/2+0.5))
            meshn_image = renderer.render_mesh(meshn, background, viewpoint_cam.cam_dist, viewpoint_cam.elev, viewpoint_cam.azim)

            gaussians.update_everything_cat(verts_final[0], rot_delta[0], scale_coef[0], RGB2SH((nv[:, None]+1)/2), None, opacity_final[0])

            render_pkg = render(viewpoint_cam, gaussians, ppt, background)

            image_nv = render_pkg["render"]

            loss_n = huber_loss(image_nv, viewpoint_cam.normal_image, 0.1) + 0.005 * feature_loss(image_nv, viewpoint_cam.normal_image) \
                + huber_loss(image_nv, meshn_image, 0.1)
        else:
            loss_n = 0
            image_nv = None

        reg, _ = chamfer_distance(verts_final, meshes[frame].verts_padded())

        loss = loss_deform + loss_reg + loss_n + 0.1*reg
        # loss = loss_deform + loss_reg + loss_n
        loss.backward()

        with torch.no_grad():
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                DeformModel.optimizer.step()
                DeformModel.optimizer.zero_grad(set_to_none=True)

            if iteration % 500 == 0:
                print(f"[rank {rank}] {idname} step: {iteration}, loss: {loss.item():.5f}")

            # if iteration % 500 == 0 and iteration <= rigid_fit_steps:
            if False:
                gt = viewpoint_cam.original_image
                recon = image
                out_final = torch.cat((gt, recon), dim=2)
                torchvision.utils.save_image(out_final, os.path.join(train_dir, f"{iteration}.jpg"))

            # if iteration % 500 == 0 and iteration >= rigid_fit_steps:
            if iteration % 500 == 0:
                gt = viewpoint_cam.original_image
                recon = image
                mesh_vis = Meshes(verts=verts_final, faces=DeformModel.faces_idx[None], textures=TexturesVertex(verts_features=torch.ones_like(verts_final)))
                mesh_img = renderer.render_mesh(mesh_vis, background, viewpoint_cam.cam_dist, viewpoint_cam.elev, viewpoint_cam.azim, light_type='directional')
                
                mesh = Meshes(verts=verts_final, faces=DeformModel.faces_idx[None],
                            textures=TexturesVertex(verts_features=DeformModel.uv_features_dc))
                mesh_image = renderer.render_mesh(mesh, background, viewpoint_cam.cam_dist, viewpoint_cam.elev, viewpoint_cam.azim)
                if image_nv is not None:
                    out_final = torch.cat((gt, recon, mesh_image, mesh_img, image_nv, viewpoint_cam.normal_image, meshn_image), dim=2)
                else:
                    out_final = torch.cat((gt, recon, mesh_image, mesh_img), dim=2)
                torchvision.utils.save_image(out_final, os.path.join(train_dir, f"{iteration}.jpg"))

            if iteration % 1000 == 0 and iteration > first_iter:
                print(f"\n[rank {rank}] [ITER {iteration}] Saving Checkpoint ({idname})")
                ckpt_path = os.path.join(model_dir, f"chkpnt{iteration}.pth")
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), ckpt_path)

                saved_chkpt_list = [30000, 50000]
                prev_ckpt = os.path.join(model_dir, f"chkpnt{iteration-1000}.pth")
                if iteration - 1000 != first_iter and iteration not in saved_chkpt_list and os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)

            # if iteration % 5000 == 0 and iteration > rigid_fit_steps:
            if iteration % 5000 == 0:
                _, faces00, aux00 = load_obj(os.path.join(data_dir, f'obj/{args.neutral}/textured.obj'), load_textures=True)
                texture_image = torchvision.io.read_image(
                    os.path.join(data_dir, f'obj/{args.neutral}/material.png')
                ).float().permute(1, 2, 0) / 255.0

                # avoid shadowing outer "iteration"
                for frame_idx in tqdm(range(len(frame_ids)), desc=f"[rank {rank}] save_mesh {idname}"):
                    verts_final, _, _, _, _, _ = DeformModel.decode(
                        viewpoint_stack[frame_idx][12].uid_pe, args.deform_fc,
                        viewpoint_stack[0][12].visibility_mask, use_LBS=args.noLBS
                    )
                    obj_path = f'{obj_dir}/{frame_ids[frame_idx]}_{iteration}.obj'
                    pytorch3d.io.save_obj(
                        f=obj_path,
                        verts=verts_final[0, :-DeformModel.num_samples],
                        faces=faces00.verts_idx,
                        verts_uvs=aux00.verts_uvs,
                        faces_uvs=faces00.textures_idx,
                        texture_map=texture_image
                    )

    print(f"[rank {rank}] Done: {idname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='', help='Single id name. If empty, process all IDs split by torchrun ranks.')
    # parser.add_argument('--data_root', type=str, default='../trellis2_gpt_textured_test_t2bs')
    parser.add_argument('--data_root', type=str, default='../trellis2_gpt_5kto10k_textured_test_t2bs')
    parser.add_argument('--run_log', type=str, default='0000')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default='ckpt/chkpnt25000.pth')
    parser.add_argument('--n_views', type=int, default=15)
    parser.add_argument('--deform_fc', action='store_true')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=5000)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--eigen_num', type=int, default=10)
    parser.add_argument('--normalize_mesh', action='store_true')
    parser.add_argument('--s', type=float, default=1.5)
    parser.add_argument('--ry', type=float, default=0.0)
    parser.add_argument('--tx', type=float, default=0.0)
    parser.add_argument('--ty', type=float, default=0.0)
    parser.add_argument('--tz', type=float, default=0.0)
    parser.add_argument('--view_independent', action='store_true')
    parser.add_argument('--noLBS', action='store_false')
    parser.add_argument('--neutral', type=str, default='mouth_open_wide')
    parser.add_argument('--inverse_n', action='store_true')
    parser.add_argument('--use_loss_n', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    # torchrun env (or fallback to single process)
    local_rank, rank, world_size = get_torchrun_env()
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"

    print(f"[rank {rank}/{world_size}] local_rank={local_rank}, device={args.device}")

    # one LPIPS module per process / GPU
    percep_module = lpips.LPIPS(net='vgg').to(args.device)

    # single-id mode (your old behavior)
    if args.idname != "":
        train_one_identity(args, lp, op, pp, percep_module, args.idname, rank=rank)
        sys.exit(0)

    # # multi-id mode: split all identities by rank
    all_ids = list_identities(args.data_root)
    my_ids = all_ids[rank::world_size]
    # my_ids = ['Akita_Dog_DreamWorks_Style_3D_long_snout_a_crown_of_leaves____a_wizard_hat_1']

    print(f"[rank {rank}] total_ids={len(all_ids)}, assigned={len(my_ids)}")
    for _id in my_ids:
        try:
            train_one_identity(args, lp, op, pp, percep_module, _id, rank=rank)
            torch.cuda.empty_cache()
        except Exception as e:
            # print(f"[rank {rank}] ERROR on {_id}: {e}", flush=True)
            # continue to next identity instead of killing the whole rank
            raise