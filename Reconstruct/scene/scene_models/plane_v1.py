from argparse import Namespace
import gc
import os
import torch.nn as nn
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import math
import threading
import warnings
import gsplat
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from Reconstruct.utils import inverse_sigmoid, to_numpy
from Reconstruct.utils import save_ply
from plyfile import PlyData, PlyElement

from diff_triangle_rasterization import ConvexRasterizationSettings, ConvexRasterizer
from Reconstruct.tools.mesh_util import get_depth_fusion_mesh
from Reconstruct.tools.pose_metrics import evaluate_poses

import lpips
from fused_ssim import fused_ssim
from Reconstruct.scene.optimizers import SparseGaussianAdam
from Reconstruct.scene.keyframe import Keyframe
from Reconstruct.utils import (
    RGB2SH,
    depth2points,
    focal2fov,
    get_lapla_norm,
    align_poses,
    psnr,
    save_poses_as_pyramid_ply, 
    sample, 
    depth_to_normal,
    getProjectionMatrix2,
    update_gaussians,
    update_triangles,
    save_img_u8,
)
from dataloaders.utils_IO import rotmat2qvec
from pathlib import Path

import open3d as o3d


def sh_to_rgb(sh_coeff):
    C0 = 0.28209479177387814
    # Assuming sh_coeff is a tensor of shape (3,) representing SH coefficients for R, G, B
    # Normalize the SH coefficients to the range [0, 255]
    rgb = (sh_coeff * C0 + 0.5) * 255
    return rgb.astype(np.uint8)

#  Sample points on a disk perpendicular to the given normal vectors
def sample_disk_along_normal(centers, normals, radii, nb_points=3):
    """
    Sample points on a disk perpendicular to the given normal vectors
    
    Args:
        centers: center coordinates, shape [N, 3]
        normals: normal vectors, shape [N, 3] (should be normalized)
        radii: disk radii, shape [N, 1] or [N]
        nb_points: number of points to sample per center, default 3
    
    Returns:
        points_per_convex: [N, nb_points, 3]
    """
    N = centers.shape[0]
    device = centers.device
    
    # Prepare storage
    points_per_convex = torch.zeros((N, nb_points, 3), device=device)
    
    # Normalize normals (in case they aren't already)
    normals = torch.nn.functional.normalize(normals, dim=1)  # [N, 3]
    
    # Construct local coordinate frame perpendicular to normals
    # Find an arbitrary vector not parallel to normal
    up = torch.zeros_like(normals)
    up[:, 2] = 1.0  # Try Z-axis first
    
    # If normal is too close to Z-axis, use X-axis instead
    parallel_mask = (torch.abs(normals[:, 2]) > 0.9)
    up[parallel_mask, 2] = 0.0
    up[parallel_mask, 0] = 1.0
    
    # Compute two orthogonal tangent vectors
    tangent1 = torch.nn.functional.normalize(
        torch.cross(normals, up, dim=1), dim=1
    )  # [N, 3]
    tangent2 = torch.cross(normals, tangent1, dim=1)  # [N, 3], already normalized
 
    
    for i in range(nb_points):
        # Angle for uniform sampling on circle
        theta = 2 * torch.pi * i / nb_points
        cos_theta = torch.cos(torch.tensor(theta, device=device))
        sin_theta = torch.sin(torch.tensor(theta, device=device))
        
        # Point on unit circle in tangent plane
        offset = cos_theta * tangent1 + sin_theta * tangent2  # [N, 3]
        
        # Scale by radius and add to center
        if radii.dim() == 2:
            
            print("radii shape: ", radii.shape)
            if radii.shape[1] > 1:
                radii = radii.mean(dim=1, keepdim=True) # [N, 1]
            
            scaled_offset = radii * offset  # [N, 1] * [N, 3] = [N, 3]
        else:
            scaled_offset = radii.unsqueeze(1) * offset  # [N] -> [N, 1] * [N, 3]
        
        points_per_convex[:, i, :] = centers + scaled_offset
    
    return points_per_convex


class SceneModel:
    """
    Scene Model class that contains the scene's Gaussians, anchors, keyframes, and methods for rendering and optimization.
    """

    def setup_functions(self):
        self.exponential_activation = torch.exp
        self.inverse_exponential_activation = torch.log
        
        # using sigmoid to activate opacity
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

    def __init__(
        self,
        width: int,
        height: int,
        K,
        args: Namespace,
        inference_mode: bool = False,
        device = "cuda:0"
    ):
        """
        Args:
            width: Width of the image.
            height: Height of the image.
            args: Arguments for the scene model. 
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.device = device
        self.width = width
        self.height = height
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device=self.device)
        self.optimization_thread = None
        self.args = args
        self.triangle_depth_ratio = args.triangle_depth_ratio  # better to be 0.0
        self.num_gs_min = args.num_gs_min
        self.num_gs_max = args.num_gs_max

        # loss weights
        self.triangle_depth_loss_weight = args.triangle_depth_loss_weight
        self.triangle_normal_loss_weight = args.triangle_normal_loss_weight
        self.triangle_rgb_loss_weight = args.triangle_rgb_loss_weight
        self.triangle_scaling_loss_weight = args.triangle_scaling_loss_weight
        self.triangle_entropy_loss_weight = args.triangle_entropy_loss_weight
        self.triangle_opacity_thr = args.triangle_opacity_thr
        self.triangle_init_delta = args.triangle_init_delta
        self.triangle_init_sigma = args.triangle_init_sigma
        
        
        self.global_geo_feat_dim = args.global_geo_feat_dim
        self.local_3dgs_feat_dim = args.local_3dgs_feat_dim

        try:
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            self.lpips = lpips.LPIPS(net="vgg").to(self.device)
            sys.stdout = original_stdout
        except:
            self.lpips = None

        if not inference_mode:
            self.num_prev_keyframes_check = args.num_prev_keyframes_check
            self.active_sh_degree = args.sh_degree
            self.max_sh_degree = args.sh_degree
            self.lambda_dssim = args.lambda_dssim
            self.init_proba_scaler = args.init_proba_scaler
            self.max_active_keyframes = args.max_active_keyframes
            self.use_last_frame_proba = args.use_last_frame_proba
            self.num_active_frames_cpu = 0
            self.scaling_reg_factor = args.scaling_reg_factor
            self.rad_decay = args.rad_decay if args.rad_decay != -1 else 1.0
            self.debug_iter = 0

            self.lr_dict = {
                "triangle_points": {
                    "lr_init": args.triangle_points_lr_init,
                    "lr_decay": args.triangle_points_lr_decay,
                },
                "gs_offset": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            # Initialize Gaussian parameters
            self.gaussian_params = {
                "triangle_frame_id": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long)    # Keyframe index associated with each triangle.
                },
                # Persistent triangle identifiers.
                "triangle_uid": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long),
                },
                "triangle_state": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.bool)    
                },
                "triangle_points": {
                    "val": torch.empty(0, 3, 3, device=self.device),       # Attention: n,3,3
                    "lr": args.triangle_points_lr_init,
                },
                "triangle_f_dc": {
                    "val": torch.empty(0, 1, 3, device=self.device),
                    "lr": args.triangle_color_feature_lr,
                },
                "triangle_f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device=self.device,
                    ),
                    "lr": args.triangle_color_feature_lr / 20.0,
                },
                "triangle_opacity": {
                    "val": torch.empty(0, 1, device=self.device),
                    "lr": args.triangle_opacity_lr,
                },
                "triangle_delta": {
                    "val": torch.empty(0, 1, device=self.device),
                    "lr": args.triangle_delta_lr,
                },
                "triangle_sigma": {
                    "val": torch.empty(0, 1, device=self.device),
                    "lr": args.triangle_sigma_lr,
                },
                "triangle_feat": {
                    "val": torch.empty(0, self.args.global_geo_feat_dim, device=self.device),
                    "lr": args.feat_lr,
                },
                # Persistent Gaussian identifiers.
                "gs_uid": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long),
                },
                "gs_triangle_idx": {
                    "val": torch.empty(0, 1, device=self.device, dtype=torch.long),
                },
                "gs_offset":{
                    "val": torch.empty(0, 3, device=self.device),
                    "lr": args.position_lr_init,
                },
                "gs_f_dc":{
                    "val": torch.empty(0, 1, 3, device=self.device),
                    "lr": args.feature_lr,
                },
                "gs_f_rest":{
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device=self.device,
                    ),
                    "lr": args.feature_lr / 20.0,
                },
                "gs_opacity": {
                    "val": torch.empty(0, 1, device=self.device),
                    "lr": args.opacity_lr,
                },
                "gs_scale": {
                    "val": torch.empty(0, 3, device=self.device),
                    "lr": args.scaling_lr,
                },
                "gs_rotation": {
                    "val": torch.empty(0, 4, device=self.device),
                    "lr": args.rotation_lr,
                },
                "gs_feat": {
                    "val": torch.empty(0, self.args.local_3dgs_feat_dim, device=self.device),
                    "lr": args.feat_lr,
                }
            }

            self.setup_functions()

            self.excluded_keys = ["triangle_frame_id", "triangle_uid", "triangle_state","gs_uid", "gs_triangle_idx"]

            self.mlp_cov = nn.Sequential(
                nn.Linear(args.global_geo_feat_dim + args.local_3dgs_feat_dim, args.global_geo_feat_dim + args.local_3dgs_feat_dim),
                nn.ReLU(True),
                nn.Linear(args.global_geo_feat_dim + args.local_3dgs_feat_dim, 7),
            ).cuda()

            self.mlp_params = {}

            for n, p in self.mlp_cov.named_parameters():
                name = "mlp_cov_" + n.replace(".", "_")
                self.mlp_params[name] = {"val": p, "lr": args.mlp_cov_lr_init}
                self.lr_dict[name] = {"lr_init": args.mlp_cov_lr_init, "lr_decay": args.mlp_cov_lr_decay}

            self.next_triangle_uid = 0
            self.next_gs_uid = 0
            # Initialize optimizer
            self.reset_optimizer()

        self.keyframes: list[Keyframe] = []
        self.K = K
        self.f = K[0, 0].item()
        self.init_intrinsics()

        self.approx_cam_centres = None
        self.gt_Rts = torch.empty(0, 4, 4, device=self.device)
        self.gt_Rts_mask = torch.empty(0, device=self.device, dtype=bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device=self.device)
        self.valid_Rt_cache = torch.empty(0, device=self.device, dtype=torch.bool)
        self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        self.lock = threading.Lock()
        self.inference_mode = inference_mode
        self.debug_info = []

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1).to(self.device)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        x = x.to(self.device)
        y = y.to(self.device)
        self.disc_kernel[0, 0, torch.sqrt(x ** 2 + y ** 2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .to(self.device)
        )
        
    
    def get_adaptive_mask(self, xyz, xyz2, camera_center, N=1, p=1.6, min_voxel_size=0.01, max_voxel_size=0.1):
        device = xyz.device
        num_xyz = xyz.shape[0]
        
        
        if xyz2 is None:
            xyz2 = torch.empty((0, 3), device=device, dtype=xyz.dtype)
        num_xyz2 = xyz2.shape[0]
        # -----------------------------
        
        c = torch.as_tensor(camera_center, device=device, dtype=xyz.dtype)
        
        # Compute a distance-dependent voxel size.
        dist_xyz = torch.norm(xyz - c, dim=1)
        dist_xyz2 = torch.norm(xyz2 - c, dim=1)
        all_dists = torch.cat([dist_xyz, dist_xyz2])
        
        
        if all_dists.numel() == 0:
            return torch.zeros(num_xyz, dtype=torch.bool, device=device)

        d_min = torch.quantile(all_dists, 0.1)
        d_max = torch.quantile(all_dists, 0.9)
        s_min, s_max = min_voxel_size, max_voxel_size

        def get_v_indices(pts, dists):
            
            denom = (d_max - d_min) if (d_max > d_min) else 1e-6
            t = torch.clamp((dists - d_min) / denom, 0, 1)
            v_size = s_min + (s_max - s_min) * torch.pow(t, p)
            return torch.floor(pts / v_size.unsqueeze(1)).to(torch.long)

        
        v_idx_xyz = get_v_indices(xyz, dist_xyz)
        v_idx_xyz2 = get_v_indices(xyz2, dist_xyz2)
        
        all_v_idx = torch.cat([v_idx_xyz2, v_idx_xyz], dim=0)
        all_v_idx -= all_v_idx.min(dim=0)[0]
        res = all_v_idx.max(dim=0)[0] + 1
        all_hashes = all_v_idx[:, 0] + all_v_idx[:, 1] * res[0] + all_v_idx[:, 2] * res[0] * res[1]

        # Prioritize existing points in xyz2 over randomly ordered candidates in xyz.
        xyz_rand_perm = torch.randperm(num_xyz, device=device)
        combined_hashes = torch.cat([
            all_hashes[:num_xyz2],
            all_hashes[num_xyz2:][xyz_rand_perm]
        ])

        # Compute the rank of each point within its voxel.
        unique_v_id, v_map = torch.unique(combined_hashes, return_inverse=True)
        num_voxels = unique_v_id.shape[0]

        sort_v, sort_idx = torch.sort(v_map, stable=True)
        
        
        ones_sorted = torch.ones_like(sort_v)
        cumsum_sorted = torch.cumsum(ones_sorted, dim=0) - 1
        
        
        diff = torch.cat([torch.tensor([1], device=device), (sort_v[1:] != sort_v[:-1]).to(torch.long)])
        first_locations = torch.where(diff == 1)[0]
        
        
        local_idx_sorted = cumsum_sorted - first_locations[sort_v]
        
        # Restore input order before selecting the candidate mask.
        local_idx = torch.empty_like(local_idx_sorted)
        local_idx[sort_idx] = local_idx_sorted
        
        
        xyz_mask_shuffled = local_idx[num_xyz2:] < N
        
        # Undo the random permutation of candidate points.
        valid_mask = torch.empty(num_xyz, dtype=torch.bool, device=device)
        valid_mask[xyz_rand_perm] = xyz_mask_shuffled
        
        return valid_mask

    def reset_optimizer(self):
        for key in self.gaussian_params:
            if key in self.excluded_keys:
                continue
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        for key in self.mlp_params:
            if not self.mlp_params[key]["val"].requires_grad:
                self.mlp_params[key]["val"].requires_grad = True
        train_params = {**self.gaussian_params, **self.mlp_params}
        self.optimizer = SparseGaussianAdam(
            train_params, (0.5, 0.99), lr_dict=self.lr_dict, excluded_keys=self.excluded_keys, device=self.device
        )

    @property
    def triangle_frame_id(self):
        return self.gaussian_params["triangle_frame_id"]["val"]

    @property
    def triangle_uid(self):
        return self.gaussian_params["triangle_uid"]["val"]

    @property
    def triangle_state(self):
        return self.gaussian_params["triangle_state"]["val"]

    @property
    def triangle_points(self):
        return self.gaussian_params["triangle_points"]["val"]
    
    @property
    def triangle_barycenters(self):
        return self.triangle_points.mean(dim=1)  
    
    @property
    def triangle_normal(self):
        triangle_vertices = self.triangle_points  
        
        p1p2 = triangle_vertices[:, 1, :] - triangle_vertices[:, 0, :]  
        p1p3 = triangle_vertices[:, 2, :] - triangle_vertices[:, 0, :]  
        
        z_axis = torch.cross(p1p2, p1p3, dim=1)  
        z_axis = torch.nn.functional.normalize(z_axis, dim=1)  
        
        return z_axis  

    @property
    def triangle_feat(self):
        return self.gaussian_params["triangle_feat"]["val"]
    
    @property
    def triangle_f_dc(self):
        return self.gaussian_params["triangle_f_dc"]["val"]

    @property
    def triangle_f_rest(self):
        return self.gaussian_params["triangle_f_rest"]["val"]
    
    @property
    def triangle_color_feature(self):
        return torch.cat([self.triangle_f_dc, self.triangle_f_rest], dim=1)

    @property
    def triangle_opacity(self):
        return self.opacity_activation(self.gaussian_params["triangle_opacity"]["val"])  # opacity_activation is sigmoid

    @property
    def triangle_delta(self):
        return self.exponential_activation(self.gaussian_params["triangle_delta"]["val"]) # delta use exponential activation
    
    @property
    def triangle_sigma(self):
        return self.exponential_activation(self.gaussian_params["triangle_sigma"]["val"]) # sigma use exponential activation
    
    # number of current triangles
    @property
    def number_of_triangles(self):
        return self.triangle_points.shape[0]   

    @property
    def gs_uid(self):
        return self.gaussian_params["gs_uid"]["val"]

    @property
    def gs_triangle_idx(self):
        return self.gaussian_params["gs_triangle_idx"]["val"]

    @property
    def gs_offset(self):
        return self.gaussian_params["gs_offset"]["val"]

    @property
    def gs_f_dc(self):
        return self.gaussian_params["gs_f_dc"]["val"]
    
    @property
    def gs_f_rest(self):
        return self.gaussian_params["gs_f_rest"]["val"]

    @property
    def gs_opacity(self):
        return torch.sigmoid(self.gaussian_params["gs_opacity"]["val"])
    
    @property
    def gs_scale(self):
        return torch.exp(self.gaussian_params["gs_scale"]["val"])

    @property
    def gs_rotation(self):
        return self.gaussian_params["gs_rotation"]["val"]
    
    @property
    def gs_feat(self):
        return self.gaussian_params["gs_feat"]["val"]

 
    
    @property
    def number_of_gaussians(self):
        return self.gs_offset.shape[0]  

    # calculate triangle area
    def get_area(self):        
        a = self.triangle_points[:, 0, :]
        b = self.triangle_points[:, 1, :]
        c = self.triangle_points[:, 2, :]
        return 0.5 * torch.norm(torch.cross(b - a, c - a), dim=1)
    
    # return sequence is p1p2, p2p3, p3p1
    def get_edge_length(self, indices = None):
        a = self.triangle_points[:, 0, :]
        b = self.triangle_points[:, 1, :]
        c = self.triangle_points[:, 2, :]

        edge1 = torch.norm(b - a, dim=1)  # p1p2 
        edge2 = torch.norm(c - b, dim=1)  # p2p3 
        edge3 = torch.norm(a - c, dim=1)  # p3p1 

        edge_lengths = torch.stack([edge1, edge2, edge3], dim=1)

        if indices is not None:
            return edge_lengths.gather(1, indices.unsqueeze(1)).squeeze(1)
        else:
            return edge_lengths    

    # important function for rendering
    @torch.no_grad()
    def pre_split(self, area_threshold=0.0016):
        num_split_per_triangle = torch.zeros((self.number_of_triangles)).int().to(self.device)
        def compute_area(vertices):
            a = vertices[:, 0, :]
            b = vertices[:, 1, :]
            c = vertices[:, 2, :]
            return 0.5 * torch.norm(torch.cross(b - a, c - a), dim=1)

        triangle_points = self.triangle_points.detach()
        
        areas = compute_area(triangle_points)

        large_faces = torch.where(areas > area_threshold)[0] 
        num_split_per_triangle[large_faces] = torch.ceil(torch.log(areas[large_faces] / area_threshold) / torch.log(torch.tensor(4.0))).int()    
        num_subtriangles_per_triangle = 4 ** num_split_per_triangle
        cumsum_of_subtriangles_per_triangle = torch.cumsum(torch.nn.functional.pad(num_subtriangles_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
        
        return num_split_per_triangle, cumsum_of_subtriangles_per_triangle

    @property
    def first_active_frame(self):
        return self.keyframes[0].index

    @property
    def last_active_frame(self):
        return self.keyframes[-1].index

    @property
    def n_active_keyframes(self):
        return self.last_active_frame - self.first_active_frame + 1

    def get_training_id(self):
        while True:
            keyframe_id = np.random.randint(
                self.first_active_frame, self.last_active_frame + 1
            )
            if self.keyframes[keyframe_id].device.type == "cuda":
                return keyframe_id

    def optimization_step(self, is_important=True, finetuning=False):

        if len(self.triangle_points) == 0:
            return None

        if (
            np.random.rand() > self.use_last_frame_proba
            or self.last_trained_id == -1
            or finetuning
        ):
            keyframe_id = self.get_training_id()
        else:
            keyframe_id = -1
        keyframe: Keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl

        # Zero gradients
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        # Render image and depth
        render_pkg_triangle, render_pkg_gs = self.render_from_id(keyframe_id, pyr_lvl=lvl, bg=torch.rand(3, device=self.device))
        
        # extract triangle images
        triangle_rgb = render_pkg_triangle["render"]
        triangle_depth = render_pkg_triangle["surf_depth"]
        triangle_alpha = render_pkg_triangle["rend_alpha"]
        triangle_update_filter = render_pkg_triangle["visibility_filter"]
        triangle_surf_normal = render_pkg_triangle["surf_normal"]
        triangle_render_normal = render_pkg_triangle["rend_normal"]
        
        # extract gs images
        gs_image = render_pkg_gs["render"]
        gs_invdepth = render_pkg_gs["invdepth"]
        gs_scale = render_pkg_gs["scale"]
        
        gt_image = keyframe.image_pyr[lvl]
        mono_depth = keyframe.get_mono_depth(lvl).detach()
        mono_idepth = keyframe.get_mono_idepth(lvl).detach()

        mono_normal_local = keyframe.get_mono_normal_local(lvl)    # [3, H, W]
        # transfer to world frame
        mono_normal_world = (keyframe.get_R().detach().T @ mono_normal_local.reshape(3, -1)).reshape(3, *mono_normal_local.shape[1:]).detach()
    
        # Triangle Loss
        c, h, w = triangle_rgb.shape

        triangle_rgb_l1_loss = ((triangle_rgb - gt_image).abs()).mean()
        triangle_ssim_loss = 1 - fused_ssim(triangle_rgb[None], gt_image[None])
        triangle_depth_loss = ((triangle_depth - mono_depth).abs()).mean()
        
        triangle_surf_normal_loss_l1 = ((triangle_surf_normal - mono_normal_world).abs()).mean()
        triangle_render_normal_loss_l1 = ((triangle_render_normal - mono_normal_world).abs()).mean()
        triangle_normal_loss_l1 = 0.5 * triangle_surf_normal_loss_l1 + 0.5 * triangle_render_normal_loss_l1 
        
        triangle_surf_normal_loss_cos =  (1 - (triangle_surf_normal * mono_normal_world).sum(dim=0))[None].mean()  # cos loss
        triangle_render_normal_loss_cos =  (1 - (triangle_render_normal * mono_normal_world).sum(dim=0))[None].mean()  # cos loss
        triangle_normal_loss_cos = 0.5 * triangle_surf_normal_loss_cos + 0.5 * triangle_render_normal_loss_cos 
        
        triangle_normal_loss = triangle_normal_loss_l1 + triangle_normal_loss_cos
        
        triangle_scaling_reg = self.get_edge_length()[triangle_update_filter].max(dim=1)[0].mean()   # reg the longest edge
        
        triangle_used_opacity = torch.clamp(self.triangle_opacity[triangle_update_filter], min=1e-3, max=1-1e-3)
        triangle_binary_opacity_entropy_loss = (-triangle_used_opacity * torch.log(triangle_used_opacity) - (1 - triangle_used_opacity) * torch.log(1 - triangle_used_opacity)).mean() 
        
        # sum of losses
        triangle_loss = (
            self.triangle_rgb_loss_weight * 
            (self.lambda_dssim * triangle_ssim_loss
            + (1 - self.lambda_dssim) * triangle_rgb_l1_loss)
            + self.triangle_depth_loss_weight * triangle_depth_loss
            + self.triangle_normal_loss_weight * triangle_normal_loss
            + self.triangle_scaling_loss_weight * triangle_scaling_reg  
            + self.triangle_entropy_loss_weight * triangle_binary_opacity_entropy_loss
        )
        
        gs_l1_loss = ((gs_image - gt_image).abs()).mean()
        gs_ssim_loss = 1 - fused_ssim(gs_image[None], gt_image[None])
        gs_depth_loss = ((gs_invdepth - mono_idepth).abs()).mean()
        gs_scaling_reg = gs_scale.prod(dim=1).mean()
        gs_loss = (
            self.lambda_dssim * gs_ssim_loss
            + (1 - self.lambda_dssim) * gs_l1_loss
            + keyframe.depth_loss_weight * gs_depth_loss
            + self.scaling_reg_factor * gs_scaling_reg     # scaling_reg_factor is 0 default
        )      
        
        loss = triangle_loss + gs_loss  

        loss.backward()

        # Optimizers
        with torch.no_grad():
            # Pose optimization
            keyframe.step()

            # Skip the scene optimization if the current keyframe is a test keyframe
            if not keyframe.is_test:
                # Scene Gaussian optimization
                self.optimizer.step(
                    render_pkg_triangle["visibility_filter"], render_pkg_triangle["radii"].shape[0],
                    render_pkg_gs["visibility_filter"], render_pkg_gs["radii"].shape[0]
                )


            keyframe.latest_depth = render_pkg_triangle["surf_depth"].detach()
            keyframe.latest_invdepth = torch.where(keyframe.latest_depth > 0.00001, 1.0 / keyframe.latest_depth, 0)
            

        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id
        
        return render_pkg_triangle["visibility_filter"]


    def optimization_loop(self, n_iters: int, is_important: bool, run_until_interupt: bool = False, debug_save_path: str = None):
        """
        Runs at least n_iters optimization steps.
        If run_until_interupt, also runs until join_optimization_thread is called (Useful to run the optimization until the next keyframe is added in streaming mode).
        """
        self.interupt_optimization = False
        i = 0
        while i < n_iters or (run_until_interupt and not self.interupt_optimization):
            self.optimization_step(is_important=is_important)
            i += 1

    def join_optimization_thread(self):
        """
        Interupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None

    def optimize_async(self, n_iters: int, is_important: bool):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(target=self.optimization_loop, args=(n_iters, is_important, True))
        self.optimization_thread.start()

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Harmonizes the exposure matrices of test keyframes by averaging the exposure of the previous and next keyframes."""
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.is_test:
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                keyframe.exposure = (
                                        self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                                    ) / 2

    @torch.no_grad()
    def evaluate(self, with_LPIPS=False, all=False):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()
        torch.cuda.empty_cache()
        # Compute image quality metrics
        metrics = {"GS_PSNR": 0.0,"GS_SSIM": 0.0,"TRI_PSNR": 0.0, "TRI_SSIM": 0.0, "Depth L1 Loss": 0.0, "Alpha Mean": 0.0, "Mean Opacity": 0.0, "Mean Delta": 0.0, "Mean Sigma": 0.0}  # use depth l1 loss for debug
        if with_LPIPS:
            metrics["GS_LPIPS"] = 0.0
            metrics["TRI_LPIPS"] = 0.0
        n_test_frames = 0
        start_index = 0 if all else self.keyframes[0].index
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.is_test:
                gt_image = keyframe.image_pyr[0].to(self.device)
                mono_depth = keyframe.get_mono_depth(0)
                render_pkg_triangle, render_pkg_gs = self.render_from_id(keyframe.index, pyr_lvl=0)
                triangle_rgb = render_pkg_triangle["render"]
                triangle_depth = render_pkg_triangle["surf_depth"]  # use depth l1 loss for debug
                triangle_alpha = render_pkg_triangle["rend_alpha"]
                gs_rgb = render_pkg_gs["render"]
                mask = torch.ones_like(triangle_rgb[:1] > 0)
                mask = mask.expand_as(triangle_rgb)
                triangle_rgb = triangle_rgb * mask
                gt_image = gt_image * mask
                triangle_depth = triangle_depth * mask
                mono_depth = mono_depth * mask
                depth_l1_loss = ((triangle_depth - mono_depth).abs()).mean()
                gs_mask = torch.ones_like(gs_rgb > 0)
                metrics["GS_PSNR"] += psnr(gs_rgb[gs_mask], gt_image[gs_mask])
                metrics["GS_SSIM"] += fused_ssim(
                    gs_rgb[None], gt_image[None], train=False
                ).item()
                metrics["TRI_PSNR"] += psnr(triangle_rgb[mask], gt_image[mask])
                metrics["TRI_SSIM"] += fused_ssim(
                    triangle_rgb[None], gt_image[None], train=False
                ).item()
                metrics["Depth L1 Loss"] += depth_l1_loss.item() # use depth l1 loss for debug
                metrics["Alpha Mean"] += triangle_alpha.mean().item()  # use alpha mean for debug, Alpha Mean --> 1 means all area is covered
                metrics["Mean Opacity"] += self.triangle_opacity.mean().item()  # use alpha mean for debug 
                metrics["Mean Delta"] += self.triangle_delta.mean().item()  # use delta mean for debug 
                metrics["Mean Sigma"] += self.triangle_sigma.mean().item()  # use sigma mean for debug 
                if with_LPIPS and self.lpips is not None:
                    metrics["GS_LPIPS"] += self.lpips(gs_rgb[None], gt_image[None]).item()
                    metrics["TRI_LPIPS"] += self.lpips(triangle_rgb[None], gt_image[None]).item()
                n_test_frames += 1

        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}
        metrics["n_test_frames"] = n_test_frames
        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        """Export held-out GS RGB, GS depth visualization, and triangle rendered normals."""
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        # Optional video: uncomment all lines marked VIDEO to enable this export.
        # writer = None
        for keyframe in self.keyframes:
            if not keyframe.is_test:
                continue
            tri, gs = self.render_from_id(keyframe.index, pyr_lvl=0, if_render_depth=True)
            stem = Path(keyframe.image_name).stem
            rgb = (gs["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = gs["rendered_depth"].squeeze(0).cpu().numpy()
            valid = np.isfinite(depth) & (depth > 0)
            depth_vis = np.zeros(depth.shape, dtype=np.uint8)
            if valid.any():
                lo, hi = depth[valid].min(), depth[valid].max()
                if hi > lo:
                    depth_vis[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            normal = tri["rend_normal"].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            if not cv2.imwrite(os.path.join(out_dir, f"{stem}_gs_rgb.png"), rgb_bgr):
                raise OSError(f"Cannot write GS RGB: {stem}")
            if not cv2.imwrite(os.path.join(out_dir, f"{stem}_gs_depth.png"), depth_vis):
                raise OSError(f"Cannot write GS depth: {stem}")
            save_img_u8(normal, os.path.join(out_dir, f"{stem}_tri_rendered_normal.png"))
            count += 1
            # normal_bgr = cv2.cvtColor((np.clip(normal, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            # panel = cv2.hconcat([rgb_bgr, cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR), normal_bgr])
            # if writer is None:
            #     writer = cv2.VideoWriter(os.path.join(out_dir, "test_views.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (panel.shape[1], panel.shape[0]))
            #     if not writer.isOpened():
            #         raise RuntimeError("Cannot open test video writer")
            # writer.write(panel)
        # if writer is not None:
        #     writer.release()
        print(f"Saved three images for each of {count} test frames to {out_dir}")



    @torch.no_grad()
    def save_all_frames(self, out_dir, gs_video_path=None, fps=24, colorize_depth=False, triangle_video_path=None, include_rend_normal=False, per_rot_deg=None, per_trans_amp=None):
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        diagnostics = []

        if gs_video_path is None:
            gs_video_path = os.path.join(out_dir, "combined_gs.mp4")
        if triangle_video_path is None:
            triangle_video_path = os.path.join(out_dir, "combined_triangle.mp4")

        writer_gs = None
        writer_tri = None

        for i, keyframe in enumerate(self.keyframes):
            # Render both triangle and Gaussian representations.
            
            render_pkg_triangle, render_pkg_gs = self.render_from_id(keyframe.index, pyr_lvl=0)

            # ================== GS ==================
            # RGB
            gs_rgb = torch.clamp(render_pkg_gs["render"], 0, 1) * 255
            gs_rgb = gs_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            gs_rgb = cv2.cvtColor(gs_rgb, cv2.COLOR_RGB2BGR)

            
            invdepth = render_pkg_gs["invdepth"].squeeze(0).detach().cpu().numpy()
            vmin = np.nanmin(invdepth)
            vmax = np.nanmax(invdepth)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                invdepth_vis = np.zeros_like(invdepth, dtype=np.uint8)
            else:
                invdepth_norm = (invdepth - vmin) / (vmax - vmin)
                invdepth_vis = np.clip(invdepth_norm * 255.0, 0, 255).astype(np.uint8)

            
            H_gs, W_gs = gs_rgb.shape[:2]
            if invdepth_vis.shape[:2] != (H_gs, W_gs):
                invdepth_vis = cv2.resize(invdepth_vis, (W_gs, H_gs), interpolation=cv2.INTER_NEAREST)
            if colorize_depth:
                gs_depth_3c = cv2.applyColorMap(invdepth_vis, cv2.COLORMAP_TURBO)
            else:
                gs_depth_3c = cv2.cvtColor(invdepth_vis, cv2.COLOR_GRAY2BGR)

            
            stem = os.path.splitext(os.path.basename(keyframe.image_name))[0]
            depth_gray_path = os.path.join(out_dir, f"{stem}_gs_inv_depth.png")
            cv2.imwrite(depth_gray_path, invdepth_vis)

            is_jpeg = os.path.splitext(keyframe.image_name)[-1].lower() in [".jpg", ".jpeg"]
            write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
            cv2.imwrite(os.path.join(out_dir, keyframe.image_name), gs_rgb, write_flag)
            diagnostics.append((i, out_dir, keyframe.image_name))

            # Compose Gaussian video frames as RGB and depth panels.
            panel_gs = cv2.hconcat([gs_rgb, gs_depth_3c])
            if writer_gs is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_gs = cv2.VideoWriter(gs_video_path, fourcc, float(fps), (panel_gs.shape[1], panel_gs.shape[0]), True)
                if not writer_gs.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter at {gs_video_path}")
            writer_gs.write(panel_gs)

            # ================== TRIANGLE ==================
            
            surf_depth = render_pkg_triangle["surf_depth"]
            triangle_depth = surf_depth.squeeze(0).detach().cpu().numpy()
            vmin = np.nanmin(triangle_depth)
            vmax = np.nanmax(triangle_depth)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                tri_depth_vis = np.zeros_like(triangle_depth, dtype=np.uint8)
            else:
                tri_depth_norm = (triangle_depth - vmin) / (vmax - vmin)
                tri_depth_vis = np.clip(tri_depth_norm * 255.0, 0, 255).astype(np.uint8)
                
            depth_gray_path = os.path.join(out_dir, f"{stem}_tri_depth.png")
            cv2.imwrite(depth_gray_path, tri_depth_vis)

            
            if colorize_depth:
                tri_depth_3c = cv2.applyColorMap(tri_depth_vis, cv2.COLORMAP_TURBO)
            else:
                tri_depth_3c = cv2.cvtColor(tri_depth_vis, cv2.COLOR_GRAY2BGR)

            # surf_normal / rend_normal -> [3,H,W] -> BGR uint8
            tri_surf_normal = render_pkg_triangle["surf_normal"]
            tri_rendered_normal = render_pkg_triangle["rend_normal"]
            surf_norm_img = (tri_surf_normal.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            surf_norm_img = np.clip(surf_norm_img * 255.0, 0, 255).astype(np.uint8)
            surf_norm_img = cv2.cvtColor(surf_norm_img, cv2.COLOR_RGB2BGR)
            
            rend_norm_img = (tri_rendered_normal.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            rend_norm_img = np.clip(rend_norm_img * 255.0, 0, 255).astype(np.uint8)
            rend_norm_img = cv2.cvtColor(rend_norm_img, cv2.COLOR_RGB2BGR)
            
            surf_normal_path = os.path.join(out_dir, f"{stem}_tri_surf_normal.png")
            cv2.imwrite(surf_normal_path, surf_norm_img)
            rend_normal_path = os.path.join(out_dir, f"{stem}_tri_rend_normal.png")
            cv2.imwrite(rend_normal_path, rend_norm_img)

            panels_tri = []

            
            H_tri, W_tri = tri_depth_3c.shape[:2]
            def ensure_size(img, size_hw):
                if img.shape[:2] != size_hw:
                    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
                return img

            tri_depth_3c = ensure_size(tri_depth_3c, (H_tri, W_tri))
            surf_norm_img = ensure_size(surf_norm_img, (H_tri, W_tri))
            rend_norm_img = ensure_size(rend_norm_img, (H_tri, W_tri))
            panels_tri.extend([tri_depth_3c, surf_norm_img, rend_norm_img])

            if include_rend_normal and "rend_normal" in render_pkg_triangle:
                tri_rend_normal = render_pkg_triangle["rend_normal"]
                rend_norm_img = (tri_rend_normal.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
                rend_norm_img = np.clip(rend_norm_img * 255.0, 0, 255).astype(np.uint8)
                rend_norm_img = cv2.cvtColor(rend_norm_img, cv2.COLOR_RGB2BGR)
                rend_norm_img = ensure_size(rend_norm_img, (H_tri, W_tri))
                panels_tri.append(rend_norm_img)

            panel_tri = cv2.hconcat(panels_tri)

            if writer_tri is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_tri = cv2.VideoWriter(triangle_video_path, fourcc, float(fps), (panel_tri.shape[1], panel_tri.shape[0]), True)
                if not writer_tri.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter at {triangle_video_path}")
            writer_tri.write(panel_tri)

        if writer_gs is not None:
            writer_gs.release()
        if writer_tri is not None:
            writer_tri.release()

        print(f"Saved {len(diagnostics)} frames to {out_dir}")
        print(f"GS video saved to: {gs_video_path}")
        print(f"Triangle video saved to: {triangle_video_path}")


    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        bg=torch.zeros(3),
        scaling_modifier=1.0,
        if_render_depth: bool = False,
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        bg = bg.to(self.device)
        keyframe = self.keyframes[keyframe_id]
        view_matrix_triangle = keyframe.get_Rt().transpose(0, 1).to(self.device)
        view_matrix_gs = keyframe.get_Rt().to(self.device)
        scale = 2 ** pyr_lvl
        width, height = self.width // scale, self.height // scale
        render_pkg_triangle = self.render_triangle(width, height, view_matrix_triangle, scaling_modifier, bg)
        render_pkg_triangle["render"] = (keyframe.exposure[:3, :3] @ render_pkg_triangle["render"].view(3, -1)) + keyframe.exposure[:3, 3, None]
        render_pkg_triangle["render"] = render_pkg_triangle["render"].clamp(0, 1).view(3, height, width)
        render_pkg_gs = self.render_gs(width, height, view_matrix_gs, bg, if_render_depth)
        render_pkg_gs["render"] = (keyframe.exposure[:3, :3] @ render_pkg_gs["render"].view(3, -1)) + keyframe.exposure[:3, 3, None]
        render_pkg_gs["render"] = render_pkg_gs["render"].clamp(0, 1).view(3, height, width)
        return render_pkg_triangle, render_pkg_gs

    def render_gs(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        bg: torch.Tensor = torch.zeros(3),
        if_render_depth: bool = False,
    ):
        with self.lock:
            # Load and blend anchors if in inference mode
            xyz = self.triangle_barycenters[self.gs_triangle_idx.squeeze(-1)] + self.gs_offset
            scaling = self.gs_scale
            rotation = self.gs_rotation
            opacity = self.gs_opacity
            feat = torch.concat([self.triangle_feat[self.gs_triangle_idx.squeeze(-1)], self.gs_feat], dim=-1)
            scale_rot = self.mlp_cov(feat)
            scaling = torch.sigmoid(scale_rot[:, :3]) * scaling
            rotation = F.normalize(rotation * scale_rot[:, 3:])   
            shs = torch.concat([self.gs_f_dc, self.gs_f_rest], dim=1)

            fl_x = width / (2 * self.tanfovx)
            fl_y = height / (2 * self.tanfovy)
            Ks = torch.tensor([
                [fl_x, 0, width / 2.0],
                [0, fl_y, height / 2.0],
                [0, 0, 1],
            ],device="cuda",)[None]

            colors, alphas, meta = gsplat.rendering.rasterization(
                means=xyz,
                quats=rotation,
                scales=scaling,
                opacities=opacity.squeeze(-1),
                colors=shs,
                viewmats=view_matrix.unsqueeze(0),
                Ks=Ks,
                width=width,
                height=height,
                render_mode="RGB+D",
                rasterize_mode="classic",
                absgrad=False,
                packed=False,
                sh_degree=self.active_sh_degree,
                eps2d=self.args.low_pass_filter_eps,
            )  

            rendered_color = colors[..., 0:3].permute([0, 3, 1, 2])
            rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
            rendered_alpha = alphas.permute([0, 3, 1, 2])
            rendered_color = rendered_color + (1.0 - rendered_alpha) * bg[None, :, None, None]
            invdepth = 1.0 / rendered_depth_unnormalized

        radii = meta['radii'][0].max(dim=1).values
        return {
            "render": rendered_color[0],
            "invdepth": invdepth[0],
            "radii": radii, 
            "visibility_filter": radii > 0,
            "scale": scaling,
            "rendered_depth": rendered_depth_unnormalized[0] if if_render_depth else None,
        }   

    def render_triangle(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3),
        top_view: bool = False,
        fov_x: Optional[float] = None,
        fov_y: Optional[float] = None,
    ):
        triangle_points = self.triangle_points
        screenspace_points = torch.zeros_like(triangle_points[:,0,:].squeeze(), dtype=self.triangle_points.dtype, requires_grad=True, device=self.device) + 0
        
        # Maximum distance from the triangle centroid to its vertices, computed in CUDA.
        triangle_scaling = torch.zeros_like(triangle_points[:,0,0].squeeze(), dtype=self.triangle_points.dtype, requires_grad=True, device=self.device).detach()
        density_factor = torch.zeros_like(triangle_points[:,0,0].squeeze(), dtype=self.triangle_points.dtype, requires_grad=True, device=self.device).detach()        
            
        try:
            screenspace_points.retain_grad()
        except:
            pass
            
        bg = bg.to(self.device)
        cam_centre = view_matrix.detach().inverse()[3, :3].to(self.device)

        # Use the scene's intrinsic parameters if not provided
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # Use the provided FOV values
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix2(znear=0.01, zfar=100.0, cx=self.K[0, 2],
                                 cy=self.K[1, 2], fx=self.K[0, 0], fy=self.K[1, 1],
                                 W=self.width, H=self.height)
                .transpose(0, 1)
                .to(self.device)
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")
        
        # Compose the detached view matrix with the projection matrix for the rasterizer.
        full_proj_transform = (view_matrix.detach().unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0) 
        
        raster_settings = ConvexRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier= 1 if top_view else scaling_modifier,
            viewmatrix=view_matrix.detach(),   # Block pose gradients through the triangle rasterizer.
            projmatrix=full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=cam_centre,
            prefiltered=False,
            debug=False
        )

        rasterizer = ConvexRasterizer(raster_settings=raster_settings)
                
        # get triangle attributes for rendering
        with self.lock:
            assert triangle_points.shape[0] > 0, "No triangles to render"
            # get triangle_opacity for rendering
            if top_view:
                triangle_opacity = torch.ones_like(self.triangle_opacity)  # Set constant opacity to visualize the triangles in the top view
            else:
                triangle_opacity = self.triangle_opacity
            
            triangle_points = triangle_points.flatten(0)  # n,3,3 --> 3n,3   # get triangle points for rendering
            
            num_split_per_triangle, cumsum_of_subtriangles_per_triangle = self.pre_split(area_threshold=0.01)  
            # The subdivision area threshold depends on the scene scale.
            
            shs = self.triangle_color_feature # get shs for rendering
            delta = self.triangle_delta # get delta for rendering
            sigma = self.triangle_sigma # get sigma for rendering

            number_of_triangles = triangle_opacity.shape[0]
            num_points_per_triangle = torch.full((number_of_triangles,), 3, dtype=torch.int, device=self.device)
            cumsum_of_points_per_triangle = torch.cumsum(num_points_per_triangle, dim=0, dtype=torch.int) - 3     # 0, 3, 6,  ..., 52650, 52653, 52656   torch.Size([17553])
            means2D = screenspace_points           # just for record some grad to perform adaptive density control    
            
            rendered_image, radii, triangle_scaling, density_factor, impact, allmap = rasterizer(
                convex_points=triangle_points,
                delta=delta,
                sigma=sigma,
                num_split_per_convex=num_split_per_triangle,    # Subdivision count per triangle.
                cumsum_of_subtriangles_per_convex=cumsum_of_subtriangles_per_triangle,   # Starting offset of each triangle in the subtriangle array.
                num_points_per_convex = num_points_per_triangle,  
                cumsum_of_points_per_convex = cumsum_of_points_per_triangle,
                number_of_points = number_of_triangles,
                shs = shs,
                colors_precomp = None,
                opacities = triangle_opacity,
                means2D = means2D,
                scaling = triangle_scaling,
                density_factor = density_factor
            )    

            # ========= Process allmap for geometry =========
            # additional regularizations
            render_alpha = allmap[1:2]

            # get normal map
            # transform normal from view space to world space
            render_normal = allmap[2:5]
            render_normal = (render_normal.permute(1,2,0) @ (view_matrix[:3,:3].detach().T)).permute(2,0,1)      

            # get median depth map
            render_depth_median = allmap[5:6]
            render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

            # get expected depth map
            render_depth_expected = allmap[0:1]
            render_depth_expected = (render_depth_expected / render_alpha)
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
            
            # get depth distortion map
            render_dist = allmap[6:7]
            surf_depth = render_depth_expected * (1-self.triangle_depth_ratio) + (self.triangle_depth_ratio) * render_depth_median
            
            # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
            surf_normal = depth_to_normal(view_matrix.detach(), width, height, full_proj_transform, surf_depth)     # depth_to_normal returns normals in world coordinates.
            surf_normal = surf_normal.permute(2,0,1)
            # remember to multiply with accum_alpha since render_normal is unnormalized.
            surf_normal = surf_normal * (render_alpha).detach()

            rets =  {"render": rendered_image,
                "viewspace_points": screenspace_points,  
                "visibility_filter" : impact > 0,        
                "radii": radii, 
                "impact": impact,             
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
                }                

        return rets
        
    def get_Rts(self):
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        return {
            "triangle_frame_id": self.triangle_frame_id[:0].detach(),
            "triangle_uid": self.triangle_uid[:0].detach(),
            "triangle_state": self.triangle_state[:0].detach(),
            "triangle_points": self.triangle_points[:0].detach(),
            "triangle_f_dc": self.triangle_f_dc[:0].detach(),
            "triangle_f_rest": self.triangle_f_rest[:0].detach(),
            "triangle_opacity": self.triangle_opacity[:0].detach(),
            "triangle_delta": self.triangle_delta[:0].detach(),
            "triangle_sigma": self.triangle_sigma[:0].detach(),
            "triangle_feat": self.triangle_feat[:0].detach(),
            "gs_uid": self.gs_uid[:0].detach(),
            "gs_triangle_idx": self.gs_triangle_idx[:0].detach(),
            "gs_offset": self.gs_offset[:0].detach(),
            "gs_f_dc": self.gs_f_dc[:0].detach(),
            "gs_f_rest": self.gs_f_rest[:0].detach(),
            "gs_opacity": self.gs_opacity[:0].detach(),
            "gs_scale": self.gs_scale[:0].detach(),
            "gs_rotation": self.gs_rotation[:0].detach(),
            "gs_feat": self.gs_feat[:0].detach(),
        }
 

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        keyframe = self.keyframes[keyframe_id]

        # Skip if the keyframe is a test keyframe
        if keyframe.is_test:
            return
        
        view_matrix = keyframe.get_Rt().transpose(0, 1).to(self.device)
        cam_centre = view_matrix.detach().inverse()[3, :3].to(self.device)
        org_img = keyframe.image_pyr[0]     
        init_proba = get_lapla_norm(org_img, self.disc_kernel, device=self.device)
        # Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        if self.triangle_points.shape[0] > 0:
            render_pkg_triangle, render_pkg_gs = self.render_from_id(keyframe_id)
            render = F.interpolate(render_pkg_triangle["render"][None], (self.height, self.width), mode="bilinear", align_corners=True)[0]
            rendered_depth = F.interpolate(render_pkg_triangle["surf_depth"][None], (self.height, self.width), mode="bilinear", align_corners=True)[0][0].clamp_min(1e-8)
            penalty = get_lapla_norm(render, self.disc_kernel, device=self.device)

        ## Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler   # default: 2
        penalty *= self.init_proba_scaler
        # sample points
        sample_mask = torch.rand_like(init_proba) < (init_proba - penalty) * self.args.gs_add_ratio  
        sample_mask = sample_mask.to(self.device)
        sampled_uv = self.uv[sample_mask]  # L
        sampled_depths = sample(
            keyframe.point_map[:, 2:],  # 1 1 H W
            sampled_uv[None, None, ...],  # 1 1 L 2
            keyframe.width,
            keyframe.height
        )[0, 0, 0]  # L

        sampled_conf = sample(
            keyframe.mono_depth_conf,  # 1 1 H W
            sampled_uv[None, None, ...],  # 1 1 L 2
            keyframe.width,
            keyframe.height
        )[0, 0, 0]  # L
        
        mono_normal_local = keyframe.normal_map_local[0] # 3 H_slam W_slam
        mono_normal_world = (keyframe.get_R().detach().T @ mono_normal_local.reshape(3, -1)).reshape(3, *mono_normal_local.shape[1:]).detach()

        sampled_normal_world = sample(
            mono_normal_world[None],  # 1 3 H W
            sampled_uv[None, None, ...],  # 1 1 L 2
            keyframe.width,
            keyframe.height
        )[0, :, 0, :].transpose(0, 1)
        
        # init_proba
        sampled_init_proba = init_proba[sample_mask] 
        
        # init f_dc
        sampled_rgb = org_img[:, sample_mask]
        quantile_depth_min = min(1e-2, torch.quantile(keyframe.point_map[:, 2], 0.02).item())
        valid_mask = ((sampled_conf >= keyframe.mono_depth_conf.mean() * 1.0/4.0)    
                    * (sampled_depths > quantile_depth_min))  # L
        sample_mask[sample_mask.clone()] = valid_mask
        sampled_uv = sampled_uv[valid_mask]  # L1 2
        sampled_depths = sampled_depths[valid_mask]  # L1 3
        sampled_conf = sampled_conf[valid_mask]
        sampled_normal_world = sampled_normal_world[valid_mask]
        sampled_init_proba = sampled_init_proba[valid_mask]
        sampled_rgb = sampled_rgb[:, valid_mask]

        # The initialization assumes equal focal lengths and a centered principal point.
        f = self.f
        centre = self.centre
        sampled_points = depth2points(sampled_uv, sampled_depths.unsqueeze(-1), f, centre)
        sampled_points = (sampled_points - keyframe.get_t()) @ keyframe.get_R() # transform point from C frame to W frame

        deduplicate_mask = self.get_adaptive_mask(sampled_points, self.triangle_barycenters, cam_centre)
        
        # Prune before inserting new primitives to update occupancy.
        ## Get which Gaussians should be pruned
        if self.triangle_barycenters.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_triangle_mask = self.triangle_opacity[:, 0] > self.triangle_opacity_thr  
            
            N_triangles = self.triangle_barycenters.shape[0]

            xyz = self.triangle_barycenters[self.gs_triangle_idx.squeeze(-1)] + self.gs_offset
            dist_flat = torch.linalg.vector_norm(
                xyz - keyframe.approx_centre[None], dim=-1
            ).clamp_min(1e-6)  # [N_gs]

            screen_size = self.f * self.gs_scale.max(dim=-1)[0] / dist_flat  # [N_gs]
            valid_gs_mask = torch.logical_and(self.gs_opacity.squeeze(-1) > 0.05, screen_size < 0.5 * self.width)
        else:
            valid_triangle_mask = torch.ones(0, device=self.device, dtype=torch.bool)
            valid_gs_mask = torch.ones(0, device=self.device, dtype=torch.bool)
            
        # Remove bindings associated with pruned triangles.
        self.triangle_state[~valid_triangle_mask] = False
        valid_gs_mask = torch.logical_and(valid_gs_mask, self.triangle_state[self.gs_triangle_idx.squeeze(-1)].squeeze(-1))
        idx_converter = torch.cumsum(self.triangle_state, dim=0).to(self.device) - 1
        self.gaussian_params["gs_triangle_idx"]["val"] = idx_converter[self.gs_triangle_idx.squeeze(-1)]
        left_triangle_num = valid_triangle_mask.sum()

        num_new = sampled_points[deduplicate_mask].shape[0]
        uids = torch.arange(
            self.next_triangle_uid, 
            self.next_triangle_uid + num_new, 
            device=self.device, 
            dtype=torch.long
        ).unsqueeze(1)
        self.next_triangle_uid += num_new  

        states = torch.ones(num_new, 1, device=self.device, dtype=torch.bool)

        # Initialize Scales 
        # Expected distance to the nearest neighbour
        triangle_scales = 1 / (torch.sqrt(sampled_init_proba))    
        triangle_scales.clamp_(1, self.width / 10) 
        # Scale by the distance to the camera centre
        triangle_scales.mul_(1 / self.f) * 3.0   
        
        triangle_scales *= torch.linalg.vector_norm(
            sampled_points - keyframe.approx_centre[None], dim=-1
        )
        # Initialize triangle_points
        # using sampled_points as means, using sampled_normal_world as normals, using triangle_scales as radii to initialize triangle_points
        triangle_points = sample_disk_along_normal(
            sampled_points,
            sampled_normal_world,
            triangle_scales,
            nb_points=3
        ).to(self.device)  # [N, 3, 3]  
        
        ## Initialize Colour
        # explicitly initialize the color of the point
        f_dc = sampled_rgb
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1)).to(self.device)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device=self.device)
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= 0.2 * sampled_conf[..., None]
        opacities = inverse_sigmoid(opacities)
        
        # Initialize deltas and sigmas
        deltas = self.inverse_exponential_activation(torch.ones((opacities.shape[0], 1), dtype=torch.float, device=self.device) * self.triangle_init_delta)  
        sigmas = self.inverse_exponential_activation(torch.ones((opacities.shape[0], 1), dtype=torch.float, device=self.device) * self.triangle_init_sigma)
        feat = torch.zeros(f_dc.shape[0], self.args.global_geo_feat_dim, device=self.device)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device=self.device,
        )
        
        init_scales = 1 / (torch.sqrt(sampled_init_proba))
        init_scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        init_scales.mul_(1 / self.f)
        init_scales *= torch.linalg.vector_norm(
            sampled_points - keyframe.approx_centre[None], dim=-1
        )
        init_scales = torch.log(init_scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        triangle_scales = triangle_scales[deduplicate_mask]
        triangle_points = triangle_points[deduplicate_mask]     
        opacities = opacities[deduplicate_mask]     
        deltas = deltas[deduplicate_mask]
        sigmas = sigmas[deduplicate_mask]
        f_dc = f_dc[deduplicate_mask]
        f_rest = f_rest[deduplicate_mask]
        feat = feat[deduplicate_mask]
        init_scales = init_scales[deduplicate_mask]
        sampled_init_proba = sampled_init_proba[deduplicate_mask]

        proba_mask = sampled_init_proba > self.args.proba_threshold
        num_gss = torch.ones(num_new, device=self.device, dtype=torch.long) * self.num_gs_min
        num_gss[proba_mask] = self.num_gs_max
        total_gs = sum(num_gss)
        uid_indices = torch.repeat_interleave(torch.arange(len(num_gss), device=self.device), num_gss)
        gs_triangle_idx = torch.arange(left_triangle_num, left_triangle_num + num_new, device=self.device, dtype=torch.long).unsqueeze(1)[uid_indices].to(self.device)
        gs_offset = torch.zeros(total_gs, 3, device=self.device)
        gs_f_dc = f_dc[uid_indices].to(self.device)
        gs_f_rest = f_rest[uid_indices].to(self.device)
        gs_opacity = opacities[uid_indices].to(self.device)
        gs_scale = init_scales[uid_indices].to(self.device)
        gs_feat = torch.zeros(total_gs, self.args.local_3dgs_feat_dim, device=self.device)
        gs_rotation = torch.zeros((total_gs, 4), device="cuda")
        gs_rotation[:, 0] = 1
        
        gs_uids = torch.arange(
            self.next_gs_uid, 
            self.next_gs_uid + uid_indices.shape[0], 
            device=self.device, 
            dtype=torch.long
        ).unsqueeze(1)
        self.next_gs_uid += uid_indices.shape[0]    

        ## Append the new Gaussians
        keyframe_id = len(self.keyframes) - 1 if keyframe_id == -1 else keyframe_id
        extension_tensors= {
            "triangle_frame_id": torch.full((num_new, 1), keyframe_id, device="cuda", dtype=torch.long),   # Keyframe index associated with each Gaussian.
            "triangle_uid": uids,
            "triangle_state": states,
            "triangle_points": triangle_points,
            "triangle_f_dc": f_dc,
            "triangle_f_rest": f_rest,
            "triangle_opacity": opacities,
            "triangle_delta": deltas,
            "triangle_sigma": sigmas,
            "triangle_feat": feat,
            "gs_uid": gs_uids,
            "gs_triangle_idx": gs_triangle_idx,
            "gs_offset": gs_offset,
            "gs_f_dc": gs_f_dc,
            "gs_f_rest": gs_f_rest,
            "gs_opacity": gs_opacity,
            "gs_scale": gs_scale,
            "gs_rotation": gs_rotation,
            "gs_feat": gs_feat,
        }

        with self.lock:
            self.optimizer.add_and_prune(extension_tensors, valid_triangle_mask, valid_gs_mask)
    
    @torch.no_grad()
    def rigid_transform(self, old_c2ws, new_c2ws):
        triangle_id = self.triangle_frame_id.squeeze(-1)
        gs_id = self.triangle_frame_id[self.gs_triangle_idx.squeeze(-1)].squeeze(-1)
        
        tri_xyz = self.triangle_points
        xyz = self.triangle_barycenters[self.gs_triangle_idx.squeeze(-1)] + self.gs_offset
        rotation = self.gs_rotation

        new_tri_xyz = update_triangles(
            old_c2ws[triangle_id], new_c2ws[triangle_id], tri_xyz
        )
        new_xyz, new_rotation = update_gaussians(
            old_c2ws[gs_id], new_c2ws[gs_id], xyz, rotation
        )

        self.gaussian_params["triangle_points"]["val"] = new_tri_xyz
        self.gaussian_params["gs_offset"]["val"] = new_xyz - self.triangle_barycenters[self.gs_triangle_idx.squeeze(-1)]
        self.gaussian_params["gs_rotation"]["val"] = new_rotation


    def init_intrinsics(self):
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = (
            getProjectionMatrix2(znear=0.01, zfar=100.0, cx=self.K[0, 2],
                                 cy=self.K[1, 2], fx=self.K[0, 0], fy=self.K[1, 1],
                                 W=self.width, H=self.height)
            .transpose(0, 1)
            .to(self.device)
        )

    def add_keyframe(self, keyframe: Keyframe):
        """
            1. Add a keyframe to the scene
            2. Sort all the keyframe indices by the optical center distance from current camera in world camera
            3. add gt pose to viewer
            4. add keyframe to anchor
            5. move keyframe to cpu if necessary
        """

        # Make sure training is not running
        self.join_optimization_thread()

        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device=self.device, dtype=torch.bool)], dim=0
        )
        gt_pose = keyframe.Rt_gt
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.focal_gt if keyframe.focal_gt is not None else self.f

        if not self.inference_mode:
            # Clear memory if there are many keyframes
            if (
                self.n_active_keyframes - self.num_active_frames_cpu
                > self.max_active_keyframes
            ):
                while True:
                    frame_id = np.random.randint(
                        self.first_active_frame, self.last_active_frame - 20
                    )
                    if self.keyframes[frame_id].device.type == "cuda":
                        self.keyframes[frame_id].to("cpu")
                        self.num_active_frames_cpu += 1
                        if self.num_active_frames_cpu % 10 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                        break

    def enable_inference_mode(self):
        """Enable inference mode and sets the anchor position to the mean of the active keyframes."""
        self.inference_mode = True
        for keyframe in self.keyframes:
            keyframe.to(self.device)
        self.num_active_frames_cpu = 0

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.gaussian_params["gs_f_dc"]["val"].shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(
            self.gaussian_params["gs_f_rest"]["val"].shape[1]
            * self.gaussian_params["gs_f_rest"]["val"].shape[2]
        ):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.gaussian_params["gs_scale"]["val"].shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.gaussian_params["gs_rotation"]["val"].shape[1]):
            l.append("rot_{}".format(i))
        return l
    
    def save_ply(self, directory: str):
        xyz = to_numpy(self.triangle_barycenters[self.gs_triangle_idx.squeeze(-1)] + self.gs_offset)
        normals = np.zeros_like(xyz)
        f_dc = to_numpy(
            self.gaussian_params["gs_f_dc"]["val"]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
        )
        f_rest = to_numpy(
            self.gaussian_params["gs_f_rest"]["val"]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
        )
        opacities = to_numpy(self.gaussian_params["gs_opacity"]["val"])    
        scaling = torch.exp(self.gaussian_params["gs_scale"]["val"])
        rotation = self.gaussian_params["gs_rotation"]["val"]
        feat = torch.concat([self.triangle_feat[self.gs_triangle_idx.squeeze(-1)], self.gs_feat], dim=-1)
        scale_rot = self.mlp_cov(feat)            
        scaling = to_numpy(torch.log(torch.sigmoid(scale_rot[:, :3]) * scaling))
        rotation = to_numpy(rotation * scale_rot[:, 3:])

        dtype_org = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scaling, rotation), axis=1
        )

        save_ply(attributes, dtype_org, os.path.join(directory, 'gs.ply'))



    def save_triangle(self, directory: str):
        """Export the unfiltered triangle soup with the original SH0 vertex colors."""
        tri_pts = to_numpy(self.gaussian_params["triangle_points"]["val"])
        vertices = tri_pts.reshape(-1, 3).astype(np.float32)
        f_dc = to_numpy(self.gaussian_params["triangle_f_dc"]["val"].flatten(start_dim=1))
        colors = np.repeat(sh_to_rgb(f_dc).clip(0, 255).astype(np.uint8), 3, axis=0)
        vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                        ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
        for i, field in enumerate(("x", "y", "z")):
            vertex_data[field] = vertices[:, i]
        for i, field in enumerate(("red", "green", "blue")):
            vertex_data[field] = colors[:, i]
        faces = np.empty(len(tri_pts), dtype=[("vertex_indices", "i4", (3,))])
        faces["vertex_indices"] = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        PlyData([PlyElement.describe(vertex_data, "vertex"),
                 PlyElement.describe(faces, "face")]).write(os.path.join(directory, "triangles.ply"))

    @torch.no_grad()
    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0,
             insufficient_match_cnt: int = -1, cholesky_failed_cnt: int = -1,
             from_finetune: bool = False):
        """Save four streaming files, plus selected scene assets after refinement."""
        metrics = {
            "num anchors": 1,
            "num keyframes": len(self.keyframes),
            "insufficient_match_cnt": insufficient_match_cnt,
            "cholesky_failed_cnt": cholesky_failed_cnt,
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        metrics.update(self.evaluate(True, True))
        metrics["num_3dgs"] = self.number_of_gaussians
        metrics["num_triangles"] = self.number_of_triangles
        metrics["average_3dgs_pre_tri"] = float(self.number_of_gaussians) / (float(self.number_of_triangles) + 1e-6)
        if not path:
            return metrics
        os.makedirs(path, exist_ok=True)

        estimated, ground_truth, camera_poses = [], [], []
        for keyframe in sorted(self.keyframes, key=lambda frame: frame.frame_time):
            Twc = torch.linalg.inv(keyframe.get_Rt()).cpu().numpy()
            qw, qx, qy, qz = rotmat2qvec(Twc[:3, :3]).tolist()
            estimated.append([keyframe.frame_time, *Twc[:3, 3].tolist(), qx, qy, qz, qw])
            camera_poses.append(Twc)
            if keyframe.Rt_gt is not None:
                Twc_gt = torch.linalg.inv(keyframe.Rt_gt).cpu().numpy()
                qw, qx, qy, qz = rotmat2qvec(Twc_gt[:3, :3]).tolist()
                ground_truth.append([keyframe.frame_time, *Twc_gt[:3, 3].tolist(), qx, qy, qz, qw])
        np.savetxt(os.path.join(path, "onthefly.txt"), np.asarray(estimated).reshape(-1, 8), fmt="%.10g")
        np.savetxt(os.path.join(path, "gt.txt"), np.asarray(ground_truth).reshape(-1, 8), fmt="%.10g")
        save_poses_as_pyramid_ply(camera_poses, os.path.join(path, "onthefly.ply"), size=0.06, color="red")
        pose_evaluation = evaluate_poses(estimated, ground_truth)
        if pose_evaluation["status"] == "ok":
            metrics["ATE_sim3_rmse_m"] = pose_evaluation["sim3"]["ate_rmse_m"]
            metrics["ATE_se3_rmse_m"] = pose_evaluation["se3"]["ate_rmse_m"]

        metadata = {
            **metrics,
            "stage": "refine" if from_finetune else "stream",
            "finetune_epoch": getattr(self, "export_finetune_epoch", 0) if from_finetune else 0,
            "finetune_iterations": getattr(self, "export_finetune_iterations", 0) if from_finetune else 0,
            "requested_finetune_iterations": self.args.save_at_finetune_iteration,
            "time_definition": "Streaming plus completed refinement compute, excluding exports.",
            "pose_evaluation": pose_evaluation,
            "config": {
                "width": self.width, "height": self.height, "sh_degree": self.max_sh_degree,
                "f": self.f, "global_geo_feat_dim": self.global_geo_feat_dim,
                "local_3dgs_feat_dim": self.local_3dgs_feat_dim,
                "scene_model_name": "plane_v1", "K": self.K.detach().float().cpu().tolist(),
            },
            "keyframes": [frame.to_json() for frame in self.keyframes],
        }
        with open(os.path.join(path, "metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=4, allow_nan=False)
        if not from_finetune:
            return metrics

        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        self.save_ply(pcd_path)
        self.save_triangle(pcd_path)
        mesh = get_depth_fusion_mesh(self, scene_type="plane_v1", voxel_length=0.02, sdf_trunc=0.2)
        mesh_path = os.path.join(pcd_path, "depth_fusion_mesh.ply")
        if len(mesh.vertices) == 0 or not o3d.io.write_triangle_mesh(mesh_path, mesh):
            raise RuntimeError(f"Failed to export a nonempty depth fusion mesh: {mesh_path}")
        self.save_test_frames(os.path.join(path, "test_images"))

        # Optional all-frame images and videos: uncomment this call to restore both.
        # self.save_all_frames(os.path.join(path, "all_images"))

        # Optional triangle postprocessing: uncomment this complete block to restore it.
        # postprocess_stats = self.postprocess_triangles_for_psdr(
        #     path=os.path.join(path, "postprocess_triangles"),
        #     impact_per_image_thr=getattr(self.args, "postprocess_impact_thr", 20),
        #     opacity_thr=getattr(self.args, "postprocess_opacity_thr", 0.5),
        #     n_samples_per_triangle=getattr(self.args, "postprocess_n_samples_per_triangle", 11),
        #     enable_upsampling=getattr(self.args, "postprocess_triangles_enable_upsampling", True),
        # )
        # print(f"Postprocess triangles stats: {postprocess_stats}")
        return metrics

    def get_closest_keyframe(
            self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.reset_optimizer()
        for key, param_group in self.optimizer.params.items():
            if key in self.optimizer.lr_dict:    # actually is triangle_points
                if key.startswith("mlp"):
                    param_group["lr"] = self.optimizer.lr_dict[key]["lr_init"]
                elif key == "triangle_points":
                    num_triangles = self.gaussian_params["triangle_points"]["val"].shape[0]
                    new_lr = torch.full(
                        (num_triangles, 3, 3),
                        self.optimizer.lr_dict[key]["lr_init"],
                        dtype=torch.float,
                        device=self.device
                    )
                    param_group["lr"] = new_lr
                elif key == "gs_offset":
                    num_gs = self.gaussian_params["gs_offset"]["val"].shape[0]
                    new_lr = torch.full(
                        (num_gs, 3),
                        self.optimizer.lr_dict[key]["lr_init"],
                        dtype=torch.float,
                        device=self.device
                    )
                    param_group["lr"] = new_lr

        # Optimize the anchor by going through its keyframes
        for _ in range(len(self.keyframes)):
            self.optimization_step(finetuning=True)

    # Optional triangle postprocessing for external plane extraction.
    # Orient normals consistently and prune low-contribution triangles.
    # Optionally sample points on the retained triangle surfaces.
    @torch.no_grad()
    def postprocess_triangles_for_psdr(
        self, 
        path, 
        impact_per_image_thr=20, 
        opacity_thr=0.5,
        n_samples_per_triangle=11,
        account_impact_per_image_thr=20,
        flip_ratio_thr=0.5,
        enable_upsampling=True
    ):
        triangle_normals = self.triangle_normal.detach().clone()  # [N, 3]
        triangle_barycenters = self.triangle_barycenters.detach()  # [N, 3]
        triangle_points = self.triangle_points.detach()  # [N, 3, 3]
        triangle_opacity = self.triangle_opacity.detach()  # [N, 1]
        
        num_triangles = triangle_normals.shape[0]
        
        
        triangle_acc_impact = torch.zeros(num_triangles, dtype=torch.float32, device=self.device)
        triangle_acc_denom = torch.zeros(num_triangles, dtype=torch.float32, device=self.device)
        triangle_acc_denom_for_flip = torch.zeros(num_triangles, dtype=torch.float32, device=self.device)
        triangle_acc_times_need_flip = torch.zeros(num_triangles, dtype=torch.float32, device=self.device)
        
        
        train_keyframes = [kf for kf in self.keyframes if not kf.is_test]
        print(f"Processing {len(train_keyframes)} training keyframes for triangle postprocessing...")
        
        
        from tqdm import tqdm
        for keyframe in tqdm(train_keyframes, desc="Processing keyframes"):
            try:
                
                render_pkg_triangle, _ = self.render_from_id(keyframe.index, pyr_lvl=0)
                
                
                if "impact" in render_pkg_triangle:
                    convex_impact = render_pkg_triangle["impact"]  # [N]
                    convex_update_filter = render_pkg_triangle.get("visibility_filter", torch.ones(num_triangles, dtype=torch.bool, device=self.device))
                    
                    # Accumulate impact and the number of contributing views.
                    triangle_acc_impact += convex_impact
                    triangle_acc_denom[convex_update_filter] += 1.0
                    
                    view_matrix = keyframe.get_Rt().transpose(0, 1).to(self.device)
                    world_view_transform = view_matrix
                    
                    # Transform triangle normals into camera coordinates.
                    triangle_normals_camera = torch.matmul(triangle_normals, world_view_transform[:3, :3])
                    
                    # Transform triangle centroids into camera coordinates.
                    triangle_barycenters_camera = (
                        torch.matmul(triangle_barycenters, world_view_transform[:3, :3])
                        + world_view_transform.T[:3, 3]
                    )
                    
                    # Use the normal-centroid dot product to determine orientation.
                    dot_products = (triangle_barycenters_camera * triangle_normals_camera).sum(dim=-1)
                    
                    flip_seen_mask = convex_impact > account_impact_per_image_thr
                    flip_mask = (dot_products > 0) & flip_seen_mask
                    
                    triangle_acc_denom_for_flip[flip_seen_mask] += 1
                    triangle_acc_times_need_flip[flip_mask] += 1
            except Exception as e:
                print(f"Warning: Failed to process keyframe {keyframe.index}: {e}")
                continue
        
        # Flip normals when the fraction of supporting views exceeds the threshold.
        flip_ratio = triangle_acc_times_need_flip / (triangle_acc_denom_for_flip + 0.0001)
        true_flip_mask = flip_ratio > flip_ratio_thr
        triangle_normals[true_flip_mask] *= -1
        
        print(f"Flipped {true_flip_mask.sum().item()} triangles based on flip ratio")
        
        # Compute mean impact per contributing image.
        triangle_impact_per_image = triangle_acc_impact / (triangle_acc_denom + 0.0001)
        triangle_impact_per_image[triangle_acc_denom < 1.0] = 0.0
        
        print(f"triangle_impact_per_image mean: {triangle_impact_per_image.mean().item()}")
        print(f"triangle_impact_per_image max: {triangle_impact_per_image.max().item()}")
        
        # Prune triangles with low mean impact or opacity.
        prune_mask_low_impact = triangle_impact_per_image < impact_per_image_thr
        prune_mask_low_opacity = (triangle_opacity < opacity_thr).squeeze(-1)
        postprocess_prune_mask = torch.logical_or(prune_mask_low_impact, prune_mask_low_opacity)
        
        print(f"prune_mask_low_impact sum: {prune_mask_low_impact.sum().item()}")
        print(f"prune_mask_low_opacity sum: {prune_mask_low_opacity.sum().item()}")
        print(f"postprocess_prune_mask sum: {postprocess_prune_mask.sum().item()}")
        
        
        keep_mask = ~postprocess_prune_mask
        triangle_points_keep = triangle_points[keep_mask].detach().cpu().numpy()  # [M, 3, 3]
        triangle_normals_keep = triangle_normals[keep_mask].detach().cpu().numpy()  # [M, 3]
        triangle_opacity_keep = triangle_opacity[keep_mask].detach().cpu().numpy()  # [M, 1]
        triangle_impact_keep = triangle_impact_per_image[keep_mask].detach().cpu().numpy()  # [M]

        triangle_normals_normalized = triangle_normals_keep / (np.linalg.norm(triangle_normals_keep, axis=1, keepdims=True) + 1e-8)
        # Map normal components from [-1, 1] to RGB values in [0, 255].
        triangle_colors_from_normal = (triangle_normals_normalized + 1.0) * 0.5 * 255
        triangle_colors_from_normal = np.clip(triangle_colors_from_normal, 0, 255).astype(np.uint8)
        
        num_triangles_keep = triangle_points_keep.shape[0]
        
        
        os.makedirs(path, exist_ok=True)
        output_triangle_path = os.path.join(path, f"postprocess_triangle_{impact_per_image_thr}_{opacity_thr}.ply")
        self._save_triangles_to_ply(
            output_triangle_path, 
            triangle_points_keep, 
            triangle_normals_keep, 
            triangle_colors_from_normal
        )
        print(f"Saved postprocessed triangles to {output_triangle_path}")
        
        # Optionally sample points on each retained triangle.
        if enable_upsampling:
            sampled_points = []
            sampled_normals = []
            sampled_colors = []
            centroids = []
            
            for i in range(num_triangles_keep):
                v0, v1, v2 = triangle_points_keep[i]
                n = triangle_normals_keep[i]
                c = triangle_colors_from_normal[i]
                
                centroid = (v0 + v1 + v2) / 3.0
                centroids.append(centroid)
                
                
                points = self._sample_points_on_triangle(v0, v1, v2, n_samples_per_triangle)
                sampled_points.append(points)
                sampled_normals.append(np.tile(n, (n_samples_per_triangle, 1)))
                sampled_colors.append(np.tile(c, (n_samples_per_triangle, 1)))
                
                # Include the three original vertices of each triangle.
                sampled_points.append([v0, v1, v2])
                sampled_normals.append([n, n, n])
                sampled_colors.append([c, c, c])
            
            
            sampled_points = np.vstack(sampled_points)
            sampled_normals = np.vstack(sampled_normals)
            sampled_colors = np.vstack(sampled_colors)
            centroids = np.array(centroids)
            
            
            all_points = np.vstack([centroids, sampled_points])
            all_normals = np.vstack([np.tile(triangle_normals_keep, (1, 1)), sampled_normals])
            all_colors = np.vstack([np.tile(triangle_colors_from_normal, (1, 1)), sampled_colors])
            
            
            output_points_path = os.path.join(path, f"upsampling_post_points_{impact_per_image_thr}_{opacity_thr}_{n_samples_per_triangle}.ply")
            self._save_points_to_ply(output_points_path, all_points, all_normals, all_colors)
            print(f"Saved upsampled points to {output_points_path}")
            
            num_points_upsampled = all_points.shape[0]
        else:
            num_points_upsampled = 0
        
        return {
            "num_triangles_original": num_triangles,
            "num_triangles_kept": num_triangles_keep,
            "num_points_upsampled": num_points_upsampled,
            "flip_ratio": flip_ratio[keep_mask].mean().item() if keep_mask.any() else 0.0,
        }
    
    def _sample_points_on_triangle(self, v0, v1, v2, n):
        u = np.random.rand(n, 1)
        v = np.random.rand(n, 1)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        sampled_points = (1 - u - v) * v0 + u * v1 + v * v2
        return sampled_points
    
    def _save_triangles_to_ply(self, file_path, triangle_points, triangle_normals, triangle_colors):        
        num_triangles = triangle_points.shape[0]
        vertices = triangle_points.reshape(-1, 3)  # [3N, 3]
        faces = np.arange(num_triangles * 3, dtype=np.int32).reshape(num_triangles, 3)
        
        
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        vertex_data = np.empty(vertices.shape[0], dtype=vertex_dtype)
        vertex_data['x'] = vertices[:, 0].astype(np.float32)
        vertex_data['y'] = vertices[:, 1].astype(np.float32)
        vertex_data['z'] = vertices[:, 2].astype(np.float32)
        
        # Repeat each triangle normal and color for its three vertices.
        normals_repeated = np.repeat(triangle_normals, 3, axis=0)
        colors_repeated = np.repeat(triangle_colors, 3, axis=0)
        vertex_data['nx'] = normals_repeated[:, 0].astype(np.float32)
        vertex_data['ny'] = normals_repeated[:, 1].astype(np.float32)
        vertex_data['nz'] = normals_repeated[:, 2].astype(np.float32)
        vertex_data['red'] = colors_repeated[:, 0]
        vertex_data['green'] = colors_repeated[:, 1]
        vertex_data['blue'] = colors_repeated[:, 2]
        
        
        face_dtype = [('vertex_indices', 'i4', (3,))]
        face_data = np.empty(num_triangles, dtype=face_dtype)
        face_data['vertex_indices'] = faces
        
        ply = PlyData([
            PlyElement.describe(vertex_data, 'vertex'),
            PlyElement.describe(face_data, 'face')
        ])
        ply.write(file_path)
    
    def _save_points_to_ply(self, file_path, points, normals, colors):
        vertex_data = np.array(
            [
                (p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2])
                for p, n, c in zip(points, normals, colors)
            ],
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ]
        )
        ply_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([ply_element]).write(file_path)    
        
        
