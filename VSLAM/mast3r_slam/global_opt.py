import pypose as pp
import torch
from VSLAM.SharedKeyframes import SharedKeyframes
from VSLAM.mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from VSLAM.utils_mast3r import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, config, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.config = config
        self.cfg = config["local_opt"]
        # First endpoint of each factor-graph edge.
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        # Second endpoint of each factor-graph edge.
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        # Image-i matches in image-j pixel order.
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        # Image-j matches in image-i pixel order.
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        # Validity of image-i matches in image-j pixel order.
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        # Validity of image-j matches in image-i pixel order.
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        # Feature confidence for correspondences in camera-i coordinates.
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        # Feature confidence for correspondences in camera-j coordinates.
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)

        self.window_size = self.cfg["window_size"]

        self.K = K


    # Add factors, retaining temporal neighbors even when match coverage is low.
    def add_factors(self, ii, jj, min_match_frac, embeddings, is_reloc=False):  
        
        # The backend passes historical keyframes as ii and the current keyframe as jj.
        kf_ii = [self.frames[idx].to(self.device) for idx in ii]
        kf_jj = [self.frames[idx].to(self.device) for idx in jj]
        feat_i = torch.cat([embeddings[idx][0] for idx in ii])
        feat_j = torch.cat([embeddings[idx][0] for idx in jj])
        pos_i = torch.cat([embeddings[idx][1] for idx in ii])
        pos_j = torch.cat([embeddings[idx][1] for idx in jj])
        shape_i = [torch.tensor(kf_i.img.shape[1:])[None] for kf_i in kf_ii]
        shape_j = [torch.tensor(kf_j.img.shape[1:])[None] for kf_j in kf_jj]

        (
            idx_i2j,  # Image-i pixel indices in image-j pixel order.
            idx_j2i,  # Image-j pixel indices in image-i pixel order.
            valid_match_j,  # Validity mask for idx_i2j.
            valid_match_i,  # Validity mask for idx_j2i.
            Qii,  # Feature confidence for image i in camera i coordinates.
            Qjj,  # Feature confidence for image j in camera j coordinates.
            Qji,  # Feature confidence for image j in camera i coordinates.
            Qij,  # Feature confidence for image i in camera j coordinates.
        ) = mast3r_match_symmetric(self.config,
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        ) 

        
        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)
        valid_Qj = Qj > self.cfg["Q_conf"]   # 1.5
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi

        
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj 
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni 

        # Reject low-coverage edges unless their endpoints are temporally consecutive.
        
        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)
        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac   # 0.1
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges    # Preserve the edge to the immediately preceding keyframe.
        
        if invalid_edges.any() and is_reloc:
            return False, None

        
        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0   
        return added_new_edges, ii_tensor 

    
    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    
    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx].to(self.device) for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])  # 
        T_WCs = pp.quat2unit(pp.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))) 
        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    
    # Represent each symmetric match as two directed factors.
    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def solve_GN_rays(self):

        
        # Expand symmetric correspondences into two directed factors.
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return
        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        
        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]
        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        
        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]  # default 1
        unique_kf_idx = self.get_unique_kf_idx() 
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()  # Concatenate both directions of every edge for optimization.

        C_thresh = self.cfg["C_conf"] # 0.0
        Q_thresh = self.cfg["Q_conf"] # 1.5
        pixel_border = self.cfg["pixel_border"] # -10
        z_eps = self.cfg["depth_eps"] # 1e-6
        max_iter = self.cfg["max_iters"] # 10
        sigma_pixel = self.cfg["sigma_pixel"] # 1.0
        sigma_depth = self.cfg["sigma_depth"] # 1e+1
        delta_thresh = self.cfg["delta_norm"] # 1e-8

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size
        mast3r_slam_backends.gauss_newton_calib(
            pose_data,  
            Xs,  
            Cs,  
            K,  
            ii,  # Source endpoint of each directed factor.
            jj,  # Target endpoint of each directed factor.
            idx_ii2jj,  # Source-image matches in target-image pixel order.
            valid_match,  # Match-validity mask combined with the feature-confidence threshold.
            Q_ii2jj,   # Correspondence confidence.
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
        # set all scale to 1.0
        

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
        
        
    # add for support SE(3)
    #  calib_proj mode
    def solve_GN_calib_se3(self):
        """SE(3) version of calibrated Gauss-Newton optimization"""
        K = self.K
        pin = self.cfg["pin"]   # default 1
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        # Convert Sim(3) to SE(3) format (remove scale, keep only 7 DOF: 3 trans + 4 quat)
        pose_data = T_WCs.data[:, 0, :7].contiguous()  # Remove scale component

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size
        
        
        # Record poses before optimization for diagnostics.
        pose_before = pose_data.clone()
        
            
            
        # Call SE(3) version
        mast3r_slam_backends.gauss_newton_calib_se3(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
        
        # Measure the pose change after optimization.
        pose_change = torch.norm(pose_data - pose_before, dim=1)
        # print(f"Pose changes: {pose_change}")
        
        
        if torch.any(pose_change > 1.0):  
            print("[WARNING] Large pose changes detected!")

        # Update the keyframe T_WC (SE(3) format)
        # Convert back to Sim(3) format with scale=1
        T_WCs_updated = T_WCs.clone()
        T_WCs_updated.data[pin:, 0, :7] = pose_data[pin:]
        T_WCs_updated.data[pin:, 0, 7] = 1.0  # Set scale to 1 for SE(3)
        
        self.frames.update_T_WCs(T_WCs_updated[pin:], unique_kf_idx[pin:])
