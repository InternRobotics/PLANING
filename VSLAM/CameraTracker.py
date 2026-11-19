
import pypose as pp
import torch
from VSLAM.ImageFrame import ImageFrame
from VSLAM.mast3r_slam.geometry import (
    act_Sim3, act_SE3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from VSLAM.mast3r_slam.nonlinear_optimizer import check_convergence, huber
from VSLAM.utils_mast3r import mast3r_match_asymmetric, mast3r_inference_mono
from VSLAM.utils_uncertainty import local_diag_cov_from_X1, visualize_covariance_max, local_cov_from_X1

class CameraTracker:
    def __init__(self, config, min_displacement, thres_keyframe,
                 model, frames,
                 H_slam, W_slam, K_slam, device, lidar_processor, use_SE3_to_optimize_pose,
                 insufficient_match_cnt=None, cholesky_failed_cnt=None, covariance_filter=False):


        self.config = config
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device
        self.H_slam = H_slam
        self.W_slam = W_slam
        self.K_slam = K_slam
        self.min_displacement = min_displacement
        self.thres_keyframe = thres_keyframe
        
        self.lidar_processor = lidar_processor
        self.use_SE3_to_optimize_pose = use_SE3_to_optimize_pose
        # Reuse the shared failure counter when provided.
        if insufficient_match_cnt is not None:
            self.insufficient_match_cnt = insufficient_match_cnt
        else:
            self.insufficient_match_cnt = 0
            
        if cholesky_failed_cnt is not None:
            self.cholesky_failed_cnt = cholesky_failed_cnt
        else:
            self.cholesky_failed_cnt = 0
            
        self.covariance_filter = covariance_filter
            
    
        self.last_embedding = None
        self.last_dist = 0
        self.reset_idx_f2k()
        

    def track_init(self, frame: ImageFrame):
        X_init, C_init,  feat, pos = mast3r_inference_mono(self.model, frame)
        frame.update_pointmap(X_init, C_init, lidar_processor=self.lidar_processor)   # Initialize the point map in the first camera coordinate system.
        self.last_embedding = [feat, pos]
        return False, True, True

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None  # Current-frame pixel indices, ordered by reference-keyframe pixels.

    
    def track(self, frame: ImageFrame):
        if frame.frame_id == 0:
            return self.track_init(frame)

        keyframe = self.keyframes.last_keyframe().to(self.device)   
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf, featf, posf = (
            mast3r_match_asymmetric(self.config,self.model, frame, keyframe,
                                    idx_i2j_init=self.idx_f2k, embeddings_j = self.last_embedding)) # Reuse the reference-keyframe embedding.
        self.idx_f2k = idx_f2k.clone()   
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0] 
        Qk = torch.sqrt(Qff[idx_f2k] * Qkf) # Geometric-mean feature confidence, ordered by keyframe pixels.

        use_calib = self.config["use_calib"]
        frame.update_pointmap(Xff, Cff, lidar_processor=self.lidar_processor)   
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None
        # Get poses and point correspondneces and confidences
        (Xf, Xf_cov, Xk, Xk_cov, T_WCf, T_WCk, Cf, Ck,
         meas_k, valid_meas_k) = self.get_points_poses(frame, keyframe, idx_f2k, img_size)

        valid_Cf = Cf > self.cfg["C_conf"] # default 0.0
        valid_Ck = Ck > self.cfg["C_conf"] # default 0.0
        valid_Q = Qk > self.cfg["Q_conf"]  # default 1.5, evaluate the match score 
        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q # Optimization mask: valid matches with sufficient point and feature confidence.
        valid_kf = valid_match_k & valid_Q  # Keyframe-selection mask: valid matches with sufficient feature confidence.

        
        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:   
            if hasattr(self.insufficient_match_cnt, 'value'):
                self.insufficient_match_cnt.value += 1
                print(f"Insufficient match {frame.frame_id}, Insufficient times: {self.insufficient_match_cnt.value}")
            else:
                self.insufficient_match_cnt += 1
                print(f"Insufficient match {frame.frame_id}, Insufficient times: {self.insufficient_match_cnt}")
            return True, False, False     

        try:
            # Track
            if not use_calib:
                
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                if self.use_SE3_to_optimize_pose:
                    T_WCf, T_CkCf = self.opt_pose_calib_SE3(
                        Xf, Xf_cov, Xk, Xk_cov,T_WCf,T_WCk,Qk,valid_opt,
                        meas_k,               # Keyframe pixel coordinates and log depth.
                        valid_meas_k,         # Validity mask for keyframe depth measurements.
                        K,img_size,
                    )
                else:
                    T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                        Xf, Xf_cov, Xk, Xk_cov,T_WCf,T_WCk,Qk,valid_opt,
                        meas_k,               # Keyframe pixel coordinates and log depth.
                        valid_meas_k,         # Validity mask for keyframe depth measurements.
                        K,img_size,
                    )                         
                      
        except Exception as e:
            if hasattr(self.cholesky_failed_cnt, 'value'):
                self.cholesky_failed_cnt.value += 1
                print(f"Cholesky failed {frame.frame_id}, Fail times: {self.cholesky_failed_cnt.value}, error: {e}")
            else:
                self.cholesky_failed_cnt += 1
                print(f"Cholesky failed {frame.frame_id}, Fail times: {self.cholesky_failed_cnt}, error: {e}")
            return True, False, False
        T_WCf = pp.quat2unit(T_WCf)
        frame.T_WC = T_WCf    

        
        is_keyframe = self.check_keyframe(idx_f2k, valid_kf, valid_match_k)
        if is_keyframe:
            self.reset_idx_f2k()
            self.last_embedding = [featf, posf]
            is_keyframe_map = True   
            self.last_dist = 0
        else:
            is_keyframe_map, dist = self.check_keyframe_map(idx_f2k, valid_opt)
            if is_keyframe_map:
                self.last_dist = dist

        return False, is_keyframe, is_keyframe_map
    
    def check_keyframe(self, idx_f2k, valid_kf, valid_match_k):
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()  # Fraction of reference-keyframe pixels with valid matches.
        # Measure correspondence uniqueness in the current frame.
        unique_frac_f = (
                torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )
        add_new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"] # 0.333
        return add_new_kf


    def check_keyframe_map(self, idx_f2k, valid_opt):
        uf = idx_f2k % self.W_slam
        vf = idx_f2k // self.W_slam
        uvf = torch.stack([uf, vf], dim=-1)  # (H*W, 2): current-frame pixels matched to reference-keyframe pixels.

        uk, vk = torch.meshgrid(torch.arange(self.W_slam).to(self.device),
                                torch.arange(self.H_slam).to(self.device),
                                indexing="xy")
        uk = uk.reshape(-1)
        vk = vk.reshape(-1)
        uvk = torch.stack([uk, vk], dim=-1)     # (H*W, 2): reference-keyframe pixel grid.
        dist = torch.norm((uvf - uvk).float(), dim=-1)[valid_opt.squeeze(-1)]  # Pixel displacement of valid correspondences.

        dist_quantile = torch.quantile(dist, self.thres_keyframe).item()   # 0.8
        add_new_kf = (dist_quantile - self.last_dist) > self.min_displacement # 0.03

        return add_new_kf, dist_quantile

    # Return matched points, poses, confidences, and depth validity in keyframe pixel order.
    def get_points_poses(self, frame, keyframe, idx_f2k, img_size):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC
        T_WCf = pp.quat2unit(T_WCf)
        T_WCk = pp.quat2unit(T_WCk)

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        # Backproject predicted depths using the calibrated pixel rays.
        Xf = constrain_points_to_ray(img_size, Xf[None], self.K_slam).squeeze(0)
        Xk = constrain_points_to_ray(img_size, Xk[None], self.K_slam).squeeze(0)
        
        Xf_cov, _ = local_cov_from_X1(Xf, self.H_slam, self.W_slam)
        Xk_cov, _ = local_cov_from_X1(Xk, self.H_slam, self.W_slam)

        # Setup pixel coordinates
        uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
        uv_k = uv_k.view(-1, 2)
        meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)

        # Avoid any bad calcs in log
        valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
        meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xf_cov[idx_f2k], Xk, Xk_cov, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost
    
    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)
        T_CkCf = T_WCk.Inv().mul(T_WCf)
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        
        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            # Transform current-frame points into the keyframe coordinate system.
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            r = rd_k - rd_f_Ck
            # Differentiate the residual with respect to the relative Sim(3) pose.
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.add(tau_ij_sim3)
            T_CkCf = pp.quat2unit(T_CkCf)
            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk.mul(T_CkCf)

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(self, Xf, Xf_cov, Xk, Xk_cov,
                            T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size):
        device = Xf.device
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)  
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)   

        T_CkCf = T_WCk.Inv().mul(T_WCf)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)   
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )

            
            # Project 3D covariance into pixel/log-depth space and filter large determinants.
            if self.covariance_filter:
                R_CkCf = T_CkCf.matrix()[:, :3, :3]
                Xfk_cov = R_CkCf @ Xf_cov @ R_CkCf.transpose(-1, -2)
                X, Y, Z = Xf_Ck.unbind(-1)
                o = torch.zeros_like(X)
                JCOV = torch.stack([self.K_slam[0,0]/Z, o, -self.K_slam[0,0]*X/(Z**2),
                                 o, self.K_slam[1,1]/Z, -self.K_slam[1,1]*Y/(Z**2),
                                 o, o, 1/Z], dim=-1).reshape(-1, 3, 3)
                pixfk_cov = JCOV @ Xfk_cov @ JCOV.transpose(-1, -2)
                pixfk_cov_det = torch.det(pixfk_cov)
                valid_cov = pixfk_cov_det < max(torch.quantile(pixfk_cov_det, 0.9), 1)
                valid_cov = valid_cov[..., None]
            else:
                valid_cov = torch.ones_like(valid_meas_k)
            
            valid2 = valid_proj & valid_meas_k & valid_cov # Combine projection, keyframe-depth, and covariance validity masks.
            sqrt_info2 = valid2 * sqrt_info
            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf


            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = pp.sim3(tau_ij_sim3).Exp().mul(T_CkCf)
            T_CkCf = pp.quat2unit(T_CkCf)

            if check_convergence(
                step,
                self.cfg["rel_error"], 
                self.cfg["delta_norm"], 
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk.mul(T_CkCf)
        return T_WCf, T_CkCf

    def opt_pose_calib_SE3(self,Xf, Xf_cov, Xk, Xk_cov, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size):
 
        device = Xf.device
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        if T_WCf.data.shape[-1] == 8:
            T_WCf_SE3 = pp.SE3(T_WCf.data[:, 0:7])
            T_WCk_SE3 = pp.SE3(T_WCk.data[:, 0:7])
            scale_WCf = T_WCf.data[:, 7]
            scale_WCk = T_WCk.data[:, 7]
            
            if torch.abs(scale_WCf - 1.0) > 1e-4 or torch.abs(scale_WCk - 1.0) > 1e-4:
                print("ERROR: scale is not 1, Can not use SE3 to optimize pose!")
                print(f"scale_WCf: {scale_WCf}, scale_WCk: {scale_WCk}")
                exit()
                
        else:
            T_WCf_SE3 = T_WCf
            T_WCk_SE3 = T_WCk
            scale_WCf = 1.0
            scale_WCk = 1.0
            
        T_CkCf_SE3 = T_WCk_SE3.Inv().mul(T_WCf_SE3)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_SE3(T_CkCf_SE3, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            
            
            # Project 3D covariance into pixel/log-depth space and filter large determinants.
            if self.covariance_filter:
                R_CkCf = T_CkCf_SE3.matrix()[:, :3, :3]
                Xfk_cov = R_CkCf @ Xf_cov @ R_CkCf.transpose(-1, -2)
                X, Y, Z = Xf_Ck.unbind(-1)
                o = torch.zeros_like(X)
                JCOV = torch.stack([self.K_slam[0,0]/Z, o, -self.K_slam[0,0]*X/(Z**2),
                                 o, self.K_slam[1,1]/Z, -self.K_slam[1,1]*Y/(Z**2),
                                 o, o, 1/Z], dim=-1).reshape(-1, 3, 3)
                pixfk_cov = JCOV @ Xfk_cov @ JCOV.transpose(-1, -2)
                pixfk_cov_det = torch.det(pixfk_cov)
                valid_cov = pixfk_cov_det < max(torch.quantile(pixfk_cov_det, 0.9), 1)
                valid_cov = valid_cov[..., None]
            else:
                valid_cov = torch.ones_like(valid_meas_k)
            
            valid2 = valid_proj & valid_meas_k & valid_cov # Combine projection, keyframe-depth, and covariance validity masks.
            sqrt_info2 = valid2 * sqrt_info
            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf
            
            tau_ij_se3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf_SE3 = pp.se3(tau_ij_se3).Exp().mul(T_CkCf_SE3)
            T_CkCf_SE3 = pp.quat2unit(T_CkCf_SE3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"] * 6.0 / 7.0,   # maybe right, sim3 --> se3
                old_cost,
                new_cost,
                tau_ij_se3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf_SE3 = T_WCk_SE3.mul(T_CkCf_SE3)

        if T_WCf.data.shape[-1] == 8:  # Sim3 
            T_WCf = pp.Sim3(torch.cat((T_WCf_SE3.data[:, 0:7], torch.ones(1, 1, device=T_WCf_SE3.data.device, dtype=T_WCf_SE3.data.dtype) * scale_WCf), dim=-1))
            T_CkCf = pp.Sim3(torch.cat([T_CkCf_SE3.data[:, 0:7], torch.ones(1, 1, device=T_CkCf_SE3.data.device, dtype=T_CkCf_SE3.data.dtype)], dim=-1))  # (1, 8)
        else: # SE3
            T_WCf = T_WCf_SE3
            T_CkCf = T_CkCf_SE3        
        return T_WCf, T_CkCf
