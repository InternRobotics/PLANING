import pathlib
import time

import numpy as np
import pypose as pp
from tqdm import tqdm
import torch
import json
from VSLAM.CameraTracker import CameraTracker
import VSLAM.mast3r_slam.evaluate as eval
from VSLAM.ImageFrame import Mode, ImageFrame
from VSLAM.utils_mast3r import load_mast3r

from Reconstruct.utils import save_poses_as_pyramid_ply
import os


class Frontend:
    @torch.no_grad()
    def __init__(self, args, config,
                 dataset,
                 keyframes, states,
                 device="cuda:0",
                 lidar_processor=None,
                 insufficient_match_cnt=None,
                 cholesky_failed_cnt=None):
        self.args = args
        self.config = config
        self.use_calib = config["use_calib"]
        self.device = device
        self.dataset = dataset
        (self.H_slam,
         self.W_slam,
         self.K_slam) = (dataset.H_slam,
                         dataset.W_slam,
                         dataset.K_slam.to(device))

        self.keyframes = keyframes 
        self.states = states 
        self.model = load_mast3r(device=self.device)  
        self.model.share_memory()
        self.covariance_filter = args.covariance_filter
        
        self.lost_time = 0
        
        self.lidar_processor = lidar_processor
        self.use_SE3_to_optimize_pose = args.use_SE3_to_optimize_pose
        self.insufficient_match_cnt = insufficient_match_cnt
        self.cholesky_failed_cnt = cholesky_failed_cnt
        self.min_displacement = max(self.args.min_displacement * self.W_slam, 30)
        self.tracker = CameraTracker(self.config, self.min_displacement, args.thres_keyframe,
                                     self.model, self.keyframes,
                                     self.H_slam, self.W_slam, self.K_slam,
                                     self.device,
                                     self.lidar_processor, self.use_SE3_to_optimize_pose,
                                     insufficient_match_cnt=self.insufficient_match_cnt,
                                     cholesky_failed_cnt=self.cholesky_failed_cnt,
                                     covariance_filter=self.covariance_filter,
                                     )

        self.frames_info = []   
        self.frames_Twc_gt = []
        self._T_hist = []


    def init_pose_per_frame(self, current_idx, Twc_gt=None, if_use_uniform_motion=False):
        if current_idx == 0:
            T_WC = pp.identity_Sim3(1, device=self.device) 
        else:
            if if_use_uniform_motion:   
                if len(self._T_hist) >= 2:
                    T_prev = self._T_hist[-2]
                    T_last = self._T_hist[-1]
                    if T_last.data.shape[-1] == 8:
                        T_prev_se3 = pp.SE3(T_prev.data[:, :7])
                        T_last_se3 = pp.SE3(T_last.data[:, :7])
                        delta_se3 = T_prev_se3.Inv().mul(T_last_se3)
                        T_pred_se3 = T_last_se3.mul(delta_se3)         
                        s_last = T_last.data[:, 7:8]                   
                        T_WC = pp.Sim3(torch.cat([T_pred_se3.data[:, :7], s_last], dim=-1))
                    else:  
                        delta_se3 = T_prev.Inv().mul(T_last)
                        T_WC = T_last.mul(delta_se3)      
                else:
                    T_WC = pp.Sim3(self.states.T_WC).to(self.device)

            else:
                T_WC = pp.Sim3(self.states.T_WC).to(self.device)
    
    
        if Twc_gt is not None:
            T_WC.data[:, :7] = Twc_gt.data[:, :7]
        T_WC = pp.quat2unit(T_WC)
        return T_WC

    @torch.no_grad()
    def run(self):
        frameID = 0
        bar = tqdm(total=len(self.dataset), desc=f"Frontend: ", position=0, leave=True)
        while True:
            try:
                original_image, info = self.dataset.getnext()
            except Exception as e:
                break
            is_test = info["is_test"]
            timestamp = info["timestamp"]
            img_slam = self.dataset.transform.to_slam(original_image, device=self.device)

            Twc_gt = None 
            if "Twc_gt" in info:
                x, y, z, qx, qy, qz, qw = info["Twc_gt"]
                self.frames_Twc_gt.append([timestamp, x, y, z, qx, qy, qz, qw])
                if self.args.use_gt_pose:
                    Twc_gt = pp.Sim3(torch.tensor([[x, y, z, qx, qy, qz, qw, 1]]))
                    Twc_gt = pp.quat2unit(Twc_gt).to(self.device)
            T_WC = self.init_pose_per_frame(frameID, Twc_gt, if_use_uniform_motion=self.args.use_uniform_motion) 
            frame = ImageFrame(frameID, 0, float(timestamp), img_slam,
                               T_WC, K=self.K_slam if self.K_slam is not None else None)          
            if self.lidar_processor is not None:
                frame.add_LiDAR_info(info["lidar_frame_id"], info["lidar_timestamp"], info["inter_lidar_pose_L0_Lt"])
            lost, is_keyframe, is_keyframe_map = self.tracker.track(frame)    
            
            if lost:
                self.lost_time += 1   
                print(f"[LOST Tracking!!!] Lost time: {self.lost_time}; FrameID: {frameID}")
            
            if not lost:
                self._T_hist.append(pp.Sim3(frame.T_WC))
                if len(self._T_hist) > 2:
                    self._T_hist = self._T_hist[-2:]       
            
            if self.args.use_same_set_of_keyframes:  
                is_keyframe = is_keyframe or is_keyframe_map

            keyframe_style = -1
            if lost:
                keyframe_style = 0
            elif is_keyframe:
                
                self.keyframes.append(frame)   
                keyframe_style = 1
            elif is_keyframe_map or is_test or self.args.use_all_frames: 
                
                keyframe = self.keyframes.last_keyframe().to(self.device)
                self.store_relative_pose(frame, len(self.keyframes) - 1, keyframe.T_WC.Inv().mul(frame.T_WC))
                keyframe_style = 2
            else:  
                keyframe = self.keyframes.last_keyframe().to(self.device)
                self.store_relative_pose(frame, len(self.keyframes) - 1, keyframe.T_WC.Inv().mul(frame.T_WC))
                
            if keyframe_style != -1:    
                keyframe_dict = {
                    "keyframe_style": keyframe_style,
                    "is_important": is_keyframe_map or is_test,
                    "is_test": is_test,
                    "keyframe_id": len(self.keyframes) - 1,
                    "frame_id": frame.frame_id,
                    "T_WC": frame.T_WC.data.cpu(),
                }
                self.states.msg2Backend(keyframe_dict)
                self.states.queue_backend_execute()   
            self.states.set_frame(frame)

            if self.args.sync_hard:
                while True:
                    with self.states.lock:
                        if self.states.backend_execute.value == 0:   
                            break
                    time.sleep(0.001)


            frameID += 1
            bar.update(1)
            

        self.states.set_mode(Mode.OPTIMIZING)
        while self.states.get_mode() != Mode.TERMINATED:
            time.sleep(0.1)

    def stop(self):
        self.states.set_mode(Mode.TERMINATED)

    def store_relative_pose(self, frame, index_keyframe, Tckc):
        self.frames_info.append([frame.frame_id, frame.frame_time, index_keyframe, Tckc])

    def sav_results(self, path):
        print("Saving the poses and point clouds...")
        save_dir = pathlib.Path(path).joinpath("slam")
        save_dir.mkdir(exist_ok=True, parents=True)

        configFile = save_dir / f"config.json"
        with open(configFile, 'w') as f:
            json.dump(self.config, f, indent=4)

        Twc_est = eval.save_traj(save_dir,  self.keyframes, self.frames_info)
        est_ply_path = os.path.join(save_dir, "frame_poses.ply")
        save_poses_as_pyramid_ply(Twc_est, est_ply_path, size=0.06, color='red')
        if self.args.save_point_could:
            eval.save_keyframe(save_dir, self.keyframes,1.5, device=self.device, use_calib=self.use_calib)
        if len(self.frames_Twc_gt) > 0:
            self.frames_Twc_gt = np.array(self.frames_Twc_gt)
            eval.evaluate_trajectory(save_dir, Twc_est, self.frames_Twc_gt)
            
            eval.save_gt_traj(save_dir, self.frames_Twc_gt)
            
            gt_ply_path = os.path.join(save_dir, "gt_poses.ply")
            save_poses_as_pyramid_ply(self.frames_Twc_gt, gt_ply_path, size=0.06, color='green')
