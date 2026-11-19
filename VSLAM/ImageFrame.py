import dataclasses
from enum import Enum
from typing import Optional
import torch
import pypose as pp
import os
import time
class Mode(Enum):
    TRACKING = 0
    RELOC = 1
    OPTIMIZING = 2
    TERMINATED = 3



@dataclasses.dataclass
class ImageFrame:
    frame_id: int 
    cam_id: int
    frame_time: float
    img: torch.Tensor                                     
    T_WC: pp.Sim3 = pp.identity_Sim3(1)                    

    X_canon: Optional[torch.Tensor] = None                
    C: Optional[torch.Tensor] = None                       
    N: int = 0                                            
    N_updates: int = 0                                    
    K: Optional[torch.Tensor] = None                       

    lidar_frame_id: int = None                             
    lidar_timestamp: float = None                         
    inter_lidar_pose_L0_Lt: torch.Tensor = None            
    
    def __init__(self, frame_id, cam_id, frame_time, img, T_WC, K=None):
        self.frame_id = frame_id
        self.cam_id = cam_id
        self.frame_time = frame_time
        self.img = img
        self.T_WC = T_WC
        self.K = K

    def add_LiDAR_info(self, lidar_frame_id, lidar_timestamp, inter_lidar_pose_L0_Lt):
        if lidar_frame_id is not None:
            self.lidar_frame_id = lidar_frame_id
        if lidar_timestamp is not None:
            self.lidar_timestamp = lidar_timestamp
        if inter_lidar_pose_L0_Lt is not None:
            self.inter_lidar_pose_L0_Lt = inter_lidar_pose_L0_Lt

    
    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor, lidar_processor=None):
        if self.N == 0:
            if lidar_processor is not None: 
                lidar_points = lidar_processor.get_lidar_points(self.lidar_frame_id, self.lidar_timestamp, self.inter_lidar_pose_L0_Lt, accumulate_n=30, return_local=True, device=self.device)

                lidar_point_map = lidar_processor.get_lidar_point_map(self.K, lidar_processor.T_C_L, self.img.shape[1], self.img.shape[2], lidar_points,
                    far_depth_th=200.0,
                    near_depth_th=0.01,
                    image=self.img,
                    depth_cluster_width=-1, 
                    frame_id=self.frame_id
                    )
                depth_lidar = lidar_point_map[:, 2]
                depth_mast3r = X[:, 2]
                mask_lidar_valid = torch.isfinite(depth_lidar) & (depth_lidar > 0.01)

                if mask_lidar_valid.sum() < self.img.shape[1] * self.img.shape[2] - 10:
                    print("Too Many Invalid Points, use MASt3R points to replace")
                    mask_mast3r_valid = torch.isfinite(depth_mast3r) 
                    total_mask = mask_lidar_valid & mask_mast3r_valid 
                    depth_lidar_valid = depth_lidar[total_mask]
                    depth_mast3r_valid = depth_mast3r[total_mask]
                    scale_tensor =  depth_lidar_valid / depth_mast3r_valid 
                    scaled_X_mast3r = X * scale_tensor.mean()  
                    scaled_valid_mast3r_point = scaled_X_mast3r[total_mask]
                    lidar_point_map[~mask_lidar_valid] = scaled_X_mast3r[~mask_lidar_valid]   

                self.X_canon = lidar_point_map.clone()    

                self.C = torch.ones_like(C) * 100.0      
                
                self.N = 1
                self.N_updates = 1
                
                return
            
            else:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.N_updates = 1
                return
            

        if lidar_processor is None: 
            self.X_canon = (((self.C * self.X_canon) + (C * X))
                            / (self.C + C))    
            self.C = self.C + C
            self.N += 1
            self.N_updates += 1
  
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None

    def clear(self):
        del self.img
        del self.X_canon
        del self.C
        del self.N
        del self.N_updates
        del self.K

    def to(self, device):
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                
                if isinstance(field_value, torch.Tensor):
                    setattr(self, field_name, field_value.to(device))
                
                elif hasattr(field_value, 'to') and callable(getattr(field_value, 'to')):
                    setattr(self, field_name, field_value.to(device))
        return self

    @property
    def device(self):
        return self.img.device