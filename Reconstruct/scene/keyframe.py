from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F
from Reconstruct.scene.optimizers import BaseAdam
from Reconstruct.utils import sample, sixD2mtx, point_map_to_normal
from dataloaders.utils_IO import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe in the scene, containing the image, camera parameters, and other information used for optimization.
    """
    def __init__(
        self,
        image: torch.Tensor,
        image_name: str,
        is_test: bool,
        Rt: torch.Tensor,
        mapper_keyframe_idx: int,
        last_keyframe_index: int,
        last_keyframe_frame_id: int,
        is_slam_keyframe: bool,
        f: torch.Tensor,
        args: Namespace,
        T_CkCf: torch.Tensor = None,
        prev_kf: Keyframe = None,
        inference_mode: bool = False,
        Rt_gt: torch.Tensor = None,
        focal_gt: torch.Tensor = None,
        point_map: torch.Tensor = None, # H W 3
        point_conf: torch.Tensor = None, # H W
        frame_time: float = None,     
        device_mapper = "cuda:0",
        prior_exposure: torch.Tensor = None,
    ):
        self.device_mapper = device_mapper
        self.image_pyr = [image]
        self.image_name = image_name
        self.is_test = is_test
        self.width = image.shape[2] if image is not None else None
        self.height = image.shape[1] if image is not None else None
        self.index = mapper_keyframe_idx
        self.last_keyframe_index = last_keyframe_index
        self.last_keyframe_frame_id = last_keyframe_frame_id
        self.is_slam_keyframe = is_slam_keyframe
        self.T_CkCf = T_CkCf  
        self.latest_invdepth = None
        self.latest_depth = None  # for triangle rendering

        self.Rt_gt = Rt_gt
        self.focal_gt = focal_gt
        
        self.frame_time = frame_time  

        if not inference_mode:
            # mono_depth prediction
            self.point_map = point_map.permute(2, 0, 1)[None] # 1 3 H_slam W_slam
            depth_foundation = self.point_map[:, 2:, ...]
            depth = depth_foundation  # for supervising triangle rendering
            idepth = torch.where(depth_foundation != 0, 1.0 / (depth_foundation + 1e-4), 1e4)
            self.mono_depth_conf = point_conf[None, None, ...].to(torch.float32)  # 1 1 H_slam W_slam
            
            # add for triangle rendering
            # using point_map to calculate normal image
            self.normal_map_local  = point_map_to_normal(self.point_map)  # 1 3 H_slam W_slam
            # Create normal pyramid: first interpolate to target size, then build pyramid
            self.normal_local_pyr = [
                F.interpolate(
                    self.normal_map_local,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )[0]
            ]
            
            self.idepth_pyr = [
                F.interpolate(
                    idepth,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )[0]
            ]
            self.idepth_conf_pyr = [
                F.interpolate(
                    self.mono_depth_conf,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )[0]
            ]
            
            # for triangle rendering
            self.depth_pyr = [
                F.interpolate(
                    depth,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )[0]
            ]
            
            # Create the multi-scale inverse depth pyramid
            for _ in range(args.pyr_levels - 1):
                self.idepth_pyr.append(F.avg_pool2d(self.idepth_pyr[-1], 2))
                self.depth_pyr.append(F.avg_pool2d(self.depth_pyr[-1], 2))  # for triangle rendering
                self.idepth_conf_pyr.append(F.avg_pool2d(self.idepth_conf_pyr[-1], 2))
                self.normal_local_pyr.append(F.avg_pool2d(self.normal_local_pyr[-1], 2))  # for triangle rendering

            self.centre = torch.tensor(
                [(self.width - 1) / 2, (self.height - 1) / 2]
            ).to(self.device_mapper)
            self.f = f
            self.depth_loss_weight = args.depth_loss_weight_init
            self.depth_loss_weight_decay = args.depth_loss_weight_decay
            # Build the multiscale pyramids
            for _ in range(args.pyr_levels - 1):
                self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
            self.pyr_lvl = args.pyr_levels - 1

        # Optimizable parameters
        self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone().contiguous())
        self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone().contiguous())
        if prior_exposure is not None:
            self.exposure = torch.nn.Parameter(prior_exposure)
        else:
            exposure = (
                torch.eye(3, 4, device=self.device_mapper)
                if prev_kf is None
                else prev_kf.exposure.clone().detach()
            )
            self.exposure = torch.nn.Parameter(exposure)

        # Optimizer
        if not inference_mode: # Only create optimizer in training mode
            lr_poses = 0 if self.index == 0 else args.lr_poses
            if self.is_test:
                lr_poses = 1e-4
            params = {
                "rW2C": {"val": self.rW2C, "lr": lr_poses},
                "tW2C": {"val": self.tW2C, "lr": lr_poses},
            }
            if not self.is_test:
                params["exposure"] = {"val": self.exposure, "lr": args.lr_exposure}
            self.optimizer = BaseAdam(params, betas=(0.8, 0.99))
            self.num_steps = 0
        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        for i in range(len(self.image_pyr)):
            self.image_pyr[i] = self.image_pyr[i].to(device)
            if self.idepth_pyr is not None:
                self.idepth_pyr[i] = self.idepth_pyr[i].to(device)
            # for triangle rendering
            if self.depth_pyr is not None:
                self.depth_pyr[i] = self.depth_pyr[i].to(device)
            if self.normal_local_pyr is not None:
                self.normal_local_pyr[i] = self.normal_local_pyr[i].to(device)
        if not only_train:
            if self.latest_invdepth is not None:
                self.latest_invdepth = self.latest_invdepth.to(device)
                
            if self.latest_depth is not None:
                self.latest_depth = self.latest_depth.to(device) # for triangle rendering

    @property
    def device(self):
        # Prefer the device of the image pyramid.
        if self.image_pyr and self.image_pyr[0] is not None:
            return self.image_pyr[0].device
        # Otherwise use the camera-center tensor device.
        if hasattr(self, "centre") and isinstance(self.centre, torch.Tensor):
            return self.centre.device
        # Fall back to the configured mapper device.
        try:
            return torch.device(self.device_mapper)
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device=self.device)
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])
        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        """
        Get the centre of the keyframe in the world coordinate
        """
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()
        
    def get_exposure(self):
        return self.exposure


    def get_mono_idepth(self, lvl=0):
        """
            return 1 H W
        """
        if self.idepth_pyr[lvl].device.type != "cuda":
            self.idepth_pyr[lvl] = self.idepth_pyr[lvl].to(self.device_mapper)
        return self.idepth_pyr[lvl]
    
    def get_mono_depth(self, lvl=0):
        """
            return 1 H W
        """
        if self.depth_pyr[lvl].device.type != "cuda":
            self.depth_pyr[lvl] = self.depth_pyr[lvl].to(self.device_mapper)
        return self.depth_pyr[lvl]
    
    def get_mono_normal_local(self, lvl=0):
        """
            return 3 H W
        """
        if self.normal_local_pyr[lvl].device.type != "cuda":
            self.normal_local_pyr[lvl] = self.normal_local_pyr[lvl].to(self.device_mapper)
        return self.normal_local_pyr[lvl]


    @torch.no_grad()
    def sample_conf(self, uv):
        return sample(self.mono_depth_conf, uv.view(1, 1, -1, 2), self.width, self.height)[0, 0, 0]

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self):
        # Optimizer step
        self.optimizer.step()   
        self.depth_loss_weight *= self.depth_loss_weight_decay
        self.num_steps += 1
        # decrement pyr_lvl
        if self.num_steps % 5 == 0:
            if self.pyr_lvl > 0:
                self.image_pyr.pop()
                if self.idepth_pyr is not None:
                    self.idepth_pyr.pop()
                # for triangle rendering
                if self.depth_pyr is not None:
                    self.depth_pyr.pop()
                if self.normal_local_pyr is not None:
                    self.normal_local_pyr.pop()
                self.pyr_lvl -= 1



    @torch.no_grad()
    def transform_T_WC(self, T_gt_est: torch.Tensor, scale_gt_est: float):
        """
        Transform the keyframe's T_WC to the new coordinate system
        """
        T_wc_org = torch.linalg.inv(self.get_Rt().detach().clone())
        T_wc_new = T_wc_org 
        T_wc_new[:3, 3] *= scale_gt_est  
        T_wc_new = T_gt_est @ T_wc_new  
        self.set_Rt(torch.linalg.inv(T_wc_new))


    @torch.no_grad()
    def scale_depth_pyr_and_point_map(self, scale_gt_est: float):
        """
        Scale the keyframe's depth
        """
        self.depth_pyr = [depth * scale_gt_est for depth in self.depth_pyr]
        self.idepth_pyr = [idepth * 1.0 / scale_gt_est for idepth in self.idepth_pyr]
        self.point_map *= scale_gt_est 

    def to_json(self):
        info = {"is_test": self.is_test}
        info["name"] = self.image_name
        if self.Rt_gt is not None:
            info["gt_Rt"] = self.Rt_gt.cpu().numpy().tolist()

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.item(),
            "exposure": self.get_exposure().detach().cpu().numpy().tolist(),
        }

    @classmethod
    def from_json(cls, config, index, height, width, device):
        if "gt_Rt" in config["info"]:
            config["info"]["Rt"] = torch.tensor(config["info"]["gt_Rt"]).cuda()
        if "exposure" in config:
            exposure = torch.tensor(config["exposure"], dtype=torch.float32, device=device)
        else:
            exposure = None
        keyframe = cls(
            image=None,
            image_name=config["info"]["name"],
            is_test=config["info"]["is_test"],
            Rt=torch.tensor(config["Rt"]).cuda(),
            f=None,
            args=None,
            inference_mode=True,
            mapper_keyframe_idx=index,
            last_keyframe_index=None,
            last_keyframe_frame_id=None,
            is_slam_keyframe=False,
            prior_exposure=exposure,
        )
        keyframe.height = height
        keyframe.width = width
        keyframe.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], device=device
        )
        return keyframe

    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        # first param of params is focal length in pixels
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )

        image = BaseImage(
            id=id,
            name=self.image_name,
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
