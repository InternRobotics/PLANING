"""
BaseDataset interface for unified data loading (images, poses, pointmaps, etc.)
All dataset types (disk, stream, MASt3R-SLAM, etc.) should implement this interface.
"""
import os
from argparse import Namespace
from pathlib import Path

import cv2
import torch
import logging
import numpy as np
from typing import Tuple, Dict, Any

import yaml

from dataloaders.CameraModel import PinholeCamera
from dataloaders.utils_IO import read_model, qvec2rotmat

class BaseDataset:
    """
        Base dataset interface for unified data loading.
    """
    def __init__(self, args: Namespace):
        """
            Must specific the following variables:

            Required:
                image_dir           [os.Path]            the folder path that contains images
                image_name_list     [list: string]       the list of image full names
                timestamp           [list: float]        the timestamp of images

            Optional:
                Twc_gt              [numpy.array]        poses of each camera: tx ty tz qx qy qz qw
        """

        assert len(self.image_name_list) == len(self.timestamp), \
            "Mismatch between number of times and number of images"
        if self.Twc_gt is not None:
            assert len(self.Twc_gt) == len(self.image_name_list), \
            "Mismatch between number of poses and number of images"

        # dataset downsample
        if hasattr(args, "image_sampling") and args.image_sampling > 1:
            self.image_name_list = self.image_name_list[::args.image_sampling]
            self.timestamp = self.timestamp[::args.image_sampling]
            if self.Twc_gt is not None:
                self.Twc_gt = self.Twc_gt[::args.image_sampling]
        self.start_at = args.start_at
        self.end_at = min(min(len(self.image_name_list), args.end_at), 8000)     

        self.image_name_list = self.image_name_list[self.start_at:self.end_at]
        self.timestamp = self.timestamp[self.start_at:self.end_at]
        if self.Twc_gt is not None:
            self.Twc_gt = self.Twc_gt[self.start_at:self.end_at]
        self.image_paths = [
            os.path.join(self.image_dir, image_name)
            for image_name in self.image_name_list
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # decide test images
        self.infos = {
            name: {
                "is_test": (args.test_hold > 0) and (i % args.test_hold == 0) and (i != 0),
                "name": name,
                "timestamp": self.timestamp[i],
            }
            for i, name in enumerate(self.image_name_list)
        }

        # args: camera intrinsics for slam
        intrinsics = {}
        if args.calib is None:
            H, W, fx, fy, cx, cy = self.estimate_calib()
            intrinsics["width"] = W
            intrinsics["height"] = H
            # 8 parameters: fx, fy, cx, cy, k1, k2, p1, p2.
            # 9 parameters: append k3.
            # 12 parameters: append k3, k4, k5, k6.
            intrinsics["calibration"] = [fx, fy, cx, cy]
        else:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            if hasattr(self, "frame_intrinsic"):
                intrinsics["calibration"] = [self.frame_intrinsic[0, 0, 0],
                                             self.frame_intrinsic[0, 1, 1],
                                             self.frame_intrinsic[0, 0, 2],
                                             self.frame_intrinsic[0, 1, 2]]
        self.downsampling = args.downsampling
        self.load_calib(intrinsics, args.max_size_slam)

        # class variable
        self.current_index = 0


    def estimate_calib(self):
        # GeoCalib is optional when --calib supplies camera intrinsics.
        from geocalib import GeoCalib

        results = []
        H, W = None, None
        model = GeoCalib().cuda()
        for i in range(4):
            image_path = self.image_paths[i]
            image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
            H, W = image.shape[:2]
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
            results.append(model.calibrate(image.cuda())["camera"].K)
        results = torch.concatenate(results, dim=0).mean(dim=0)
        fx, fy, cx, cy = results[0, 0], results[1, 1], results[0, 2], results[1, 2]

        return H, W, fx.item(), fy.item(), cx.item(), cy.item()


    def load_calib(self, intrinsics, max_size_slam=512):

        self.transform = PinholeCamera(max_size_slam,
                                    self.downsampling,
                                    intrinsics["width"],
                                    intrinsics["height"],
                                    intrinsics["calibration"])
        self.H, self.W = intrinsics["height"], intrinsics["width"]
        self.H_slam, self.W_slam = self.transform.H_slam, self.transform.W_slam
        self.H_map, self.W_map = self.transform.H_map, self.transform.W_map
        self.K_slam = self.transform.K_slam
        self.K_map = self.transform.K_map


    def __len__(self) -> int:
        """Returns the number of images/frames (or a large number for streams)."""
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        H_c, W_c = image.shape[:2]
        assert (H_c == self.H) and (W_c == self.W), \
            f"Image {index}: Size not match between the actual and the size specified in the file"
        info = self.infos[os.path.basename(image_path)]
        if self.Twc_gt is not None:
            info["Twc_gt"] = self.Twc_gt[index]
        return image, info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        return image

    def getnext(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns the next image and associated info (e.g., pose, pointmap, etc.).
        Returns:
            image (torch.Tensor): The image tensor (C, H, W)
            info (dict): Dictionary with keys like 'pose', 'pointmap', 'is_test', etc.
        """
        frame, info = self.__getitem__(self.current_index)
        self.current_index += 1
        return frame, info

    def get_image_size(self) -> Tuple[int, int]:
        """Returns (height, width) of images."""
        return self.H_map, self.W_map


