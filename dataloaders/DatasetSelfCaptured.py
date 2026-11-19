import os
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dataloaders.DatasetBasic import BaseDataset
from Reconstruct.utils import get_image_names


class SelfCapturedDataset(BaseDataset):
    """
    The main dataset class for loading images from disk in a multithreaded manner.
    It also supports loading masks and COLMAP poses if available.
    The next image can be fetched using the `getnext` method.
    """

    def __init__(self, args: Namespace):

        # load images paths and subsample image
        self.image_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.image_dir)
        # import pdb; pdb.set_trace()
        self.image_name_list.sort()
        self.timestamp = []
        for i, name in enumerate(self.image_name_list):
            stem = Path(name).stem
            try:
                ts = float(stem)
            except ValueError:
                # Try the numeric part of the filename, then fall back to its sequence index.
                digits = "".join([c for c in stem if c.isdigit()])
                try:
                    ts = float(digits) if digits else float(i)
                except ValueError:
                    ts = float(i)
            self.timestamp.append(ts)
        self.Twc_gt = None
        # Optional ScanNet camera-to-world poses, used only for evaluation.
        pose_dir = getattr(args, "gt_pose_dir", None)
        if pose_dir is not None:
            poses = []
            for name in self.image_name_list:
                pose_path = Path(pose_dir) / (Path(name).stem + ".txt")
                Twc = np.loadtxt(pose_path)
                if Twc.shape != (4, 4) or not np.isfinite(Twc).all():
                    raise ValueError(f"Invalid camera-to-world pose: {pose_path}")
                if not np.allclose(Twc[3], [0, 0, 0, 1], atol=1e-5):
                    raise ValueError(f"Invalid homogeneous pose: {pose_path}")
                quat_xyzw = Rotation.from_matrix(Twc[:3, :3]).as_quat()
                poses.append(np.concatenate((Twc[:3, 3], quat_xyzw)))
            self.Twc_gt = np.asarray(poses)

        super().__init__(args)
