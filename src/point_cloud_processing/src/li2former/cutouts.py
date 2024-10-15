import torch
import numpy as np
from collections import deque
# from .utils import scan_utils as scan_u
# from .utils import eval_utils as eval_u
# from .models import buildModel
import utils.scan_utils as scan_u
import utils.eval_utils as eval_u
from models import buildModel

class DataScan:
    def __init__(self, ros_config) -> None:
        self.config = ros_config
        self.gpu = ros_config("MODEL")["GPU"]
        self.scan_phi = None
        self.scan_stride = ros_config("SCAN_STRIDE")
        self.laser_fov = ros_config("FOV_DEGREE")
        self.history = deque(maxlen=ros_config("NUM_SCANS"))

    def __call__(self, scan):
        
        if self.scan_phi is None:
            half_fov_rad = 0.5 * np.deg2rad(self.laser_fov)
            self.scan_phi = np.linspace(
                -half_fov_rad, half_fov_rad, len(scan), dtype=np.float32
            )

        cutout_kwargs = {k.lower(): v for k, v in self.config("CUTOUT_KWARGS").items()}
        cutout = scan_u.scans2cutout(
            scan[None, ...],
            self.scan_phi,
            stride=self.config("POINT_STRIDE"),
            **cutout_kwargs
        )
        self.history.append(cutout)
        ct = np.concatenate(self.history, axis=1)
        ct = torch.from_numpy(ct).float()
        return ct 

class attnScan:
    def __init__(self, ros_config) -> None:
        self.config = ros_config
        self.gpu = ros_config("MODEL")["GPU"]
        self.scan_phi = None
        self.scan_stride = ros_config("SCAN_STRIDE")
        self.laser_fov = ros_config("FOV_DEGREE")
        self.history = deque(maxlen=ros_config("NUM_SCANS"))

    def __call__(self, scan):
        
        if self.scan_phi is None:
            half_fov_rad = 0.5 * np.deg2rad(self.laser_fov)
            self.scan_phi = np.linspace(
                -half_fov_rad, half_fov_rad, len(scan), dtype=np.float32
            )

        cutout_kwargs = {k.lower(): v for k, v in self.config("CUTOUT_KWARGS").items()}
        cutout = scan_u.scans2cutout(
            scan[None, ...],
            self.scan_phi,
            stride=self.config("POINT_STRIDE"),
            **cutout_kwargs
        )
        ct = torch.from_numpy(cutout).float()
        return ct 
