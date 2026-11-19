import torch
import pypose as pp
import torch.multiprocessing as mp

from VSLAM.ImageFrame import Mode, ImageFrame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cpu", using_LiDAR=False):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.TRACKING)
        self.queue_F2B = manager.Queue(maxsize=1000) # org 100, 1000 for server
        self.queue_B2M = manager.Queue(maxsize=1000) # org 100, 1000 for server

        self.backend_execute = manager.Value("i", 0)
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.cam_id = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.timestamp = torch.zeros(1, device=device, dtype=dtype).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.T_WC = pp.identity_Sim3(1,device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1,device=device, dtype=dtype).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()  

        self.using_LiDAR = using_LiDAR
        if self.using_LiDAR:
            self.lidar_frame_id = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
            self.lidar_timestamp = torch.zeros(1, device=device, dtype=torch.float64).share_memory_()
            self.inter_lidar_pose_L0_Lt = torch.zeros(4, 4, device=device, dtype=dtype).share_memory_()

    # set the most recent frame that have been tracked
    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.cam_id[:] = frame.cam_id
            self.timestamp[:] = frame.frame_time
            self.img[:] = frame.img.to(self.device)
            self.T_WC[:] = frame.T_WC.data.to(self.device)
            self.X[:] = frame.X_canon.to(self.device)
            self.C[:] = frame.C.to(self.device)
            self.K[:] = frame.K.to(self.device)   
            if self.using_LiDAR:
                self.lidar_frame_id[:] = frame.lidar_frame_id
                self.lidar_timestamp[:] = frame.lidar_timestamp
                self.inter_lidar_pose_L0_Lt[:] = frame.inter_lidar_pose_L0_Lt.to(self.device)

    # return the most recent frame that have been tracked
    def get_frame(self):
        with self.lock:
            frame = ImageFrame(
                int(self.dataset_idx[0]),
                int(self.cam_id[0]),
                float(self.timestamp[0]),
                self.img,
                pp.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.K = self.K 
            if self.using_LiDAR:
                frame.add_LiDAR_info(self.lidar_frame_id[0], 
                                  self.lidar_timestamp[0], 
                                  self.inter_lidar_pose_L0_Lt)
            return frame


    
    def queue_backend_execute(self):
        with self.lock:
            self.backend_execute.value += 1

    
    def dequeue_backend_execute(self):
        with self.lock:
            if self.backend_execute.value == 0:
                return
            self.backend_execute.value -= 1

    # Send a frame message from the frontend to the backend.
    def msg2Backend(self, msg):
        self.queue_F2B.put(msg)

    # Receive a frame message from the frontend.
    def msgFromFrontend(self):
        return self.queue_F2B.get(block=False)

    # Send a frame message from the backend to the mapper.
    def msg2Mapper(self, msg):
        self.queue_B2M.put(msg)

    # Receive a frame message from the backend.
    def msgFromBackend(self):
        return self.queue_B2M.get(block=False)

    # get system status
    def get_mode(self):
        with self.lock:
            return self.mode.value

    # set system status
    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1