import time
import torch
import pypose as pp
from tqdm import tqdm
from queue import Empty
from VSLAM.ImageFrame import Mode, ImageFrame
from VSLAM.mast3r_slam.geometry import constrain_points_to_ray
from VSLAM.mast3r_slam.global_opt import FactorGraph
from VSLAM.utils_mast3r import load_retriever, load_mast3r, mast3r_match_asymmetric, mast3r_inference_mono


class Backend():
    def __init__(self, args, config,
                 dataset, H_slam, W_slam, K_slam,
                 states, keyframes,
                 model=None, device="cuda:0",
                 lidar_processor=None):
        self.args = args
        self.config = config
        self.H_slam = H_slam
        self.W_slam = W_slam
        self.K_slam = K_slam.to(device)
        self.device = device

        # shared memory
        self.states = states
        self.keyframes = keyframes

        # class model
        self.dataset = dataset
        if model is None:
            self.model = load_mast3r(device=self.device)
        else:
            self.model = model
        self.retrieval_database = None
        self.factor_graph = FactorGraph(config, self.model, self.keyframes, self.K_slam, self.device)

        self.embeddings = {}
        self.points_for_map = {}
        
        self.lidar_processor = lidar_processor
        if self.lidar_processor is not None:
            self.T_C_L = self.lidar_processor.T_C_L
        else:
            self.T_C_L = None

        self.use_SE3_to_optimize_pose = args.use_SE3_to_optimize_pose  # using SE3 instead of Sim3 to optimize pose


    @torch.no_grad()
    def run(self):
        self.retrieval_database = load_retriever(self.args, self.config, self.model,
                                                 self.keyframes, device=self.device)
        bar = tqdm(total=0, bar_format='{desc}', position=2, leave=False)
        mode = self.states.get_mode()
        while mode is not Mode.TERMINATED:
            # Read Queue
            mode = self.states.get_mode()
            try:
                keyframe_dict = self.states.msgFromFrontend()   
                keyframe_style = keyframe_dict["keyframe_style"]  
                keyframe_id = keyframe_dict["keyframe_id"]
                frame_id = keyframe_dict["frame_id"]
                is_test = keyframe_dict["is_test"]
                is_important = keyframe_dict["is_important"]
            except Empty:
                time.sleep(0.001)
                if mode == Mode.OPTIMIZING:
                    self.states.set_mode(Mode.TERMINATED)
                continue

            # Execute information
            if keyframe_style == 0:
                print("Try to relocalization")
                frame = self.states.get_frame().to(self.device)  
                X_init, C_init, feat, pos = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X_init, C_init, lidar_processor=self.lidar_processor)
                with torch.cuda.device(self.device):
                    success, lc_inds = self.relocalization(frame)
                if success:
                    print("Relocalization success!")
                    self.states.set_mode(Mode.TRACKING)
                    n_kf = len(self.keyframes)
                    keyframe = self.keyframes[n_kf - 1].to(self.device)  
                    last_keyframe = self.keyframes[n_kf - 2].to(self.device)
                    (idx_k2l, valid_match_l,
                     Xkk, Ckk, Qkk,
                     Xlk, Clk, Qlk,
                     _, _) = (mast3r_match_asymmetric(self.config, self.model, keyframe, last_keyframe,
                                                      embeddings_i=self.embeddings[n_kf - 1],
                                                      embeddings_j=self.embeddings[n_kf - 2],))
                    self.prepare_for_mapper(keyframe, last_keyframe, n_kf - 2, idx_k2l,
                                            is_test=is_test,
                                            is_important=is_important,
                                            loop_keyframe_index=lc_inds,
                                            is_slam_keyframe=True)  
                self.states.dequeue_backend_execute()
            elif keyframe_style == 1:  
                lc_inds, idx_k2l, keyframe, last_keyframe = self.global_optimization(keyframe_id)
                last_keyframe_index = keyframe_id # Use the current SLAM keyframe as the mapper reference.
                self.prepare_for_mapper(keyframe, last_keyframe, last_keyframe_index, idx_k2l,
                                        is_test=is_test,
                                        is_important=is_important,
                                        loop_keyframe_index=lc_inds,
                                        is_slam_keyframe=True)

                self.states.dequeue_backend_execute()   
            elif keyframe_style == 2:  
                original_img, info = self.dataset[frame_id]  
                T_WC = pp.Sim3(keyframe_dict["T_WC"]).to(self.device)  
                img = self.dataset.transform.to_slam(original_img, device=self.device)
                timestamp = info["timestamp"]
                keyframe = ImageFrame(frame_id, 0, float(timestamp), img,
                                   T_WC, K=self.K_slam if self.K_slam is not None else None)
                if self.lidar_processor is not None:
                    keyframe.add_LiDAR_info(info["lidar_frame_id"], info["lidar_timestamp"], info["inter_lidar_pose_L0_Lt"])
                    
                last_keyframe = self.keyframes[keyframe_id].to(self.device)
                (idx_k2l, valid_match_l,
                 Xkk, Ckk, Qkk,
                 Xlk, Clk, Qlk,
                 _, _) = (mast3r_match_asymmetric(self.config, self.model, keyframe, last_keyframe,
                                                  embeddings_j=self.embeddings[keyframe_id], ))
                keyframe.update_pointmap(Xkk, Ckk, lidar_processor=self.lidar_processor)   # Update the point map using the new pairwise prediction.
                self.prepare_for_mapper(keyframe, last_keyframe, keyframe_id, idx_k2l, is_test=is_test, is_important=is_important)
                self.states.dequeue_backend_execute()  
        

    @torch.no_grad()
    def prepare_for_mapper(self, keyframe, last_keyframe, last_keyframe_index, idx_k2l,
                           is_test = False, is_important = False, loop_keyframe_index = set(),
                           is_slam_keyframe=False):
            if last_keyframe is None:   # first mapping_keyframe
                X_init, C_init = keyframe.X_canon, keyframe.get_average_conf()
                point = (constrain_points_to_ray(keyframe.img.shape[-2:], X_init[None], self.K_slam).
                         squeeze(0).reshape(self.H_slam, self.W_slam, 3))
                point_valid = (keyframe.get_average_conf() > 1.5).squeeze(-1).reshape(self.H_slam, self.W_slam)  
                T_CkC = None
            else:
                point, point_valid = self.compute_dense_point(keyframe, last_keyframe, idx_k2l)
                T_CkC = last_keyframe.T_WC.Inv().mul(keyframe.T_WC).cpu()      
            densePoint = torch.cat([point, point_valid[..., None]], dim=-1)    # H W 4, the last dimension is the confidence
            keyframe_map_dict = {
                "is_test": is_test,
                "is_important": is_important,
                "T_WC": keyframe.T_WC.data.cpu(),
                "frame_id": keyframe.frame_id,
                "densePoint": densePoint.cpu(),    # Dense camera-frame points and their validity mask.
                "is_slam_keyframe": is_slam_keyframe,
                # For loop closure
                "loop_keyframe_index": loop_keyframe_index,   
                # For map keyframe
                "T_CkC": T_CkC,
                "last_keyframe_index": last_keyframe_index,
                "last_keyframe_frame_id": last_keyframe.frame_id if last_keyframe is not None else None,
                "frame_timestamp": keyframe.frame_time,     
            }
            self.states.msg2Mapper(keyframe_map_dict)

    @torch.no_grad()
    def compute_dense_point(self, keyframe, last_keyframe, idx_k2l, valid_pixel=3):
        assert self.config["use_calib"]==True, "Only work for known calib"
        Xkk = keyframe.X_canon
        idx_k2l = idx_k2l[0]
        H, W = keyframe.img.shape[-2:]

        # get matched points on the last keyframe
        Twk = keyframe.T_WC
        Twl = last_keyframe.T_WC
        Tlk = Twl.Inv().mul(Twk)
        Xkk_calib = constrain_points_to_ray(keyframe.img.shape[-2:], Xkk[None], self.K_slam).squeeze(0) 
        Xkk_calib_match = Xkk_calib[idx_k2l] # Reorder current-frame points to match reference-keyframe pixels.
        Xkl_calib_match = Tlk.Act(Xkk_calib_match).reshape(H, W, 3)  # Transform matched points into the reference-keyframe coordinate system.

        # Compute Residual
        pkl = Xkl_calib_match / Xkl_calib_match[..., 2:]
        pkl_es = (self.K_slam[None, None, ...] @ pkl[..., None]).squeeze(-1)[..., :2]  # H W 2
        uk, vk = torch.meshgrid(torch.arange(W).to(Xkk_calib.device),
                                torch.arange(H).to(Xkk_calib.device),
                                indexing="xy")
        pll = torch.stack([uk, vk], dim=-1)
        resi = torch.norm(pkl_es - pll, dim=-1).reshape(-1)
        conf_valid = torch.where(resi < valid_pixel, 1., 1 / (resi - valid_pixel + 1))     # Convert reprojection residuals to confidence weights.

        Xkw = Twk.Act(Xkk_calib_match)  
        Tkw_se3 = pp.SE3(Twk.data[:, :7]).Inv()
        Xkk_calib_match_map = Tkw_se3.Act(Xkw)  # Camera-frame points at world-coordinate scale, in correspondence order.

        Xkk_map_sorted = Tkw_se3.Act(Twk.Act(Xkk_calib))  # Camera-frame points at world-coordinate scale, in original pixel order.
        conf_valid_sorted = torch.zeros(H * W, device=conf_valid.device, dtype=conf_valid.dtype)  # Unmatched pixels retain zero confidence.
        
        valid_range = (idx_k2l >= 0) & (idx_k2l < H * W)
        
        valid_idx = idx_k2l[valid_range]
        valid_Xkk = Xkk_calib_match_map[valid_range]
        valid_mask = conf_valid[valid_range]
        # Scatter matched points and confidences back to their pixel locations.
        Xkk_map_sorted[valid_idx] = valid_Xkk
        conf_valid_sorted[valid_idx] = valid_mask

        return Xkk_map_sorted.reshape(H, W, 3), valid_mask.reshape(H, W)

    @torch.no_grad()
    def global_optimization(self, idx, n_consec=1):
        keyframe = self.keyframes[idx].to(self.device)   
        last_keyframe = self.keyframes[idx - 1].to(self.device) if idx > 0 else None 
        idx_k2l, Xlk, Clk = None, None, None
        if last_keyframe is not None:
            (idx_k2l, valid_match_l,
             Xkk, Ckk, Qkk,
             Xlk, Clk, Qlk,
             feat_k, pos_k) = (mast3r_match_asymmetric(self.config,
                                              self.model,
                                              keyframe,
                                              last_keyframe,
                                              embeddings_i=None,
                                              embeddings_j=self.embeddings[idx-1]))
        else:
            Xkk, Ckk, feat_k, pos_k = mast3r_inference_mono(self.model, keyframe)
        self.embeddings[idx] = [feat_k, pos_k]
        keyframe.update_pointmap(Xkk, Ckk, lidar_processor=self.lidar_processor)
        self.keyframes[idx] = keyframe    

        kf_idx = [] 
        # k to previous consecutive keyframes
        for j in range(min(n_consec, idx)):  
            kf_idx.append(idx - 1 - j)       
        # Retrieve related keyframes before inserting the current frame into the database.
        retrieval_inds = self.retrieval_database.update(
            self.embeddings[idx][0],   
            add_after_query=True,
            k=self.config["retrieval"]["k"],
            min_thresh=self.config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        lc_inds = set(retrieval_inds)
        lc_inds.add(idx)

        
        kf_idx = set(kf_idx)  
        kf_idx.discard(idx)   
        kf_idx = list(kf_idx)  
        frame_idx = [idx] * len(kf_idx)   
        
        kf_candidate_idxs = None

        if kf_idx:
            added_new_edges, kf_candidate_idxs = self.factor_graph.add_factors(kf_idx, frame_idx, self.config["local_opt"]["min_match_frac"],
                                          self.embeddings)

        with self.states.lock:
            self.states.edges_ii[:] = self.factor_graph.ii.cpu().tolist()   # Share factor-graph edges for visualization.
            self.states.edges_jj[:] = self.factor_graph.jj.cpu().tolist()
            
        with torch.cuda.device(self.device):
            if self.config["use_calib"]:
                if self.use_SE3_to_optimize_pose:
                    self.factor_graph.solve_GN_calib_se3() # using SE3 to optimize pose
                else:
                    self.factor_graph.solve_GN_calib()
            else:
                self.factor_graph.solve_GN_rays()
                
        keyframe = self.keyframes[idx].to(self.device)
        last_keyframe = self.keyframes[idx - 1].to(self.device) if idx > 0 else None
        if last_keyframe is not None:
            T_WCk = pp.Sim3(self.keyframes.T_WC[idx].to(self.device))
            T_WCl = pp.Sim3(self.keyframes.T_WC[idx-1].to(self.device))
            T_ClCk = T_WCl.Inv().mul(T_WCk)
            Xll = T_ClCk.Act(Xlk)     # Transform the new reference-keyframe prediction into its own camera coordinates.
            last_keyframe.update_pointmap(Xll, Clk, lidar_processor=self.lidar_processor)   # Fuse the new pairwise prediction into the previous keyframe point map.
            self.keyframes[idx - 1] = last_keyframe    

        return lc_inds, idx_k2l, keyframe, last_keyframe

    # Retrieve candidate keyframes for relocalization.
    @torch.no_grad()
    def relocalization(self, frame):
        with self.keyframes.lock:
            
            kf_idx = []
            img = frame.img[None]
            img_shape = torch.tensor(img.shape[2:])[None].to(self.device)
            feat, pos, _ = self.model._encode_image(img, img_shape)
            retrieval_inds = self.retrieval_database.update(
                feat,
                add_after_query=False,
                k=self.config["retrieval"]["k"],
                min_thresh=self.config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds
            successful_loop_closure = False
            if kf_idx:
                self.keyframes.append(frame)    
                
                kf_idx = list(kf_idx)  # convert to list
                n_kf = len(self.keyframes)
                self.embeddings[n_kf-1] = [feat, pos]
                frame_idx = [n_kf - 1] * len(kf_idx)
                print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
                
                # Try to add loop factors to reloc this image
                added_new_edges, kf_candidate_idxs = self.factor_graph.add_factors(
                        kf_idx,
                        frame_idx,
                        self.config["reloc"]["min_match_frac"],
                        is_reloc=self.config["reloc"]["strict"],
                        embeddings=self.embeddings,
                        )

                if added_new_edges:
                    self.retrieval_database.update(
                        feat,
                        add_after_query=True,
                        k=self.config["retrieval"]["k"],
                        min_thresh=self.config["retrieval"]["min_thresh"],
                    )
                    successful_loop_closure = True
                    self.keyframes.T_WC[n_kf - 1] = self.keyframes.T_WC[kf_idx[0]].clone()
                else:
                    self.keyframes.pop_last()

            if successful_loop_closure:
                with torch.cuda.device(self.device):
                    if self.config["use_calib"]:
                        if self.use_SE3_to_optimize_pose:
                            self.factor_graph.solve_GN_calib_se3() 
                        else:
                            self.factor_graph.solve_GN_calib()
                    else:
                        self.factor_graph.solve_GN_rays()
                        self.states.set_frame(self.keyframes[-1])
                        
                        
            return successful_loop_closure, set(kf_idx)

