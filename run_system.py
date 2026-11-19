import os
# Set LD_LIBRARY_PATH early for curope module before any imports
import torch
torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = torch_lib_path + ':' + os.environ['LD_LIBRARY_PATH']
else:
    os.environ['LD_LIBRARY_PATH'] = torch_lib_path

from queue import Empty
import shutil
import pypose as pp
from Reconstruct.scene.keyframe import Keyframe
from VSLAM.Backend import Backend
from VSLAM.ImageFrame import Mode
from VSLAM.SharedKeyframes import SharedKeyframes
from VSLAM.SharedStates import SharedStates
from dataloaders.utils_load import load_dataset
from VSLAM.utils_config import load_config
from VSLAM.Frontend import Frontend


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import time

import numpy as np
from tqdm import tqdm
from dataloaders.args import get_args
import torch.multiprocessing as mp
import os

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    args = get_args()
    if os.path.exists(args.model_path) and os.listdir(args.model_path):
        raise FileExistsError(f"Result directory must be empty: {args.model_path}")
    config = load_config(args.config)
    device_frontend = args.device_frontend
    device_backend = args.device_backend
    device_mapper = args.device_mapper
    device_shared = args.device_shared

    dataset, is_stream = load_dataset(args)
    H_map, W_map = dataset.H_map, dataset.W_map
    H_slam, W_slam = dataset.H_slam, dataset.W_slam
    K_slam, K_map = dataset.K_slam, dataset.K_map
    lidar_processor = None
    using_LiDAR = False

    manager = mp.Manager()
    insufficient_match_cnt = manager.Value('i', 0)
    cholesky_failed_cnt = manager.Value('i', 0)
    # keyframes = SharedKeyframes(config, manager, H_slam, W_slam, K_slam, device=device_shared)
    keyframes = SharedKeyframes(config, manager, H_slam, W_slam, K_slam, device=device_shared, buffer=1000, using_LiDAR=using_LiDAR)  # limit the number of keyframes for 32G cpu, 1000 for server
    states = SharedStates(manager, H_slam, W_slam, device=device_shared, using_LiDAR=using_LiDAR) 

    frontend = Frontend(args, config, dataset, keyframes, states, device=device_frontend, lidar_processor=lidar_processor, insufficient_match_cnt=insufficient_match_cnt, 
                   cholesky_failed_cnt=cholesky_failed_cnt)
    backend = Backend(args, config, dataset,
                      H_slam, W_slam, K_slam,
                      states, keyframes,
                      model=frontend.model if device_frontend == device_backend else None,
                      device=device_backend,
                      lidar_processor=lidar_processor)
    backend_thread = mp.Process(target=backend.run)
    backend_thread.start()
    print(f"Using dataset: {args.source_path}")
    frontend_thread = mp.Process(target=frontend.run)
    frontend_thread.start()

    from Reconstruct.scene.scene_models.plane_v1 import SceneModel
    scene_model = SceneModel(W_map, H_map, K_map.to(device_backend), args, device=device_mapper)
    print("Initialize reconstruction model ok!, using base model: ", args.base_model)
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    runtimes = {key: [0, 0] for key in runtimes}
    metrics = {}
    pbar = tqdm(total=0, desc=f"GS Mapper: ", position=1, leave=False)
    reconstruction_start_time = time.time()
    mapper_index = 0
    related_frames = {}

    while True:
        mode = states.get_mode()
        try:
            keyframe_map_dict = states.msgFromBackend()
        except Empty:
            time.sleep(0.001)
            if mode == Mode.TERMINATED:
                break
            continue

        frame_id = keyframe_map_dict["frame_id"]
        frame_time = keyframe_map_dict["frame_timestamp"]  
        last_keyframe_index = keyframe_map_dict["last_keyframe_index"]
        last_keyframe_frame_id = keyframe_map_dict.get("last_keyframe_frame_id", None)  # 
        if last_keyframe_index not in related_frames:
            related_frames[last_keyframe_index] = []
        related_frames[last_keyframe_index].append(mapper_index)
        
        T_CkC = keyframe_map_dict["T_CkC"]
        T_CkC = T_CkC.to(device_mapper) if T_CkC is not None else None
        original_img, info = dataset[frame_id]   
        T_WC = pp.Sim3(keyframe_map_dict["T_WC"]).to(device_mapper)
        densePoint = keyframe_map_dict["densePoint"].to(device_mapper)
        is_important = keyframe_map_dict["is_important"]
        is_test = keyframe_map_dict["is_test"]
        point_map = densePoint[...,:3].to(device_mapper)
        point_map_conf = densePoint[...,3].to(device_mapper)
        Twc = pp.SE3(T_WC.data[:, :7])
        Twc = pp.quat2unit(Twc)
        Tcw = Twc.Inv().matrix()[0]

        image_for_mapper = dataset.transform.to_map(original_img, device=device_mapper)

        keyframe_map = Keyframe(image_for_mapper,
                                info["name"],
                                is_test,
                                Tcw,
                                mapper_index,
                                last_keyframe_index,
                                last_keyframe_frame_id,
                                keyframe_map_dict["is_slam_keyframe"],
                                torch.tensor([K_map[0, 0].item()]).to(device_mapper),
                                args,
                                T_CkCf=T_CkC,
                                point_map=point_map.to(device_mapper),
                                point_conf=point_map_conf.to(device_mapper),
                                Rt_gt=(pp.SE3(torch.as_tensor(info["Twc_gt"], dtype=torch.float32,
                                                              device=device_mapper)).Inv().matrix()
                                       if "Twc_gt" in info else None),
                                frame_time=frame_time,     
                                device_mapper=device_mapper
                                )
        
        if keyframe_map_dict["is_slam_keyframe"] and frame_id > 0:
            old_c2ws = torch.zeros(len(scene_model.keyframes), 4, 4).to(device_mapper)
            new_c2ws = torch.zeros(len(scene_model.keyframes), 4, 4).to(device_mapper)
            for index in related_frames.keys():
                for mapper_frame_id in related_frames.get(index, []):
                    if mapper_frame_id == len(scene_model.keyframes):
                        continue
                    frame = scene_model.keyframes[mapper_frame_id]
                    
                    if frame.is_slam_keyframe:
                        frame_slam = keyframes[frame.last_keyframe_index]
                        T_WCk = pp.SE3(frame_slam.T_WC.data[:, :7])
                        T_WCk = pp.quat2unit(T_WCk).to(device_mapper)
                        T_WCf = T_WCk
                    else:
                        frame_slam = keyframes[frame.last_keyframe_index]
                        T_WCk = frame_slam.T_WC.to(device_mapper)
                        T_WCf = T_WCk.mul(frame.T_CkCf)
                        T_WCf = pp.SE3(pp.quat2unit(T_WCf).data[:, :7])
                    new_Rt = T_WCf.Inv().matrix()[0]
                    old_Rt = frame.get_Rt()
                    
                    frame.set_Rt(new_Rt.to(old_Rt.device))
                    view_matrix = frame.get_Rt().transpose(0, 1).to(device_mapper)
                    centre = view_matrix.detach().inverse()[3, :3].to(device_mapper)
                    old_c2ws[mapper_frame_id] = torch.linalg.inv(old_Rt).to(device_mapper)
                    new_c2ws[mapper_frame_id] = torch.linalg.inv(new_Rt).to(device_mapper)
            
            old_c2ws = old_c2ws.squeeze(1)
            new_c2ws = new_c2ws.squeeze(1)
            scene_model.rigid_transform(old_c2ws, new_c2ws)


        with torch.cuda.device(device_mapper):
            scene_model.add_keyframe(keyframe_map)
            if is_important:
                scene_model.add_new_gaussians()
            num_iterations = args.num_key_iterations if is_important else args.num_common_iterations   
            if is_stream:
                scene_model.optimize_async(num_iterations, is_important)
            else:
                scene_model.optimization_loop(num_iterations, is_important, debug_save_path=args.model_path)

            if (
                    len(scene_model.keyframes) % args.test_frequency == 0
                    and args.test_frequency > 0
                    and (args.test_hold > 0)
            ):
                metrics = scene_model.evaluate()

            if (
                    mapper_index % args.save_every == 0
                    and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{mapper_index:05d}")
                )

            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            if args.display_runtimes:
                for key, value in runtimes.items():
                    if value[1] > 0:
                        bar_postfix += [
                            f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                        ]
            bar_postfix += [
                f"\033[36mKeyframes:{len(scene_model.keyframes)}\033[0m",
                f"\033[36m{'Triangles' if hasattr(scene_model, 'number_of_triangles') else 'Gaussians'} :{scene_model.number_of_triangles if hasattr(scene_model, 'number_of_triangles') else scene_model.number_of_gaussians}\033[0m",
            ]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)
        mapper_index += 1   
        pbar.n = frame_id
        pbar.refresh()

    reconstruction_time = time.time() - reconstruction_start_time
    scene_model.enable_inference_mode()

    with torch.cuda.device(device_mapper):
        print("Saving the reconstruction to:", args.model_path)
        metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset), insufficient_match_cnt=insufficient_match_cnt.value, cholesky_failed_cnt=cholesky_failed_cnt.value)
        print("Start finetuning!")
        if len(args.save_at_finetune_epoch) > 0 or len(args.save_at_finetune_iteration) > 0:           
            if len(args.save_at_finetune_epoch) > 0:
                finetune_epochs = max(args.save_at_finetune_epoch)
            else:
                iterations_per_epoch = len(scene_model.keyframes)
                save_epochs = []
                for iteration in args.save_at_finetune_iteration:
                    epoch = (iteration + iterations_per_epoch - 1) // iterations_per_epoch  
                    if epoch > 0:  
                        save_epochs.append(epoch)
                
                save_epochs = sorted(list(set(save_epochs)))
                args.save_at_finetune_epoch = save_epochs
                finetune_epochs = max(save_epochs) if save_epochs else 1
            torch.cuda.empty_cache()
            scene_model.inference_mode = False
            pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
            for epoch in pbar:
                epoch_start_time = time.time()
                scene_model.finetune_epoch()
                epoch_time = time.time() - epoch_start_time
                reconstruction_time += epoch_time
                if epoch + 1 in args.save_at_finetune_epoch:
                    torch.cuda.empty_cache()
                    scene_model.inference_mode = True
                    scene_model.export_finetune_epoch = epoch + 1
                    scene_model.export_finetune_iterations = (epoch + 1) * len(scene_model.keyframes)
                    metrics = scene_model.save(
                        os.path.join(args.model_path, str(epoch + 1)), reconstruction_time,
                        insufficient_match_cnt=insufficient_match_cnt.value,
                        cholesky_failed_cnt=cholesky_failed_cnt.value, from_finetune=True
                    )
                    bar_postfix = []
                    for key, value in metrics.items():
                        bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                    pbar.set_postfix_str(",".join(bar_postfix))
                    scene_model.inference_mode = False
                    torch.cuda.empty_cache()
            scene_model.inference_mode = True

        if args.save_to_data_for_gsplat:
            colmap_dir = os.path.join(args.model_path, "colmap")
            target_dir = os.path.join(args.source_path, "artdeco_colmap")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            print(colmap_dir, target_dir)
            shutil.copytree(colmap_dir, target_dir)
        print(
            ", ".join(
                f"{metric}: {value:.3f}"
                if isinstance(value, float)
                else f"{metric}: {value}"
                for metric, value in metrics.items()
            )
        )
    frontend_thread.join()
    backend_thread.join()
