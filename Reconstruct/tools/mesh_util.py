import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import List
import torch

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    arr = np.sort(cluster_n_triangles.copy())
    k = min(int(cluster_to_keep), len(arr))
    if k == 0:
        
        return mesh_0
    n_cluster = max(int(arr[-k]), 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def refuse_mesh(depths: List[np.ndarray], 
                poses: List[np.ndarray], 
                intrinsics: List[np.ndarray], 
                H: int, 
                W: int, 
                voxel_length: float=0.05, 
                sdf_trunc: float=0.08, 
                depth_trunc: float=100.0
                ):
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose, K, depth_pred in tqdm(zip(poses, intrinsics, depths)):
        depth_pred = depth_pred.reshape(H, W)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        
        depth_pred = o3d.geometry.Image(depth_pred[..., None])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = pose

        volume.integrate(rgbd, intrinsic, extrinsic)   # Open3D expects a world-to-camera extrinsic matrix.
    
    return volume.extract_triangle_mesh()


@torch.no_grad()
def get_depth_fusion_mesh(scene_model, scene_type: str="plane_v1", voxel_length: float=0.05, sdf_trunc: float=0.08, if_using_gaussian: bool=False):
    depths = []
    poses = []
    intrinsics = []            
    for i, keyframe in enumerate(scene_model.keyframes):
        pose = keyframe.get_Rt().detach().clone()  # w2c
        poses.append(pose.cpu().numpy())
        intrinsics.append(scene_model.K[:3, :3].cpu().numpy())
        
        if if_using_gaussian:
            gaussian_render_pkg = scene_model.render_from_id(keyframe.index, pyr_lvl=0, if_render_depth=True)
            # Select the Gaussian package when rendering returns both triangle and Gaussian outputs.
            if isinstance(gaussian_render_pkg, tuple) or isinstance(gaussian_render_pkg, list):
                gaussian_render_pkg = gaussian_render_pkg[1]
            gaussian_depth = gaussian_render_pkg["rendered_depth"]  # all models use surf_depth as depth name
            gaussian_depth = gaussian_depth.squeeze(0).detach().cpu().numpy()  # [H,W]
            depths.append(gaussian_depth)
        else:
            if scene_type == "plane_v1":
                triangle_render_pkg, _= scene_model.render_from_id(keyframe.index, pyr_lvl=0)
                surf_depth = triangle_render_pkg["surf_depth"]  # all models use surf_depth as depth name
                surf_depth = surf_depth.squeeze(0).detach().cpu().numpy()  # [H,W]
                depths.append(surf_depth)
            else:
                raise ValueError(f"Scene type {scene_type} not supported for depth fusion mesh")


    H = depths[0].shape[0]
    W = depths[0].shape[1]    
    
    mesh = refuse_mesh(depths, poses, intrinsics, H, W, voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    mesh = post_process_mesh(mesh, cluster_to_keep=1000)
    return mesh