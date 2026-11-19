import argparse
import os

def get_args():
    
    def str2bool(v): return str(v).lower() in ("1","true","t","yes","y","on")
    
    parser = argparse.ArgumentParser(description="Options for data loading and training")

    ## Data and Images
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to the scene directory containing the image folder.")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png or jpeg).")
    parser.add_argument('--downsampling', type=float, default=1.0, help="Downsampling ratio for input images")
    parser.add_argument('--max_size_slam', type=int, default=512,
                        help="Maximum size of slam image process.")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the start of dataset.")
    parser.add_argument('--end_at', type=int, default=100000000,
                        help="Number of frames to skip from the end of dataset.")
    parser.add_argument('--image_sampling', type=int, default=0,
                        help='sample every N images')
    parser.add_argument("-d", "--dataset_name", default="selfCaptured", choices=["selfCaptured"])
    parser.add_argument('--save_to_data_for_gsplat', action="store_true")
    parser.add_argument('--rigid_transform_gaussians', action="store_true")
    parser.add_argument("--base_model", default="plane_v1", choices=["plane_v1"])

    ## Gaussian Learning Rates
    parser.add_argument('--lr_poses', type=float, default=1e-4, help="Pose learning rate")
    parser.add_argument('--lr_exposure', type=float, default=5e-4, 
                        help="Exposure compensation learning rate")
    parser.add_argument('--lr_depth_scale_offset', type=float, default=1e-4)
    parser.add_argument('--position_lr_init', type=float, default=0.0005,    
                        help="Initial position learning rate")
    parser.add_argument('--position_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for position learning rate")
    parser.add_argument('--feat_lr', type=float, default=0.004, help="feat learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.005, help="Feature learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="Opacity learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.01, help="Scaling learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.002, help="Rotation learning rate")
    parser.add_argument('--triangle_points_lr_init', type=float, default=0.0003,  
                        help="Initial triangle points learning rate")
    parser.add_argument('--triangle_points_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for triangle points learning rate") 
    parser.add_argument('--triangle_color_feature_lr', type=float, default=0.005,    
                        help="Initial triangle color feature learning rate")
    parser.add_argument('--triangle_opacity_lr', type=float, default=0.1,    
                        help="Initial triangle opacity learning rate")
    parser.add_argument('--triangle_delta_lr', type=float, default=0.005,    
                        help="Initial triangle delta learning rate")
    parser.add_argument('--triangle_sigma_lr', type=float, default=0.0045,    
                        help="Initial triangle sigma learning rate")
    
    parser.add_argument('--triangle_depth_ratio', type=float, default=0.0,
                        help="Depth ratio for render_depth_median, 0.0 means no render_depth_median, better to be 0.0") 
    parser.add_argument('--triangle_depth_loss_weight', type=float, default=30)  
    parser.add_argument('--triangle_normal_loss_weight', type=float, default=13)  
    parser.add_argument('--triangle_rgb_loss_weight', type=float, default=0)        
    parser.add_argument('--triangle_scaling_loss_weight', type=float, default=0) 
    parser.add_argument('--triangle_entropy_loss_weight', type=float, default=0.1) 
    parser.add_argument('--triangle_opacity_thr', type=float, default=0.2)  
    
    parser.add_argument('--triangle_init_delta', type=float, default=0.1)  
    parser.add_argument('--triangle_init_sigma', type=float, default=0.00095)  
    
    parser.add_argument('--global_geo_feat_dim', type=int, default=16)
    parser.add_argument('--local_3dgs_feat_dim', type=int, default=16)
    
    
    # Motion initialization
    parser.add_argument("--use_uniform_motion", type=str2bool, default=False)    # Optionally initialize sequential-frame poses using constant velocity.
   
    ## Training schedule and losses
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help="Weight for DSSIM loss")
    parser.add_argument('--num_key_iterations', type=int, default=30, 
                        help="Number of training iterations per key frame")
    parser.add_argument('--num_common_iterations', type=int, default=0, 
                        help="Number of training iterations per common frame")
    parser.add_argument('--depth_loss_weight_init', type=float, default=1e-2)
    parser.add_argument('--depth_loss_weight_decay', type=float, default=0.9, 
                        help="Weight decay for depth loss, multiply depth loss weight by this factor every iterations")
    parser.add_argument('--save_at_finetune_epoch', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning.")
    parser.add_argument('--save_at_finetune_iteration', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified iterations when fine-tuning.")
    parser.add_argument('--use_last_frame_proba', type=float, default=0.2, 
                        help="Probability of using the last registered frame for each training iteration")


    parser.add_argument('--scaling_reg_factor', type=float, default=0)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    parser.add_argument('--gs_add_ratio', type=float, default=1.0)
    parser.add_argument('--rad_decay', type=float, default=5 ** 0.5)
    parser.add_argument('--use_all_frames', action="store_true")

    parser.add_argument('--enable_reboot', action='store_true')
    parser.add_argument('--enable_scaling', action='store_true')

    parser.add_argument('--num_gs_min', type=int, default=4)
    parser.add_argument('--num_gs_max', type=int, default=8)
    parser.add_argument('--proba_threshold', type=float, default=0.4)    
    parser.add_argument('--mlp_cov_lr_init', type=float, default=0.004,
                        help="Initial feature learning rate")
    parser.add_argument('--mlp_cov_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for feature learning rate")  
    parser.add_argument('--low_pass_filter_eps', type=float, default=0.01, help="Epsilon for low-pass filter")

    # Incremental pose optimization
    parser.add_argument('--num_prev_keyframes_check', type=int, default=20,
                        help="Number of previous keyframes to check for matches with new keyframe")


    ## Gaussian initialization options
    parser.add_argument('--sh_degree', default=3)
    parser.add_argument('--local_feat_dim', type=int, default=16)
    parser.add_argument('--global_feat_dim', type=int, default=16)
    parser.add_argument('--pyr_levels', type=int, default=2,
                        help="Number of pyramid levels. Each level l will downsample the image 2^l times in width and height")
    parser.add_argument('--init_proba_scaler', type=float, default=2,
                        help="Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")

    # Anchor management
    parser.add_argument('--anchor_overlap', type=float, default=0.3,
                        help="Size of the overlapping regions when blending between anchors")

    ## Keyframe management
    parser.add_argument('--max_active_keyframes', type=int, default=400,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")

    ## Evaluation
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Holdout for test set, will exclude every test_hold image from the Gaussian optimization and use them for testing. The test frames will still be used for training the pose. If set to -1, no keyframes will be excluded from training.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true', 
                        help="Display runtimes for each step in the tqdm bar")

    ## Checkpoint options
    parser.add_argument('-m', '--model_path', default="", 
                        help="Directory to store the renders from test view and checkpoints after training. If not set, will be set to results/xxxxxx.")
    parser.add_argument('--save_every', default=-1, type=int, 
                        help="Frequency of exporting renders w.r.t input frames.")
    parser.add_argument("--save_point_could", action="store_true")

    #################################### system ########################################
    parser.add_argument("--device_frontend", default="cuda:1")
    parser.add_argument("--device_backend", default="cuda:0")
    parser.add_argument("--device_mapper", default="cuda:1")
    parser.add_argument("--device_shared", default="cpu")

    # VSLAM
    parser.add_argument("--covariance_filter", type=str2bool, default=True) 
    parser.add_argument("--accurate_loop_closure", type=str2bool, default=True) 
    parser.add_argument("--use_gt_pose", action="store_true")
    parser.add_argument('--min_displacement', type=float, default=0.03,
                        help="Minimum median keypoint displacement for a new keyframe to be added. Relative to the image width")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--calib", default=None)
    parser.add_argument("--sync_hard", action="store_false")
    parser.add_argument('--thres_keyframe', type=float, default=0.8,
                        help="Size of the overlapping regions when blending between anchors")
    parser.add_argument("--use_same_set_of_keyframes", action="store_true")
    
    # Pose optimization
    parser.add_argument("--use_SE3_to_optimize_pose", type=str2bool, default=False)      # use SE3 instead of Sim3 to optimize pose
    parser.add_argument(
        "--gt_pose_dir", default=None,
        help="Optional directory of image-stem-matched 4x4 camera-to-world TXT poses, for evaluation only.",
    )
    args = parser.parse_args()

    ## Set the output directory if not specified
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
