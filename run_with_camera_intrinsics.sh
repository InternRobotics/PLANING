#!/bin/bash

# Run ScanNetV2 with known camera intrinsics.
# Usage: bash run_with_camera_intrinsics.sh [num_gs_min] [num_gs_max]

num_gs_min=${1:-4}
num_gs_max=${2:-8}

# Resolve project paths independently of the caller's working directory.
work_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd -- "$work_dir" || exit 1

gpu=0
base_model=plane_v1
dataset_name=selfCaptured

use_all_frames="True"
num_key_iterations=20
num_common_iterations=10
# Requested iterations are rounded up to complete epochs.
save_at_finetune_iteration=(15000)
# A nonempty epoch list overrides the iteration schedule.
save_at_finetune_epoch=()
gs_add_ratio=1.0
triangle_points_lr_init=0.0003

downsampling=1.0
image_sampling=1
lr_poses=0.0001
position_lr_init=0.0005

# Images are read from <base_dir>/<scene>/<img_dir>/.
base_dir="$work_dir/data/scannet_v2"
img_dir="images"
default_scenes=("scene0000_00")
# Results are saved to <result_dir>/<scene>/<timestamp>/.
result_dir="$work_dir/result/with_intrinsics"

echo "====================== PLANING: ScanNetV2 with camera intrinsics ======================"
echo "num_gs_min: ${num_gs_min}"
echo "num_gs_max: ${num_gs_max}"
# Read feature dimensions from the parser defaults without overriding them.
python - -s "$base_dir" -m "$result_dir" <<'PY'
from dataloaders.args import get_args
args = get_args()
print(f"global_geo_feat_dim: {args.global_geo_feat_dim}")
print(f"local_3dgs_feat_dim: {args.local_3dgs_feat_dim}")
PY
echo "result_dir: ${result_dir}"
echo "======================================================================================="

time_tag=$(date "+%Y-%m-%d_%H:%M:%S")
scenes=("${default_scenes[@]}")

for s in "${scenes[@]}"; do
    echo "Running scene=${s}"
    src_dir="${base_dir%/}/${s}"
    calib_file="$src_dir/intrinsic/camera.yaml"
    dst_dir="${result_dir%/}/${s}/${time_tag}"

    if [ ! -d "$src_dir/$img_dir" ]; then
        echo "Image directory not found: $src_dir/$img_dir" >&2
        exit 1
    fi
    if [ ! -f "$calib_file" ]; then
        echo "Camera calibration not found: $calib_file" >&2
        exit 1
    fi

    args=(
        -s "$src_dir"
        -i "$img_dir"
        -m "$dst_dir"
        --config "$work_dir/config/base.yaml"
        --calib "$calib_file"
        -d "$dataset_name"
        --device_frontend cuda:${gpu}
        --device_backend cuda:${gpu}
        --device_mapper cuda:${gpu}
        --device_shared cpu
        --downsampling ${downsampling}
        --test_hold 8
        --lr_poses ${lr_poses}
        --base_model ${base_model}
        --num_key_iterations ${num_key_iterations}
        --num_common_iterations ${num_common_iterations}
        --gs_add_ratio ${gs_add_ratio}
        --image_sampling ${image_sampling}
        --triangle_points_lr_init ${triangle_points_lr_init}
        --position_lr_init ${position_lr_init}
        --num_gs_min ${num_gs_min}
        --num_gs_max ${num_gs_max}
    )

    # Ground-truth poses are optional and used only for evaluation.
    if [ -d "$src_dir/pose_unnormalized" ]; then
        args+=(--gt_pose_dir "$src_dir/pose_unnormalized")
    fi

    if [ "${use_all_frames,,}" = "true" ]; then
        args+=(--use_all_frames)
    fi
    if [ -n "${save_at_finetune_epoch}" ] && [ ${#save_at_finetune_epoch[@]} -gt 0 ]; then
        args+=(--save_at_finetune_epoch "${save_at_finetune_epoch[@]}")
    elif [ -n "${save_at_finetune_iteration}" ] && [ ${#save_at_finetune_iteration[@]} -gt 0 ]; then
        args+=(--save_at_finetune_iteration "${save_at_finetune_iteration[@]}")
    fi

    python "$work_dir/run_system.py" "${args[@]}"
done
