"""Pose evaluation with exact frame IDs and explicit SE(3)/Sim(3) alignment."""
import copy
import numpy as np
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D


def evaluate_poses(estimated, ground_truth):
    """Evaluate c2w TUM rows without modifying reconstruction poses or scale."""
    if not ground_truth:
        return {"status": "no_ground_truth", "matched_poses": 0}
    est = np.asarray(estimated, dtype=np.float64).reshape(-1, 8)
    gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 8)
    if not np.isfinite(est).all() or not np.isfinite(gt).all():
        raise ValueError("Pose evaluation received nonfinite poses")
    if len(np.unique(est[:, 0])) != len(est) or len(np.unique(gt[:, 0])) != len(gt):
        raise ValueError("Pose evaluation requires unique frame IDs")
    _, ei, gi = np.intersect1d(est[:, 0], gt[:, 0], return_indices=True)
    if len(ei) < 3:
        return {"status": "insufficient_matches", "matched_poses": len(ei)}
    est, gt = est[ei], gt[gi]
    def trajectory(rows):
        return PoseTrajectory3D(positions_xyz=rows[:, 1:4],
                                orientations_quat_wxyz=np.roll(rows[:, 4:8], 1, axis=1),
                                timestamps=rows[:, 0])
    ref, pred = trajectory(gt), trajectory(est)
    result = {"status": "ok", "matched_poses": len(ei), "association": "exact_frame_id",
              "trajectory": "mapper_camera_to_world", "gt_used_for_optimization": False}
    for name, correct_scale in [("se3", False), ("sim3", True)]:
        aligned = copy.deepcopy(pred)
        _, _, scale = aligned.align(ref, correct_scale=correct_scale)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, aligned))
        rotation = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        rotation.process_data((ref, aligned))
        rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=1,
                          delta_unit=metrics.Unit.frames, all_pairs=False)
        rpe.process_data((ref, aligned))
        result[name] = {"ate_rmse_m": float(ape.get_statistic(metrics.StatisticsType.rmse)),
                        "rotation_rmse_deg": float(rotation.get_statistic(metrics.StatisticsType.rmse)),
                        "rpe_1_frame_rmse_m": float(rpe.get_statistic(metrics.StatisticsType.rmse)),
                        "alignment_scale": float(scale)}
    return result
