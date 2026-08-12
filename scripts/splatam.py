import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import eval, report_loss, report_progress
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import build_rotation, calc_ssim, densify, prune_gaussians

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


_native_lpips_model = None
_NATIVE_QUALITY_PROTOCOL = "valid_depth_rgb_v2"


def _capture_session():
    if os.environ.get("THREEDGS_SLAM_NATIVE_CAPTURE") in (None, "", "0"):
        return None
    try:
        from simulator.instrumentation import capture_session
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "native capture is enabled but simulator is unavailable; add the 3DGS-SLAM repository to PYTHONPATH"
        ) from error
    return capture_session("SplaTAM")


def _capture_context(params, variables, frame, iteration, stage, source, source_frame=None):
    capture = variables.get("_native_capture")
    if capture is None:
        return None
    stable_ids = variables["_native_gaussian_ids"].require_size(int(params["means3D"].shape[0]))
    return capture.make_rasterizer_context(
        stable_ids,
        frame=frame,
        iteration=iteration,
        stage=stage,
        source=source,
        source_frame=source_frame,
    )


def _rasterize_with_capture(rasterizer, render_args, params, variables, frame, iteration, stage, source, source_frame=None,
                            evidence=None):
    capture = variables.get("_native_capture")
    if capture is None:
        return rasterizer(**render_args)
    from diff_gaussian_rasterization import set_binning_capture_callback

    context = _capture_context(params, variables, frame, iteration, stage, source, source_frame)
    evidence_destination = evidence
    with capture.capture_rasterizer(set_binning_capture_callback, context) as captured_evidence:
        rendered = rasterizer(**render_args)
        _record_native_impact_tile(variables, context, stage, captured_evidence, rasterizer)
    if evidence_destination is not None:
        evidence_destination["rasterizer_sequence"] = captured_evidence.sequence
    return rendered


def _record_native_impact_tile(variables, context, stage, evidence, rasterizer):
    """Keep the final frame-zero Mapping tile used by an optional withdrawal replay."""
    if not variables.get("_native_impact_replay_enabled"):
        return
    if context.frame != 0 or stage != "mapping_render_rgb":
        return
    if context.iteration != variables.get("_native_impact_capture_iteration"):
        return
    if evidence.sequence is None or evidence.payload is None:
        raise RuntimeError("native Mapping rasterizer did not emit a tile-list observation")

    tile_ranges = evidence.payload["tile_ranges"].detach().cpu().tolist()
    point_list = evidence.payload["point_list"].detach().cpu().tolist()
    for tile_index, (start, end) in enumerate(tile_ranges):
        if end <= start:
            continue
        point_indices = point_list[start:end]
        stable_ids = tuple(context.stable_gaussian_ids[index] for index in point_indices)
        if not stable_ids or len(set(stable_ids)) != len(stable_ids):
            raise RuntimeError("native Mapping tile does not contain a unique stable Gaussian update group")
        variables["_native_impact_group"] = {
            "frame": context.frame,
            "iteration": context.iteration,
            "source_frame": context.source_frame,
            "source": context.source,
            "rasterizer_sequence": evidence.sequence,
            "tile_index": tile_index,
            "tile_range": [start, end],
            "stable_gaussian_ids": stable_ids,
        }
        return


def _record_loss_mask(variables, frame, iteration, stage, mask, source_frame=None):
    capture = variables.get("_native_capture")
    if capture is not None:
        return capture.record_native_event(
            "loss_mask",
            "SplaTAM.get_loss",
            {
                "frame": frame,
                "iteration": iteration,
                "stage": stage,
                "valid_pixel_mask": mask,
                **({"source_frame": source_frame} if source_frame is not None else {}),
            },
        )
    return None


def _capture_replace_ids(params, variables, operation, source):
    capture = variables.get("_native_capture")
    if capture is None:
        return
    stable_ids = variables["_native_gaussian_ids"]
    before_count = len(stable_ids.ids)
    active = stable_ids.replace(int(params["means3D"].shape[0]))
    capture.record_native_event(
        "structure",
        source,
        {
            "operation": operation,
            "before_count": before_count,
            "after_count": int(params["means3D"].shape[0]),
            "stable_gaussian_ids": active,
        },
    )


def _record_optimizer_step(variables, params, optimizer, frame, iteration, stage, phase, parameter_names=None):
    capture = variables.get("_native_capture")
    if capture is not None:
        observed_parameters = params
        if parameter_names is not None:
            observed_parameters = {name: params[name] for name in parameter_names}
        capture.record_native_event(
            "optimizer",
            "SplaTAM.rgbd_slam",
            capture.optimizer_step_payload(
                observed_parameters,
                optimizer,
                frame=frame,
                iteration=iteration,
                stage=stage,
                phase=phase,
            ),
        )


def _record_pose(variables, frame, iteration, phase, rotation, translation, losses, extended_tracking,
                 source_frame, last_optimizer_iteration=None, termination=None, best_loss=None,
                 coverage_stage="tracking_pose_coverage", optimizer_stage="tracking"):
    capture = variables.get("_native_capture")
    if capture is not None:
        if phase == "ground_truth_assigned":
            payload = {
                "frame": frame,
                "iteration": iteration,
                "origin": "ground_truth_assignment",
                "parameterization": "quaternion_translation_7d",
                "ground_truth_rotation": rotation,
                "ground_truth_translation": translation,
            }
        else:
            normalized_rotation = F.normalize(rotation)
            payload = {
                "frame": frame,
                "iteration": iteration,
                "origin": "native_optimizer",
                "phase": phase,
                "parameterization": "quaternion_translation_7d",
                "parameters": {
                    "unnormalized_quaternion": rotation,
                    "translation": translation,
                },
                "normalization": {
                    "operation": "torch.nn.functional.normalize",
                    "quaternion": normalized_rotation,
                },
                "committed_pose": {
                    "quaternion": normalized_rotation,
                    "translation": translation,
                },
                "optimizer_observation": {
                    "source": "SplaTAM.rgbd_slam",
                    "frame": frame,
                    "iteration": last_optimizer_iteration,
                    "stage": optimizer_stage,
                },
                "coverage": {
                    "loss_mask": {
                        "frame": frame,
                        "iteration": last_optimizer_iteration,
                        "stage": coverage_stage,
                        "source_frame": source_frame,
                    },
                },
                "convergence": {
                    "losses": losses,
                    "best_loss": best_loss,
                    "extended_tracking": extended_tracking,
                    "completed_iterations": iteration,
                    "termination": termination,
                },
            }
        return capture.record_native_event(
            "pose",
            "SplaTAM.rgbd_slam",
            payload,
        )
    return None


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    capture = _capture_session()
    if capture is not None:
        stable_ids = capture.gaussian_ids("SplaTAM")
        active = stable_ids.initialize(int(num_pts))
        variables["_native_capture"] = capture
        variables["_native_gaussian_ids"] = stable_ids
        capture.record_native_event(
            "structure",
            "SplaTAM.initialize_params",
            {
                "operation": "initialize_params",
                "before_count": 0,
                "after_count": int(num_pts),
                "stable_gaussian_ids": active,
            },
        )

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None,
             capture_iteration=0, capture_frame=None, capture_tracking_observation=False,
             capture_stage_override=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    capture_stage = capture_stage_override or (
        "tracking_render" if tracking else "mapping_render" if mapping else "initialization_render"
    )
    capture_iteration = tracking_iteration if tracking_iteration is not None else capture_iteration
    capture_frame = iter_time_idx if capture_frame is None else capture_frame
    rgb_evidence = {} if capture_tracking_observation else None
    im, radius, _, = _rasterize_with_capture(
        Renderer(raster_settings=curr_data['cam']),
        rendervar,
        params,
        variables,
        capture_frame,
        capture_iteration,
        f"{capture_stage}_rgb",
        "SplaTAM.get_loss.rgb",
        source_frame=iter_time_idx,
        evidence=rgb_evidence,
    )
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = _rasterize_with_capture(
        Renderer(raster_settings=curr_data['cam']),
        depth_sil_rendervar,
        params,
        variables,
        capture_frame,
        capture_iteration,
        f"{capture_stage}_depth_silhouette",
        "SplaTAM.get_loss.depth_silhouette",
        source_frame=iter_time_idx,
    )
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
    else:
        color_mask = torch.ones_like(curr_data['im'], dtype=torch.bool)
    rgb_loss_mask_sequence = _record_loss_mask(
        variables,
        capture_frame,
        capture_iteration,
        f"{capture_stage}_rgb",
        {"color": color_mask},
        source_frame=iter_time_idx,
    )
    depth_loss_mask_sequence = _record_loss_mask(
        variables,
        capture_frame,
        capture_iteration,
        f"{capture_stage}_depth_silhouette",
        {"depth": mask},
        source_frame=iter_time_idx,
    )
    if tracking:
        # Preserve the exact native depth-validity and silhouette mask that
        # constrains the tracking solve; it is not reconstructed downstream.
        pose_coverage_stage = "tracking_pose_coverage" if capture_stage == "tracking_render" else f"{capture_stage}_pose_coverage"
        _record_loss_mask(
            variables,
            capture_frame,
            capture_iteration,
            pose_coverage_stage,
            {"coverage": mask},
            source_frame=iter_time_idx,
        )

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    if capture_tracking_observation:
        if not tracking:
            raise RuntimeError("capture_tracking_observation requires SplaTAM Tracking loss")
        capture = variables.get("_native_capture")
        if capture is not None and (
            rgb_evidence is None
            or rgb_evidence.get("rasterizer_sequence") is None
            or rgb_loss_mask_sequence is None
            or depth_loss_mask_sequence is None
        ):
            raise RuntimeError("streaming block capture requires direct RGB rasterizer and RGB/depth loss-mask observations")
        return loss, variables, weighted_losses, {
            "im": im,
            "depth": depth,
            "depth_mask": mask,
            "color_mask": color_mask,
            "capture_frame": capture_frame,
            "capture_iteration": capture_iteration,
            "capture_stage": capture_stage,
            "rgb_rasterizer_sequence": rgb_evidence.get("rasterizer_sequence") if rgb_evidence is not None else None,
            "rgb_loss_mask_sequence": rgb_loss_mask_sequence,
            "depth_loss_mask_sequence": depth_loss_mask_sequence,
        }

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = _rasterize_with_capture(
        Renderer(raster_settings=curr_data['cam']),
        depth_sil_rendervar,
        params,
        variables,
        time_idx,
        0,
        "mapping_growth_silhouette",
        "SplaTAM.add_new_gaussians",
    )
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)
        capture = variables.get("_native_capture")
        if capture is not None:
            stable_ids = variables["_native_gaussian_ids"]
            before_count = len(stable_ids.ids)
            added = stable_ids.append(int(new_pt_cld.shape[0]))
            capture.record_native_event(
                "structure",
                "SplaTAM.add_new_gaussians",
                {
                    "operation": "add_new_gaussians",
                    "before_count": before_count,
                    "after_count": int(params["means3D"].shape[0]),
                    "added_stable_ids": added,
                },
            )

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


_GAUSSIAN_PARAMETER_NAMES = (
    "means3D",
    "rgb_colors",
    "unnorm_rotations",
    "logit_opacities",
    "log_scales",
)


def _snapshot_tensor_values(values):
    return {name: value.detach().clone() for name, value in values.items() if isinstance(value, torch.Tensor)}


def _restore_tensor_values(values, snapshot):
    with torch.no_grad():
        for name, value in snapshot.items():
            if name not in values or not isinstance(values[name], torch.Tensor):
                raise RuntimeError(f"cannot restore native replay value {name!r}")
            if values[name].shape != value.shape:
                raise RuntimeError(f"native replay shape changed for {name!r}")
            values[name].copy_(value)


def _snapshot_impact_baseline(params, variables):
    stable_ids = tuple(variables["_native_gaussian_ids"].ids)
    variables["_native_impact_baseline"] = {
        "stable_gaussian_ids": stable_ids,
        "parameters": {name: params[name].detach().clone() for name in _GAUSSIAN_PARAMETER_NAMES},
    }


def _withdraw_native_mapping_group(params, variables, group):
    baseline = variables.get("_native_impact_baseline")
    if baseline is None:
        raise RuntimeError("native impact replay has no frame-zero Mapping baseline")
    baseline_indices = {identifier: index for index, identifier in enumerate(baseline["stable_gaussian_ids"])}
    current_indices = {identifier: index for index, identifier in enumerate(variables["_native_gaussian_ids"].ids)}
    stable_ids = tuple(group["stable_gaussian_ids"])
    if not stable_ids or any(identifier not in baseline_indices or identifier not in current_indices for identifier in stable_ids):
        raise RuntimeError("native impact tile is no longer present after Mapping")

    current = [current_indices[identifier] for identifier in stable_ids]
    original = [baseline_indices[identifier] for identifier in stable_ids]
    with torch.no_grad():
        for name in _GAUSSIAN_PARAMETER_NAMES:
            current_index = torch.tensor(current, device=params[name].device)
            original_index = torch.tensor(original, device=params[name].device)
            params[name].index_copy_(0, current_index, baseline["parameters"][name].index_select(0, original_index))


def _tracking_result_summary(result):
    return {
        "completed_iterations": result["completed_iterations"],
        "termination": result["termination"],
        "extended_tracking": result["extended_tracking"],
        "best_loss": result["best_loss"],
        "losses": result["losses"],
        "native_optimizer_parameters": {
            "unnormalized_quaternion": result["rotation"].detach().cpu().tolist(),
            "translation": result["translation"].detach().cpu().tolist(),
        },
        "committed_pose": {
            "quaternion": F.normalize(result["rotation"]).detach().cpu().tolist(),
            "translation": result["translation"].detach().cpu().tolist(),
        },
    }


def _tracking_result_is_finite(result):
    return bool(
        torch.isfinite(result["rotation"]).all()
        and torch.isfinite(result["translation"]).all()
        and all(np.isfinite(value) for value in result["losses"].values())
        and np.isfinite(result["best_loss"])
    )


def _load_frozen_streaming_capture_contract():
    """Read default or explicitly selected frozen capture geometry."""
    try:
        from simulator.config import ArchitectureConfig
    except ModuleNotFoundError as error:
        raise RuntimeError("streaming accuracy capture requires the repository simulator on PYTHONPATH") from error

    path = os.environ.get("THREEDGS_SLAM_NATIVE_CAPTURE_ARCHITECTURE")
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(_BASE_DIR)), "simulator", "config", "default_architecture.json")
    architecture = ArchitectureConfig.load(path)
    return {
        "block_edge_pixels": architecture.block_edge_pixels,
        "active_block_slots": architecture.architecture["active_block_slots"],
        "minimum_inflight_blocks": architecture.minimum_inflight_blocks,
    }


def _snapshot_tracking_state(params, variables, time_idx):
    return {
        "rotation": params["cam_unnorm_rots"][..., time_idx].detach().clone(),
        "translation": params["cam_trans"][..., time_idx].detach().clone(),
        "max_2D_radius": variables["max_2D_radius"].detach().clone(),
        "seen": variables.get("seen").detach().clone() if isinstance(variables.get("seen"), torch.Tensor) else None,
    }


def _restore_tracking_state(params, variables, time_idx, snapshot):
    with torch.no_grad():
        params["cam_unnorm_rots"][..., time_idx].copy_(snapshot["rotation"])
        params["cam_trans"][..., time_idx].copy_(snapshot["translation"])
        variables["max_2D_radius"].copy_(snapshot["max_2D_radius"])
        if snapshot["seen"] is None:
            variables.pop("seen", None)
        else:
            variables["seen"] = snapshot["seen"].detach().clone()


def _native_successor_tracking_data(dataset, target_frame, tracking_curr_data):
    """Load the immediate next RGB-D input without changing the strict trajectory state."""
    successor_frame = target_frame + 1
    if successor_frame >= len(dataset):
        raise RuntimeError("native streaming successor lies outside the configured Tracking dataset")
    color, depth, _, ground_truth_pose = dataset[successor_frame]
    color = color.permute(2, 0, 1) / 255
    depth = depth.permute(2, 0, 1)
    return {
        "cam": tracking_curr_data["cam"],
        "im": color,
        "depth": depth,
        "id": successor_frame,
        "intrinsics": tracking_curr_data["intrinsics"],
        "w2c": tracking_curr_data["w2c"],
        "iter_gt_w2c_list": list(tracking_curr_data["iter_gt_w2c_list"]) + [torch.linalg.inv(ground_truth_pose)],
    }


def _native_tracking_block_loss(curr_data, observation, loss_weights, x, y, width, height):
    rows = slice(y, y + height)
    columns = slice(x, x + width)
    depth_error = torch.abs(curr_data["depth"][:, rows, columns] - observation["depth"][:, rows, columns])
    image_error = torch.abs(curr_data["im"][:, rows, columns] - observation["im"][:, rows, columns])
    depth_loss = depth_error[observation["depth_mask"][:, rows, columns]].sum()
    image_loss = image_error[observation["color_mask"][:, rows, columns]].sum()
    return depth_loss * loss_weights["depth"] + image_loss * loss_weights["im"]


def _capture_native_tracking_blocks(params, variables, config, time_idx, iter_time_idx, tracking_curr_data, capture_stage,
                                    capture_iteration, contract):
    """Capture ordered native block-loss gradients with bounded host synchronization."""
    if not config["tracking"]["use_l1"]:
        raise RuntimeError("streaming accuracy capture requires SplaTAM's native L1 Tracking loss")
    loss, variables, _, observation = get_loss(
        params,
        tracking_curr_data,
        variables,
        iter_time_idx,
        config["tracking"]["loss_weights"],
        config["tracking"]["use_sil_for_loss"],
        config["tracking"]["sil_thres"],
        config["tracking"]["use_l1"],
        config["tracking"]["ignore_outlier_depth_loss"],
        tracking=True,
        plot_dir=None,
        visualize_tracking_loss=False,
        tracking_iteration=capture_iteration,
        capture_frame=time_idx,
        capture_tracking_observation=True,
        capture_stage_override=capture_stage,
    )
    del loss

    edge = contract["block_edge_pixels"]
    required = contract["minimum_inflight_blocks"]
    slot_limit = contract["active_block_slots"]
    height, width = observation["depth_mask"].shape[-2:]
    if required > slot_limit:
        raise RuntimeError("frozen streaming candidate count exceeds the active block-slot contract")
    if not isinstance(observation["depth_loss_mask_sequence"], int):
        raise RuntimeError("streaming block capture requires a direct depth loss-mask observation")

    rotation_gradient = torch.zeros_like(params["cam_unnorm_rots"])
    translation_gradient = torch.zeros_like(params["cam_trans"])
    padded_mask = F.pad(observation["depth_mask"], (0, (-width) % edge, 0, (-height) % edge))
    tile_valid_pixels = padded_mask.unfold(-2, edge, edge).unfold(-2, edge, edge).sum(dim=(-1, -2)).detach().to("cpu")
    selected_tiles = []
    for y in range(0, height, edge):
        for x in range(0, width, edge):
            block_height = min(edge, height - y)
            block_width = min(edge, width - x)
            valid_pixels = int(tile_valid_pixels[..., y // edge, x // edge].item())
            if valid_pixels == 0:
                continue
            selected_tiles.append((x, y, block_width, block_height, valid_pixels))
            if len(selected_tiles) == required:
                break
        if len(selected_tiles) == required:
            break
    if len(selected_tiles) != required:
        raise RuntimeError(f"native Tracking did not produce {required} valid observable blocks")

    captured_gradients = []
    observability = []
    gradient_norms = []
    for x, y, block_width, block_height, valid_pixels in selected_tiles:
        block_loss = _native_tracking_block_loss(
            tracking_curr_data, observation, config["tracking"]["loss_weights"], x, y, block_width, block_height
        )
        rotation_grad, translation_grad = torch.autograd.grad(
            block_loss,
            (params["cam_unnorm_rots"], params["cam_trans"]),
            retain_graph=True,
            allow_unused=False,
        )
        gradient_norm = torch.sqrt(rotation_grad.square().sum() + translation_grad.square().sum())
        captured_gradients.append((rotation_grad, translation_grad, x, y, block_width, block_height, valid_pixels))
        gradient_norms.append(gradient_norm)
        observability.append(
            torch.isfinite(rotation_grad).all()
            & torch.isfinite(translation_grad).all()
            & torch.isfinite(gradient_norm)
            & (gradient_norm > 0)
        )
    # The default stream preserves tile order; materialize the evidence batch once.
    observable = torch.stack(observability).detach().to("cpu").tolist()
    gradient_norm_values = torch.stack(gradient_norms).detach().to("cpu").tolist()
    if not all(observable):
        raise RuntimeError("native Tracking selected a non-observable frozen streaming block")
    completed_blocks = []
    for completion_order, (rotation_grad, translation_grad, x, y, block_width, block_height, valid_pixels) in enumerate(captured_gradients):
        rotation_gradient.add_(rotation_grad.detach())
        translation_gradient.add_(translation_grad.detach())
        completed_blocks.append(
            {
                "block_slot": completion_order,
                "coordinates": {"x": x, "y": y, "width": block_width, "height": block_height},
                "valid_pixels": valid_pixels,
                "gradient_norm": float(gradient_norm_values[completion_order]),
                "pose_observability_valid": True,
                "completion_order": completion_order,
                "rasterizer_sequence": observation["rgb_rasterizer_sequence"],
                "loss_mask_sequence": observation["rgb_loss_mask_sequence"],
                "validity_mask_sequence": observation["depth_loss_mask_sequence"],
            }
        )
    return completed_blocks, rotation_gradient, translation_gradient


def _capture_native_successor_tracking_blocks(params, variables, config, target_frame, successor_tracking_data,
                                              capture_stage, capture_iteration, contract):
    """Audit a candidate or final pose against the immediate next native RGB-D frame."""
    successor_frame = target_frame + 1
    if successor_tracking_data.get("id") != successor_frame:
        raise RuntimeError("native streaming successor audit received a non-successor Tracking input")
    successor_snapshot = _snapshot_tracking_state(params, variables, successor_frame)
    try:
        initialize_camera_pose(params, successor_frame, forward_prop=config["tracking"]["forward_prop"])
        return _capture_native_tracking_blocks(
            params,
            variables,
            config,
            successor_frame,
            successor_frame,
            successor_tracking_data,
            capture_stage,
            capture_iteration,
            contract,
        )
    finally:
        _restore_tracking_state(params, variables, successor_frame, successor_snapshot)


def _native_tracking_quality(params, variables, config, iter_time_idx, tracking_curr_data):
    """Measure the active map by SplaTAM's native Tracking renderer without changing solver state."""
    capture = variables.get("_native_capture")
    radius = variables["max_2D_radius"].detach().clone()
    seen = variables.get("seen")
    seen_snapshot = seen.detach().clone() if isinstance(seen, torch.Tensor) else None
    variables["_native_capture"] = None
    try:
        loss, _, losses, observation = get_loss(
            params,
            tracking_curr_data,
            variables,
            iter_time_idx,
            config["tracking"]["loss_weights"],
            config["tracking"]["use_sil_for_loss"],
            config["tracking"]["sil_thres"],
            config["tracking"]["use_l1"],
            config["tracking"]["ignore_outlier_depth_loss"],
            tracking=True,
            plot_dir=None,
            visualize_tracking_loss=False,
            capture_tracking_observation=True,
        )
        global _native_lpips_model
        if _native_lpips_model is None:
            _native_lpips_model = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(observation["im"].device).eval()
        try:
            from simulator.quality import measure_valid_depth_rgb_quality
        except ModuleNotFoundError as error:
            raise RuntimeError("native quality capture requires the repository simulator on PYTHONPATH") from error
        result = {
            **measure_valid_depth_rgb_quality(
                observation["im"],
                tracking_curr_data["im"],
                tracking_curr_data["depth"],
                _native_lpips_model,
            ),
            "tracking_image_l1_sum": float(losses["im"].detach().item()),
            "tracking_depth_l1_sum": float(losses["depth"].detach().item()),
            "tracking_weighted_loss": float(loss.detach().item()),
        }
        del loss
        return result
    finally:
        with torch.no_grad():
            variables["max_2D_radius"].copy_(radius)
            if seen_snapshot is None:
                variables.pop("seen", None)
            else:
                variables["seen"] = seen_snapshot
        variables["_native_capture"] = capture


def _native_tracking_ate(params, tracking_curr_data, time_idx):
    from utils.eval_helpers import evaluate_ate

    with torch.no_grad():
        gt_trajectory = tracking_curr_data["iter_gt_w2c_list"][:time_idx + 1]
        estimated_trajectory = [tracking_curr_data["w2c"]]
        for frame in range(1, time_idx + 1):
            rotation = F.normalize(params["cam_unnorm_rots"][..., frame].detach())
            translation = params["cam_trans"][..., frame].detach()
            pose = torch.eye(4, device=rotation.device, dtype=rotation.dtype)
            pose[:3, :3] = build_rotation(rotation)
            pose[:3, 3] = translation
            estimated_trajectory.append(pose)
        return float(evaluate_ate(gt_trajectory, estimated_trajectory))


def _run_native_streaming_candidate(params, variables, config, time_idx, iter_time_idx, tracking_curr_data,
                                    contract, commit_iteration):
    capture = variables.get("_native_capture")
    if capture is None:
        raise RuntimeError("streaming candidate requires an active native capture")
    blocks, rotation_gradient, translation_gradient = _capture_native_tracking_blocks(
        params,
        variables,
        config,
        time_idx,
        iter_time_idx,
        tracking_curr_data,
        "streaming_candidate_render",
        commit_iteration,
        contract,
    )
    optimizer = initialize_optimizer(params, config["tracking"]["lrs"], tracking=True)
    params["cam_unnorm_rots"].grad = rotation_gradient
    params["cam_trans"].grad = translation_gradient
    _record_optimizer_step(
        variables,
        params,
        optimizer,
        time_idx,
        commit_iteration,
        "streaming_candidate",
        "before_step",
        parameter_names=("cam_unnorm_rots", "cam_trans"),
    )
    optimizer.step()
    _record_optimizer_step(
        variables,
        params,
        optimizer,
        time_idx,
        commit_iteration,
        "streaming_candidate",
        "after_step",
        parameter_names=("cam_unnorm_rots", "cam_trans"),
    )
    optimizer.zero_grad(set_to_none=True)
    candidate_rotation = params["cam_unnorm_rots"][..., time_idx].detach().clone()
    candidate_translation = params["cam_trans"][..., time_idx].detach().clone()
    accepted = bool(torch.isfinite(candidate_rotation).all().item() and torch.isfinite(candidate_translation).all().item())
    if not accepted:
        raise RuntimeError("native streaming candidate failed SplaTAM's finite pose validity check")
    schedule_sequence = capture.record_native_event(
        "streaming_schedule",
        "SplaTAM.native_streaming_candidate",
        {
            "frame": time_idx,
            "iteration": commit_iteration,
            "candidate_round": 0,
            "candidate_pose": {
                "parameterization": "quaternion_translation_7d",
                "parameters": {
                    "unnormalized_quaternion": candidate_rotation,
                    "translation": candidate_translation,
                },
                "normalization": {
                    "operation": "torch.nn.functional.normalize",
                    "quaternion": F.normalize(candidate_rotation),
                },
                "accepted": True,
            },
            "blocks": blocks,
        },
    )
    return {"schedule_sequence": schedule_sequence, "block_count": len(blocks)}


def _record_native_streaming_accuracy(params, variables, config, time_idx, iter_time_idx, tracking_curr_data,
                                      successor_tracking_data, eval_dir, strict_result, pre_tracking_state, contract):
    """Compare strict Tracking with a direct candidate followed by a native final recomputation."""
    capture = variables.get("_native_capture")
    strict_pose_sequence = strict_result.get("pose_sequence")
    if capture is None or strict_pose_sequence is None:
        raise RuntimeError("streaming accuracy capture requires a strict native pose observation")
    strict_capture_iteration = strict_result["completed_iterations"] - 1
    if strict_capture_iteration < 0:
        raise RuntimeError("streaming accuracy capture requires at least one native Tracking iteration")

    strict_quality = _native_tracking_quality(params, variables, config, iter_time_idx, tracking_curr_data)
    strict_ate = _native_tracking_ate(params, tracking_curr_data, time_idx)
    _restore_tracking_state(params, variables, time_idx, pre_tracking_state)
    candidate = _run_native_streaming_candidate(
        params, variables, config, time_idx, iter_time_idx, tracking_curr_data, contract, strict_capture_iteration
    )
    candidate_successor_blocks, _, _ = _capture_native_successor_tracking_blocks(
        params,
        variables,
        config,
        time_idx,
        successor_tracking_data,
        "streaming_candidate_successor_render",
        strict_capture_iteration,
        contract,
    )
    _restore_tracking_state(params, variables, time_idx, pre_tracking_state)
    final_result, _, _ = _run_native_tracking_solver(
        params,
        variables,
        config,
        time_idx,
        iter_time_idx,
        tracking_curr_data,
        eval_dir,
        emit_progress=False,
        capture_iterations={strict_result["completed_iterations"] - 1},
        capture_stage="streaming_final_render",
        optimizer_stage="streaming_final",
    )
    if final_result["completed_iterations"] != strict_result["completed_iterations"]:
        raise RuntimeError("strict and final native Tracking solves reached different iteration counts")
    final_capture_iteration = final_result["completed_iterations"] - 1
    if final_capture_iteration != strict_capture_iteration:
        raise RuntimeError("strict and final native Tracking renders use different final iterations")
    final_pose_sequence = final_result.get("pose_sequence")
    if final_pose_sequence is None:
        raise RuntimeError("streaming final recomputation did not emit a native pose observation")
    final_quality = _native_tracking_quality(params, variables, config, iter_time_idx, tracking_curr_data)
    final_ate = _native_tracking_ate(params, tracking_curr_data, time_idx)
    replayed_successor_blocks, _, _ = _capture_native_successor_tracking_blocks(
        params,
        variables,
        config,
        time_idx,
        successor_tracking_data,
        "streaming_final_replay_render",
        final_capture_iteration,
        contract,
    )
    if not candidate_successor_blocks:
        raise RuntimeError("native streaming candidate produced no downstream block work")
    replay_ratio = len(replayed_successor_blocks) / len(candidate_successor_blocks)
    capture.record_native_event(
        "accuracy",
        "SplaTAM.native_streaming_accuracy",
        {
            "strict_commit": {
                "ate": strict_ate,
                "mapping_quality": strict_quality,
                "quality_protocol": _NATIVE_QUALITY_PROTOCOL,
                "tracking_convergence": {
                    "completed_iterations": strict_result["completed_iterations"],
                    "best_loss": strict_result["best_loss"],
                    "final_weighted_loss": strict_result["losses"]["loss"],
                },
                "native_pose_sequence": strict_pose_sequence,
            },
            "streaming_commit": {
                "ate": final_ate,
                "mapping_quality": final_quality,
                "quality_protocol": _NATIVE_QUALITY_PROTOCOL,
                "tracking_convergence": {
                    "completed_iterations": final_result["completed_iterations"],
                    "best_loss": final_result["best_loss"],
                    "final_weighted_loss": final_result["losses"]["loss"],
                    "candidate_successor_blocks": len(candidate_successor_blocks),
                    "replayed_successor_blocks": len(replayed_successor_blocks),
                },
                "replay_ratio": replay_ratio,
                "convergence_rounds": 1,
                "schedule_sequence": candidate["schedule_sequence"],
                "final_native_pose_sequence": final_pose_sequence,
            },
        },
    )


def _run_native_tracking_solver(params, variables, config, time_idx, iter_time_idx, tracking_curr_data, eval_dir,
                                wandb_run=None, wandb_tracking_step=0, wandb_time_step=0, emit_progress=True,
                                capture_iterations=None, capture_stage="tracking_render", optimizer_stage="tracking"):
    """Run SplaTAM's Tracking loop for normal execution and withdrawal replay."""
    optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
    candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
    candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
    current_min_loss = float(1e20)
    iter = 0
    do_continue_slam = False
    num_iters_tracking = config['tracking']['num_iters']
    termination = None
    tracking_iteration_seconds = 0.0
    pose_coverage_stage = "tracking_pose_coverage" if capture_stage == "tracking_render" else f"{capture_stage}_pose_coverage"
    progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
    while True:
        iter_start_time = time.time()
        capture = variables.get("_native_capture")
        capture_this_iteration = capture_iterations is None or iter in capture_iterations
        if capture is not None and not capture_this_iteration:
            variables["_native_capture"] = None
        try:
            loss, variables, losses = get_loss(
                params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True,
                plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                tracking_iteration=iter, capture_frame=time_idx, capture_stage_override=capture_stage,
            )
            if config['use_wandb'] and wandb_run is not None:
                wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
            loss.backward()
            _record_optimizer_step(
                variables,
                params,
                optimizer,
                time_idx,
                iter,
                optimizer_stage,
                "before_step",
                parameter_names=("cam_unnorm_rots", "cam_trans") if capture_iterations is not None else None,
            )
            optimizer.step()
            _record_optimizer_step(
                variables,
                params,
                optimizer,
                time_idx,
                iter,
                optimizer_stage,
                "after_step",
                parameter_names=("cam_unnorm_rots", "cam_trans") if capture_iterations is not None else None,
            )
            optimizer.zero_grad(set_to_none=True)
        finally:
            variables["_native_capture"] = capture
        with torch.no_grad():
            if loss < current_min_loss:
                current_min_loss = loss
                candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            if emit_progress and config['report_iter_progress']:
                if config['use_wandb'] and wandb_run is not None:
                    report_progress(
                        params, tracking_curr_data, iter + 1, progress_bar, iter_time_idx,
                        sil_thres=config['tracking']['sil_thres'], tracking=True, wandb_run=wandb_run,
                        wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'],
                    )
                else:
                    report_progress(params, tracking_curr_data, iter + 1, progress_bar, iter_time_idx,
                                    sil_thres=config['tracking']['sil_thres'], tracking=True)
            else:
                progress_bar.update(1)
        tracking_iteration_seconds += time.time() - iter_start_time
        iter += 1
        if iter == num_iters_tracking:
            if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                termination = {
                    "reason": "depth_loss_threshold",
                    "threshold_enabled": True,
                    "threshold": config['tracking']['depth_loss_thres'],
                }
                break
            if config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                do_continue_slam = True
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                num_iters_tracking = 2 * num_iters_tracking
                if config['use_wandb'] and wandb_run is not None:
                    wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx, "Tracking/step": wandb_time_step})
            else:
                termination = {
                    "reason": "iteration_limit",
                    "threshold_enabled": config['tracking']['use_depth_loss_thres'],
                    "configured_iterations": num_iters_tracking,
                }
                break

    progress_bar.close()
    if capture_iterations is not None and iter - 1 not in capture_iterations:
        raise RuntimeError("final native Tracking iteration was excluded from the requested capture scope")
    final_optimizer_iteration = iter - 1
    with torch.no_grad():
        params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
        params['cam_trans'][..., time_idx] = candidate_cam_tran
        pose_sequence = _record_pose(
            variables, time_idx, final_optimizer_iteration, "candidate_committed", candidate_cam_unnorm_rot, candidate_cam_tran,
            losses, do_continue_slam, source_frame=iter_time_idx, last_optimizer_iteration=final_optimizer_iteration,
            termination=termination, best_loss=current_min_loss, coverage_stage=pose_coverage_stage,
            optimizer_stage=optimizer_stage,
        )
    result = {
        "rotation": candidate_cam_unnorm_rot,
        "translation": candidate_cam_tran,
        "losses": {name: float(value.detach().item()) for name, value in losses.items()},
        "best_loss": float(current_min_loss.detach().item()),
        "extended_tracking": do_continue_slam,
        "completed_iterations": iter,
        "termination": termination,
        "pose_sequence": pose_sequence,
    }
    return result, tracking_iteration_seconds, wandb_tracking_step


def _record_native_withdrawal_replay(params, variables, config, time_idx, iter_time_idx, tracking_curr_data,
                                      eval_dir, baseline_result):
    """Withdraw one exact Mapping tile group and run the unmodified Tracking solver."""
    group = variables.get("_native_impact_group")
    capture = variables.get("_native_capture")
    capture_frame = variables.get("_native_streaming_capture_frame")
    if group is None or capture is None or time_idx != capture_frame:
        return 0.0

    replay_start_time = time.time()
    params_after_tracking = _snapshot_tensor_values(params)
    variables_after_tracking = _snapshot_tensor_values(variables)
    variables["_native_capture"] = None
    try:
        _withdraw_native_mapping_group(params, variables, group)
        initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
        replay_result, _, _ = _run_native_tracking_solver(
            params, variables, config, time_idx, iter_time_idx, tracking_curr_data, eval_dir,
            emit_progress=False,
        )
    finally:
        _restore_tensor_values(params, params_after_tracking)
        for name, value in variables_after_tracking.items():
            variables[name] = value
        variables["_native_capture"] = capture

    verification_outcome = _tracking_result_is_finite(replay_result)
    capture.record_native_event(
        "impact",
        "SplaTAM.native_withdrawal_replay",
        {
            "frame": time_idx,
            "iteration": group["iteration"],
            "stage": "tracking_withdrawal_replay",
            "stable_gaussian_ids": group["stable_gaussian_ids"],
            "consumer_group": 0,
            "consumer_group_generation": 0,
            "offline_label": "conservative",
            "label_source": "native Mapping-tile withdrawal replay; no upstream low-impact acceptance rule",
            "withdrawal_replay_id": f"splatam-frame-{time_idx}-map-{group['frame']}-iter-{group['iteration']}-tile-{group['tile_index']}",
            "false_positive": False,
            "false_negative": False,
            "local_verification": {
                "outcome": verification_outcome,
                "method": "native_tracking_solver_finite_result",
            },
            "consumer_group_binding": {
                "native_rasterizer_sequence": group["rasterizer_sequence"],
                "native_tile_index": group["tile_index"],
                "native_tile_range": group["tile_range"],
                "fresh_replay_group": True,
            },
            "withdrawal": {
                "source_frame": group["frame"],
                "source_iteration": group["iteration"],
                "source_view": group["source_frame"],
                "parameter_names": list(_GAUSSIAN_PARAMETER_NAMES),
                "operation": "restore_pre_mapping_values_for_native_tile_group",
            },
            "baseline_tracking": _tracking_result_summary(baseline_result),
            "withdrawn_tracking": _tracking_result_summary(replay_result),
        },
    )
    return time.time() - replay_start_time


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        _capture_replace_ids(params, variables, "load_checkpoint", "SplaTAM.rgbd_slam")
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0

    if config.get("native_impact_replay", False):
        if variables.get("_native_capture") is None:
            raise RuntimeError("native_impact_replay requires THREEDGS_SLAM_NATIVE_CAPTURE=1")
        mapping_iterations = config["mapping"]["num_iters"]
        if not isinstance(mapping_iterations, int) or mapping_iterations <= 0:
            raise RuntimeError("native_impact_replay requires a positive integer mapping.num_iters")
        variables["_native_impact_replay_enabled"] = True
        variables["_native_impact_capture_iteration"] = mapping_iterations - 1

    if config.get("native_streaming_accuracy", False):
        if variables.get("_native_capture") is None:
            raise RuntimeError("native_streaming_accuracy requires THREEDGS_SLAM_NATIVE_CAPTURE=1")
        if config.get("native_capture_scope") != "streaming_accuracy":
            raise RuntimeError("native_streaming_accuracy requires native_capture_scope='streaming_accuracy'")
        if not config.get("native_impact_replay", False):
            raise RuntimeError("native_streaming_accuracy requires the native Impact withdrawal replay")
        if config["tracking"]["use_gt_poses"] or config["tracking"]["use_depth_loss_thres"]:
            raise RuntimeError("native_streaming_accuracy requires unextended native optimizer Tracking")
        capture_frame = config.get("native_streaming_capture_frame", num_frames - 2)
        if not isinstance(capture_frame, int) or isinstance(capture_frame, bool) or capture_frame <= 0 or capture_frame != num_frames - 2:
            raise RuntimeError(
                "native_streaming_capture_frame must select the penultimate configured frame for a real successor audit"
            )
        variables["_native_streaming_accuracy_enabled"] = True
        variables["_native_streaming_capture_frame"] = capture_frame
        variables["_native_streaming_contract"] = _load_frozen_streaming_capture_contract()
        variables["_native_minimal_capture_scope"] = True
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Tracking
        tracking_start_time = time.time()
        native_tracking_result = None
        withdrawal_replay_seconds = 0.0
        streaming_accuracy_seconds = 0.0
        streaming_target = bool(
            variables.get("_native_streaming_accuracy_enabled")
            and time_idx == variables["_native_streaming_capture_frame"]
        )
        streaming_successor = bool(
            variables.get("_native_streaming_accuracy_enabled")
            and time_idx == variables["_native_streaming_capture_frame"] + 1
        )
        capture_streaming_tracking = streaming_target or streaming_successor
        pre_tracking_state = _snapshot_tracking_state(params, variables, time_idx) if streaming_target else None
        successor_tracking_data = None
        if streaming_target:
            successor_tracking_data = _native_successor_tracking_data(
                tracking_dataset if seperate_tracking_res else dataset,
                time_idx,
                tracking_curr_data,
            )
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            capture = variables.get("_native_capture")
            if variables.get("_native_minimal_capture_scope") and not capture_streaming_tracking:
                variables["_native_capture"] = None
            try:
                if streaming_successor:
                    _capture_replace_ids(params, variables, "resume_successor_tracking_capture", "SplaTAM.rgbd_slam")
                native_tracking_result, tracking_seconds, wandb_tracking_step = _run_native_tracking_solver(
                    params, variables, config, time_idx, iter_time_idx, tracking_curr_data, eval_dir,
                    wandb_run=wandb_run if config['use_wandb'] else None,
                    wandb_tracking_step=wandb_tracking_step if config['use_wandb'] else 0,
                    wandb_time_step=wandb_time_step if config['use_wandb'] else 0,
                    capture_iterations={config["tracking"]["num_iters"] - 1} if capture_streaming_tracking else None,
                )
            finally:
                variables["_native_capture"] = capture
            tracking_iter_time_sum += tracking_seconds
            tracking_iter_time_count += native_tracking_result["completed_iterations"]
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
                _record_pose(
                    variables,
                    time_idx,
                    0,
                    "ground_truth_assigned",
                    rel_w2c_rot_quat,
                    rel_w2c_tran,
                    {},
                    False,
                    source_frame=iter_time_idx,
                )
        if streaming_target and native_tracking_result is not None:
            if successor_tracking_data is None:
                raise RuntimeError("native streaming accuracy has no immediate successor Tracking input")
            streaming_start_time = time.time()
            _record_native_streaming_accuracy(
                params,
                variables,
                config,
                time_idx,
                iter_time_idx,
                tracking_curr_data,
                successor_tracking_data,
                eval_dir,
                native_tracking_result,
                pre_tracking_state,
                variables["_native_streaming_contract"],
            )
            streaming_accuracy_seconds = time.time() - streaming_start_time
        if variables.get("_native_impact_replay_enabled") and native_tracking_result is not None:
            withdrawal_replay_seconds = _record_native_withdrawal_replay(
                params, variables, config, time_idx, iter_time_idx, tracking_curr_data, eval_dir,
                native_tracking_result,
            )
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time - withdrawal_replay_seconds - streaming_accuracy_seconds
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                capture = variables.get("_native_capture")
                if variables.get("_native_minimal_capture_scope"):
                    variables["_native_capture"] = None
                try:
                    params, variables = add_new_gaussians(params, variables, densify_curr_data,
                                                          config['mapping']['sil_thres'], time_idx,
                                                          config['mean_sq_dist_method'], config['gaussian_distribution'])
                finally:
                    variables["_native_capture"] = capture
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            if variables.get("_native_impact_replay_enabled") and time_idx == 0:
                _snapshot_impact_baseline(params, variables)
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                capture = variables.get("_native_capture")
                capture_mapping_tile = (
                    time_idx == 0 and iter == variables.get("_native_impact_capture_iteration")
                )
                if variables.get("_native_minimal_capture_scope") and not capture_mapping_tile:
                    variables["_native_capture"] = None
                try:
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True,
                                                    capture_iteration=iter, capture_frame=time_idx)
                finally:
                    variables["_native_capture"] = capture
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    if not variables.get("_native_minimal_capture_scope"):
                        _record_optimizer_step(variables, params, optimizer, time_idx, iter, "mapping", "before_step")
                    optimizer.step()
                    if not variables.get("_native_minimal_capture_scope"):
                        _record_optimizer_step(variables, params, optimizer, time_idx, iter, "mapping", "after_step")
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()
    capture = variables.get("_native_capture")
    if capture is not None:
        capture.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
