import torch
import torch.nn.functional as F


# Estimate only the diagonal entries of the local 3D covariance.
@torch.no_grad()
def local_diag_cov_from_X1(
    X1: torch.Tensor,     # (H*W, 3), flattened in row-major pixel order.
    H: int, W: int,       
    win: int = 5,         
    valid: torch.Tensor | None = None,  # Optional (H*W,) mask; defaults to finite points with positive depth.
    var_floor: float = 1e-12            # Lower bound applied to covariance entries.
) -> torch.Tensor:

    device = X1.device
    assert X1.shape[0] == H*W and X1.shape[1] == 3, "X1 should be (H*W,3)"
    pad = win // 2

    # (H,W,3)
    Xv = X1.view(H, W, 3)

    
    if valid is None:
        valid_hw1 = (torch.isfinite(Xv).all(-1) & (Xv[..., 2] > 0)).float().unsqueeze(-1)
    else:
        assert valid.shape[0] == H*W
        valid_hw1 = valid.view(H, W, 1).float()

    # Compute local averages with a box filter.
    def boxfilter(img_hwC):
        x = img_hwC.permute(2,0,1).unsqueeze(0)             # (1,C,H,W)
        x = F.pad(x, (pad,pad,pad,pad), mode='reflect')
        y = F.avg_pool2d(x, kernel_size=win, stride=1)      # (1,C,H,W)
        return y.squeeze(0).permute(1,2,0)                  # (H,W,C)

    denom = boxfilter(valid_hw1).clamp_min(1e-9)
    mean  = boxfilter(Xv * valid_hw1) / denom               # E[X]
    ex2   = boxfilter((Xv * Xv) * valid_hw1) / denom        # E[X^2]
    var   = (ex2 - mean * mean).clamp_min(var_floor)        # Var = E[X^2]-E[X]^2

    
    N = H*W
    Sigma = torch.zeros(N, 3, 3, device=device)
    var_flat = var.view(N, 3)
    Sigma[:, 0, 0] = var_flat[:, 0]   # Var(X)
    Sigma[:, 1, 1] = var_flat[:, 1]   # Var(Y)
    Sigma[:, 2, 2] = var_flat[:, 2]   # Var(Z)

    return Sigma

@torch.no_grad()
def local_cov_from_X1(
    X1: torch.Tensor,     # (H*W, 3), flattened in row-major pixel order.
    H: int, W: int,       
    win: int = 5,         # Odd neighborhood width in pixels.
    valid: torch.Tensor | None = None,  # Optional (H*W,) mask; defaults to finite points with positive depth.
    var_floor: float = 1e-12            # Lower bound applied to covariance entries.
) -> tuple[torch.Tensor, torch.Tensor]:

    device = X1.device
    assert X1.shape[0] == H*W and X1.shape[1] == 3, "X1 should be (H*W,3)"
    pad = win // 2

    # (H,W,3)
    Xv = X1.view(H, W, 3)

    
    if valid is None:
        valid_hw1 = (torch.isfinite(Xv).all(-1) & (Xv[..., 2] > 0)).float().unsqueeze(-1)
    else:
        assert valid.shape[0] == H*W
        valid_hw1 = valid.view(H, W, 1).float()

    # Compute local averages with a box filter.
    def boxfilter(img_hwC):
        x = img_hwC.permute(2,0,1).unsqueeze(0)             # (1,C,H,W)
        x = F.pad(x, (pad,pad,pad,pad), mode='reflect')
        y = F.avg_pool2d(x, kernel_size=win, stride=1)      # (1,C,H,W)
        return y.squeeze(0).permute(1,2,0)                  # (H,W,C)

    denom = boxfilter(valid_hw1).clamp_min(1e-9)
    mean  = boxfilter(Xv * valid_hw1) / denom               # E[X] = (H,W,3)
    
    # Estimate E[XX^T] over valid points in each pixel neighborhood.
    
    Xv_expanded = Xv.unsqueeze(-1)  # (H,W,3,1)
    XXT = Xv_expanded @ Xv_expanded.transpose(-2, -1)  
    
    
    # Broadcast the validity mask across the two covariance dimensions.
    valid_expanded = valid_hw1.unsqueeze(-1)  # (H,W,1,1)
    XXT_valid = XXT * valid_expanded  
    
    XXT_flat = XXT_valid.view(H, W, 9)  # (H,W,9)
    E_XXT_flat = boxfilter(XXT_flat) / denom  # (H,W,9) - E[XX^T]
    E_XXT = E_XXT_flat.view(H, W, 3, 3)  # (H,W,3,3)
    
    
    mean_expanded = mean.unsqueeze(-1)  # (H,W,3,1)
    E_X_E_XT = mean_expanded @ mean_expanded.transpose(-2, -1)  # (H,W,3,3) - E[X]E[X]^T
    
    # Covariance is E[XX^T] - E[X]E[X]^T.
    Sigma_hw = E_XXT - E_X_E_XT  # (H,W,3,3)
    
    
    N = H*W
    Sigma = Sigma_hw.view(N, 3, 3)
    
    Sigma = (Sigma + Sigma.transpose(-2, -1)) / 2  # Enforce symmetry.
    Sigma = torch.clamp(Sigma, min=var_floor)  # Clamp every covariance entry to var_floor.
    
    
    eigenvals, _ = torch.linalg.eigh(Sigma)  # Eigenvalues are returned in ascending order.
    
    lambda_0 = eigenvals[:, 0]  
    lambda_2 = eigenvals[:, 2]  
    
    lambda_2 = lambda_2.clamp_min(1e-12)  
    planarity = 1.0 - lambda_0 / lambda_2  # (N,)
    planarity = torch.clamp(planarity, min=0.0, max=1.0)
    
    
    planarity = torch.where(
        torch.isfinite(planarity),
        planarity,
        torch.zeros_like(planarity)
    )
    
    return Sigma, planarity

import numpy as np
import cv2

def visualize_covariance_max(Sigma, H, W,
                             cmap: str = "inferno",
                             win_name: str = "Max Covariance (per-pixel)",
                             low_percent: float = 20.0,
                             high_percent: float = 80.0):
    try:
        import torch
        if isinstance(Sigma, torch.Tensor):
            Sigma = Sigma.detach().cpu().numpy()
    except ImportError:
        pass

    assert Sigma.shape == (H*W, 3, 3), f"Sigma shape should be {(H*W,3,3)}, got {Sigma.shape}"

    # Use the largest diagonal variance as the per-pixel uncertainty score.
    diag = np.stack([Sigma[:,0,0], Sigma[:,1,1], Sigma[:,2,2]], axis=-1)  # (N,3)
    smax = np.max(diag, axis=1).reshape(H, W)  # (H,W)
    smax = np.linalg.det(Sigma).reshape(H, W)

    # Normalize using percentile bounds to limit the effect of outliers.
    finite_mask = np.isfinite(smax)
    if np.any(finite_mask):
        lo = np.percentile(smax[finite_mask], low_percent)
        hi = np.percentile(smax[finite_mask], high_percent)
        if hi <= lo: hi = lo + 1e-12
        norm = np.clip((smax - lo) / (hi - lo), 0, 1)
    else:
        norm = np.zeros((H, W), dtype=np.float32)

    
    gray = (norm * 255.0).astype(np.uint8)

    
    cmap_dict = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma":   cv2.COLORMAP_MAGMA,
        "plasma":  cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo":   cv2.COLORMAP_TURBO,
        "jet":     cv2.COLORMAP_JET,
    }
    cv_cmap = cmap_dict.get(cmap.lower(), cv2.COLORMAP_INFERNO)

    # OpenCV returns the colorized image in BGR order.
    color_bgr = cv2.applyColorMap(gray, cv_cmap)

    
    cv2.imshow(win_name, color_bgr)
    cv2.waitKey(-1)
    cv2.destroyWindow(win_name)

    return color_bgr  