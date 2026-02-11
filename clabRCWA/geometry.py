import torch
import math


def _coord_grid(resolution, device, dtype):
    """
    Create meshgrid coordinates in pixel units.
    
    Args:
        resolution: Grid size
        device: Torch device
        dtype: Data type
        
    Returns:
        x, y: Coordinate grids (resolution × resolution each)
        
    Note:
        Uses indexing='xy' to match RCWA convention:
        - x varies along columns (2nd dimension, Lambda_x direction)
        - y varies along rows (1st dimension, Lambda_y direction)
    """
    coords = torch.arange(resolution, device=device, dtype=dtype)
    return torch.meshgrid(coords, coords, indexing="xy")


def circle(
    resolution: int,
    p: float,
    r: float,
    x0: float = None,
    y0: float = None,
    device="cpu",
    dtype=torch.float32,
):
    """
    Generate a circular inclusion pattern.
    
    Args:
        resolution: Grid resolution
        p: Physical period (for scaling)
        r: Radius in physical units
        x0, y0: Center coordinates in physical units (default: center)
                x0 is horizontal position, y0 is vertical position
        device: Torch device
        dtype: Data type
        
    Returns:
        Binary image tensor (1 inside circle, 0 outside)
        Shape: (resolution, resolution) where [i, j] represents position (y=i, x=j)
    """
    ratio = resolution / p
    r_pix = r * ratio
    if x0 is None:
        x0_pix = resolution / 2
    else:
        x0_pix = x0 * ratio
    if y0 is None:
        y0_pix = resolution / 2
    else:
        y0_pix = y0 * ratio
    x, y = _coord_grid(resolution, device, dtype)
    img = ((x - x0_pix) ** 2 + (y - y0_pix) ** 2 < r_pix ** 2).to(dtype)
    return img


def ellipse(
    resolution: int,
    p: float,
    rx: float,
    ry: float,
    x0: float = None,
    y0: float = None,
    device="cpu",
    dtype=torch.float32,
):
    """
    Generate an elliptical inclusion pattern.
    
    Args:
        resolution: Grid resolution
        p: Physical period (for scaling)
        rx: Radius in x direction (horizontal, physical units)
        ry: Radius in y direction (vertical, physical units)
        x0, y0: Center coordinates in physical units (default: center)
                x0 is horizontal position, y0 is vertical position
        device: Torch device
        dtype: Data type
        
    Returns:
        Binary image tensor (1 inside ellipse, 0 outside)
        Shape: (resolution, resolution) where [i, j] represents position (y=i, x=j)
    """
    ratio = resolution / p
    rx_pix = rx * ratio
    ry_pix = ry * ratio
    if x0 is None:
        x0_pix = resolution / 2
    else:
        x0_pix = x0 * ratio
    if y0 is None:
        y0_pix = resolution / 2
    else:
        y0_pix = y0 * ratio
    x, y = _coord_grid(resolution, device, dtype)
    img = (((x - x0_pix) / rx_pix) ** 2 + ((y - y0_pix) / ry_pix) ** 2 < 1).to(dtype)
    return img


def rectangle(
    resolution: int,
    p: float,
    wx: float,
    wy: float,
    x0: float = None,
    y0: float = None,
    device="cpu",
    dtype=torch.float32,
):
    """
    Generate a rectangular inclusion pattern.
    
    Args:
        resolution: Grid resolution
        p: Physical period (for scaling)
        wx: Width in x direction (horizontal, physical units)
        wy: Width in y direction (vertical, physical units)
        x0, y0: Center coordinates in physical units (default: center)
                x0 is horizontal position, y0 is vertical position
        device: Torch device
        dtype: Data type
        
    Returns:
        Binary image tensor (1 inside rectangle, 0 outside)
        Shape: (resolution, resolution) where [i, j] represents position (y=i, x=j)
    """
    ratio = resolution / p
    wx_pix = wx * ratio
    wy_pix = wy * ratio
    if x0 is None:
        x0_pix = resolution / 2
    else:
        x0_pix = x0 * ratio
    if y0 is None:
        y0_pix = resolution / 2
    else:
        y0_pix = y0 * ratio
    x, y = _coord_grid(resolution, device, dtype)
    img = (
        (torch.abs(x - x0_pix) < wx_pix / 2) & 
        (torch.abs(y - y0_pix) < wy_pix / 2)
    ).to(dtype)
    return img


def square(
    resolution: int,
    p: float,
    w: float,
    x0: float = None,
    y0: float = None,
    device="cpu",
    dtype=torch.float32,
):
    """
    Generate a square inclusion pattern.
    
    Args:
        resolution: Grid resolution
        p: Physical period (for scaling)
        w: Width of square (physical units)
        x0, y0: Center coordinates in physical units (default: center)
                x0 is horizontal position, y0 is vertical position
        device: Torch device
        dtype: Data type
        
    Returns:
        Binary image tensor (1 inside square, 0 outside)
        Shape: (resolution, resolution) where [i, j] represents position (y=i, x=j)
    """
    return rectangle(resolution, p, w, w, x0, y0, device, dtype)


def binary_1d(
    resolution: int,
    p: float,
    w: float,
    x0: float = None,
    y0: float = None,
    axis: str = "x",
    device="cpu",
    dtype=torch.float32,
):
    """
    Generate a 1D binary grating pattern.
    
    Args:
        resolution: Grid resolution
        p: Physical period (for scaling)
        w: Width of stripe in physical units
        x0, y0: Center position in physical units (default: center)
                For axis='x', uses x0 (horizontal position, creates vertical stripes)
                For axis='y', uses y0 (vertical position, creates horizontal stripes)
        axis: 'x' for vertical stripes (constant along x), 
              'y' for horizontal stripes (constant along y)
        device: Torch device
        dtype: Data type
        
    Returns:
        Binary image tensor (1 inside stripe, 0 outside)
        Shape: (resolution, resolution) where [i, j] represents position (y=i, x=j)
    """
    ratio = resolution / p
    w_pix = w * ratio
    
    if axis == "x":
        center = x0
    elif axis == "y":
        center = y0
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
    
    if center is None:
        center_pix = resolution / 2
    else:
        center_pix = center * ratio
    
    coords = torch.arange(resolution, device=device, dtype=dtype)
    mask = (
        (torch.abs(coords - center_pix) < w_pix / 2) |
        (torch.abs(coords + resolution - center_pix) < w_pix / 2) |
        (torch.abs(coords - resolution - center_pix) < w_pix / 2)
    )
    img = torch.zeros((resolution, resolution), device=device, dtype=dtype)
    if axis == "x":
        img[:, mask] = 1  # Set all rows at these columns (vertical stripes)
    elif axis == "y":
        img[mask, :] = 1  # Set all columns at these rows (horizontal stripes)
    return img


def rotate(
    img: torch.Tensor,
    angle: float,
    x0: float = None,
    y0: float = None,
):
    """
    Rotate a geometry pattern counter-clockwise (from +x to +y).
    
    Args:
        img: Input geometry tensor (resolution × resolution)
        angle: Rotation angle in degrees (positive = counter-clockwise from +x to +y)
        x0, y0: Rotation center in pixel coordinates (default: image center)
                x0 is horizontal position, y0 is vertical position
        
    Returns:
        Rotated geometry tensor (same shape as input)
        
    Note:
        Uses bilinear interpolation for smooth rotation.
        Values outside the original image are filled with 0.
        
    Examples:
        # Rotate a rectangle by 45 degrees
        rect = geometry.rectangle(512, 0.4, 0.2, 0.1, device=device)
        rotated = geometry.rotate(rect, 45)
        
        # Rotate around a custom center
        rotated = geometry.rotate(rect, 30, x0=256, y0=128)
    """
    device = img.device
    dtype = img.dtype
    resolution = img.shape[0]
    
    # Default rotation center is image center
    if x0 is None:
        x0 = resolution / 2
    if y0 is None:
        y0 = resolution / 2
    
    # Convert angle to radians and compute trig values (just regular Python math)
    angle_rad = angle * math.pi / 180
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Create coordinate grids
    x, y = _coord_grid(resolution, device, dtype)
    
    # Translate to rotation center
    x_centered = x - x0
    y_centered = y - y0
    
    # Apply rotation (counter-clockwise)
    x_rot = cos_a * x_centered - sin_a * y_centered + x0
    y_rot = sin_a * x_centered + cos_a * y_centered + y0
    
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2 * x_rot / (resolution - 1) - 1
    y_norm = 2 * y_rot / (resolution - 1) - 1
    
    # Stack into grid (batch, height, width, 2)
    # grid_sample expects (x, y) order
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    
    # Prepare input for grid_sample (batch, channels, height, width)
    img_input = img.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation using bilinear interpolation
    rotated = torch.nn.functional.grid_sample(
        img_input.float(),
        grid.float(),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # Remove batch and channel dimensions and convert back to original dtype
    return rotated.squeeze(0).squeeze(0).to(dtype)


def homogeneous(resolution: int, device="cpu", dtype=torch.float32):
    """
    Generate a homogeneous (empty) pattern.
    
    Args:
        resolution: Grid resolution
        device: Torch device
        dtype: Data type
        
    Returns:
        Zero tensor with shape (resolution, resolution)
    """
    return torch.zeros((resolution, resolution), device=device, dtype=dtype)