import torch
import torch.linalg as la


def format_time(total_seconds: float) -> str:
    """
    Format time duration into human-readable string.
    
    Args:
        total_seconds: Time duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 34m 15.3s" or "45.2s")
    """
    parts = []
    days = int(total_seconds // 86400)
    if days > 0:
        parts.append(f"{days}d")
    total_seconds %= 86400
    hours = int(total_seconds // 3600)
    if hours > 0:
        parts.append(f"{hours}h")
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    if minutes > 0:
        parts.append(f"{minutes}m")
    total_seconds %= 60
    parts.append(f"{total_seconds:.3f}s")
    return " ".join(parts)


def compute_diffraction_efficiency(rs, ts, Kz_norm_ref, Kz_norm_trn, kz_norm, mu_ref, mu_trn, n_harmonics):
    """
    Compute reflection and transmission diffraction efficiencies.
    
    Args:
        rs: [rx, ry, rz] reflection field components
        ts: [tx, ty, tz] transmission field components
        Kz_norm_ref: Normalized kz for reflection
        Kz_norm_trn: Normalized kz for transmission
        kz_norm: Incident normalized kz
        mu_ref: Permeability in reflection region
        mu_trn: Permeability in transmission region
        n_harmonics: Number of harmonics
        
    Returns:
        Rs: [Rx, Ry, Rz] reflection efficiencies
        Ts: [Tx, Ty, Tz] transmission efficiencies
    """
    rx, ry, rz = rs
    tx, ty, tz = ts
    
    Rx = torch.matmul(
        torch.real(-Kz_norm_ref / mu_ref) / torch.real(kz_norm / mu_ref),
        torch.square(rx.abs())
    )
    Ry = torch.matmul(
        torch.real(-Kz_norm_ref / mu_ref) / torch.real(kz_norm / mu_ref),
        torch.square(ry.abs())
    )
    Rz = torch.matmul(
        torch.real(-Kz_norm_ref / mu_ref) / torch.real(kz_norm / mu_ref),
        torch.square(rz.abs())
    )
    
    Tx = torch.matmul(
        torch.real(Kz_norm_trn / mu_trn) / torch.real(kz_norm / mu_ref),
        torch.square(tx.abs())
    )
    Ty = torch.matmul(
        torch.real(Kz_norm_trn / mu_trn) / torch.real(kz_norm / mu_ref),
        torch.square(ty.abs())
    )
    Tz = torch.matmul(
        torch.real(Kz_norm_trn / mu_trn) / torch.real(kz_norm / mu_ref),
        torch.square(tz.abs())
    )
    
    Rs = [Rx, Ry, Rz]
    Ts = [Tx, Ty, Tz]
    
    return Rs, Ts


def extract_field_by_order(rs, ts, nx, ny, Kz_norm_ref, Kz_norm_trn, kz_norm, mu_ref, mu_trn, n_harmonics, device, dtype):
    """
    Extract field components for a specific diffraction order.
    
    Args:
        rs: [rx, ry, rz] reflection field components
        ts: [tx, ty, tz] transmission field components
        nx, ny: Diffraction order indices
        Kz_norm_ref: Normalized kz for reflection
        Kz_norm_trn: Normalized kz for transmission
        kz_norm: Incident normalized kz
        mu_ref: Permeability in reflection region
        mu_trn: Permeability in transmission region
        n_harmonics: Number of harmonics
        device: Torch device
        dtype: Data type
        
    Returns:
        r_field: [rx, ry, rz] for order (nx, ny)
        t_field: [tx, ty, tz] for order (nx, ny)
    """
    n_half = int((n_harmonics - 1) / 2)
    index = (-ny + n_half) * n_harmonics - nx + n_half
    
    Kr = torch.sqrt(torch.real(-Kz_norm_ref / mu_ref) / torch.real(kz_norm / mu_ref)).type(dtype)
    rx = torch.conj(torch.matmul(Kr, rs[0])[index])
    ry = torch.conj(torch.matmul(Kr, rs[1])[index])
    rz = torch.conj(torch.matmul(Kr, rs[2])[index])
    
    Kt = torch.sqrt(torch.real(Kz_norm_trn / mu_trn) / torch.real(kz_norm / mu_ref)).type(dtype)
    tx = torch.conj(torch.matmul(Kt, ts[0])[index])
    ty = torch.conj(torch.matmul(Kt, ts[1])[index])
    tz = torch.conj(torch.matmul(Kt, ts[2])[index])
    
    r_field = torch.tensor([rx, ry, rz], device=device)
    t_field = torch.tensor([tx, ty, tz], device=device)
    
    return r_field, t_field


def validate_energy_conservation(Rs, Ts, tolerance=1e-4):
    """
    Validate energy conservation (R + T should equal 1).
    
    Args:
        Rs: [Rx, Ry, Rz] reflection efficiencies
        Ts: [Tx, Ty, Tz] transmission efficiencies
        tolerance: Acceptable error
        
    Returns:
        is_valid: Boolean
        error: Absolute error
    """
    R_total = sum(Rs)
    T_total = sum(Ts)
    total = R_total + T_total
    error = abs(total - 1.0)
    is_valid = error < tolerance
    
    return is_valid, error.item()