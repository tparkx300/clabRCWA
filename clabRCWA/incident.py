import torch
import math
import numpy as np


class Incident:
    """Manages incident light properties: wavelength, angle, polarization, and k-vectors."""
    
    def __init__(self, lamb0, Lambda_x, Lambda_y, n_harmonics, theta, phi, amp_TE, 
                 epsilon_ref, mu_ref, epsilon_trn, mu_trn, device, dtype=torch.complex64):
        """
        Initialize incident light.
        
        Args:
            lamb0: Free space wavelength
            Lambda_x: Lattice period in x direction
            Lambda_y: Lattice period in y direction
            n_harmonics: Number of harmonics
            theta: Polar angle in degrees
            phi: Azimuthal angle in degrees
            amp_TE: TE polarization amplitude (-1 to 1)
            epsilon_ref: Permittivity in reflection region
            mu_ref: Permeability in reflection region
            epsilon_trn: Permittivity in transmission region
            mu_trn: Permeability in transmission region
            device: torch device
            dtype: Complex dtype
        """
        assert abs(amp_TE) <= 1.0, "amp_TE must be normalized (-1 to 1)"
        
        self.device = device
        self.dtype = dtype
        self.floatdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        
        self.lamb0 = lamb0
        self.k0 = 2 * math.pi / lamb0
        self.Lambda_x = Lambda_x
        self.Lambda_y = Lambda_y
        self.n_harmonics = n_harmonics
        
        self.theta = theta * math.pi / 180
        self.phi = phi * math.pi / 180
        self.amp_TE = amp_TE
        self.amp_TM = math.copysign(math.sqrt(1 - amp_TE**2), amp_TE) if amp_TE != 0 else math.sqrt(1 - amp_TE**2)
        
        # Harmonic indices
        self.harmonics_range = torch.arange(
            -(n_harmonics - 1) / 2, 
            (n_harmonics + 1) / 2, 
            1, 
            dtype=torch.int, 
            device=device
        )
        
        # Identity matrix
        self.I0 = torch.ones(n_harmonics**2, dtype=dtype, device=device)
        
        # Incident medium properties
        self.epsilon_ref = torch.tensor(epsilon_ref)
        self.mu_ref = torch.tensor(mu_ref)
        self.n_inc = torch.sqrt(torch.tensor(epsilon_ref * mu_ref))

        # Transmission region medium properties
        epsilon_trn = torch.tensor(epsilon_trn)
        mu_trn = torch.tensor(mu_trn)
        
        # Incident k-vector
        self.k_inc = self.k0 * self.n_inc * torch.tensor([
            math.sin(self.theta) * math.cos(self.phi),
            math.sin(self.theta) * math.sin(self.phi),
            math.cos(self.theta)
        ], dtype=self.n_inc.dtype, device=self.device)
        
        self.kx = self.k_inc[0]
        self.ky = self.k_inc[1]
        self.kz = self.k_inc[2]
        self.kx_norm = self.kx / self.k0
        self.ky_norm = self.ky / self.k0
        self.kz_norm = self.kz / self.k0
        
        # Setup polarization vectors
        self._setup_polarization()
        
        # Setup k-vector matrices
        self._setup_k_matrices(epsilon_trn, mu_trn)
        
    def _setup_polarization(self):
        """Setup TE and TM polarization unit vectors."""
        n_norm = torch.tensor([0, 0, 1], dtype=self.n_inc.dtype, device=self.device)
        
        if self.theta == 0.:
            # Normal incidence: Define TE as x-polarized light
            self.e_TE = torch.tensor([1, 0, 0], dtype=self.k_inc.dtype, device=self.device)
        else:
            # Oblique incidence: TE perpendicular to plane of incidence
            # Manual cross product (torch.linalg.cross doesn't support complex tensors): k_inc × n_norm
            self.e_TE = torch.tensor([
                self.k_inc[1] * n_norm[2] - self.k_inc[2] * n_norm[1],
                self.k_inc[2] * n_norm[0] - self.k_inc[0] * n_norm[2],
                self.k_inc[0] * n_norm[1] - self.k_inc[1] * n_norm[0]
            ], dtype=self.k_inc.dtype, device=self.device)
            # Normalize
            norm_TE = torch.sqrt(torch.vdot(self.e_TE, self.e_TE).real)
            self.e_TE = self.e_TE / norm_TE
        
        # TM in plane of incidence
        # Manual cross product: k_inc × e_TE
        self.e_TM = torch.tensor([
            self.k_inc[1] * self.e_TE[2] - self.k_inc[2] * self.e_TE[1],
            self.k_inc[2] * self.e_TE[0] - self.k_inc[0] * self.e_TE[2],
            self.k_inc[0] * self.e_TE[1] - self.k_inc[1] * self.e_TE[0]
        ], dtype=self.e_TE.dtype, device=self.device)
        # Normalize
        norm_TM = torch.sqrt(torch.vdot(self.e_TM, self.e_TM).real)
        self.e_TM = self.e_TM / norm_TM
        
        # Combined polarization vector
        self.Polarization = self.amp_TE * self.e_TE + self.amp_TM * self.e_TM
        
    def _setup_k_matrices(self, epsilon_trn, mu_trn):
        """Setup k-vector matrices for all harmonics."""
        Kx = torch.zeros(self.n_harmonics**2, dtype=self.epsilon_ref.dtype, device=self.device)
        Ky = torch.zeros(self.n_harmonics**2, dtype=self.epsilon_ref.dtype, device=self.device)
        
        m, n = torch.meshgrid(self.harmonics_range, self.harmonics_range, indexing='ij')
        m = m.to(torch.int64).reshape([-1])
        n = n.to(torch.int64).reshape([-1])
        
        n_half = int((self.n_harmonics - 1) / 2)
        
        Kx[(m + n_half) * self.n_harmonics + n + n_half] = (
            (self.kx - 2 * math.pi * n / self.Lambda_x).to(self.epsilon_ref.dtype) / self.k0
        )
        Ky[(m + n_half) * self.n_harmonics + n + n_half] = (
            (self.ky - 2 * math.pi * m / self.Lambda_y).to(self.epsilon_ref.dtype) / self.k0
        )
        
        # Reflection and transmission kz
        Kz_norm_ref = -torch.conj(torch.sqrt(
            (torch.conj(self.epsilon_ref) * torch.conj(self.mu_ref) * self.I0 - Kx**2 - Ky**2).to(self.dtype)
        ))
        Kz_norm_trn = torch.conj(torch.sqrt(
            (torch.conj(epsilon_trn) * torch.conj(mu_trn) * self.I0 - Kx**2 - Ky**2).to(self.dtype)
        ))
        
        self.Kx = torch.diag(Kx)
        self.Ky = torch.diag(Ky)
        self.Kz_norm_ref = torch.diag(Kz_norm_ref)
        self.Kz_norm_trn = torch.diag(Kz_norm_trn)
        
    def set_polarization(self, amp_TE):
        """
        Update polarization amplitudes.
        
        Args:
            amp_TE: TE polarization amplitude (-1 to 1)
        """
        self.amp_TE = amp_TE
        # Preserve sign: if amp_TE is negative, amp_TM should also be negative
        self.amp_TM = math.copysign(math.sqrt(1 - amp_TE**2), amp_TE) if amp_TE != 0 else math.sqrt(1 - amp_TE**2)
        self.Polarization = self.amp_TE * self.e_TE + self.amp_TM * self.e_TM
        
    def get_polarization_vectors(self, nx, ny, forward=True):
        """
        Get TE and TM polarization unit vectors for a specific diffraction order.
        Supports complex vectors for circular polarization.
        
        Args:
            nx, ny: Diffraction order indices
            forward: True for transmission, False for reflection
            
        Returns:
            e_TE, e_TM: Polarization unit vectors (may be complex)
        """
        theta, phi = self.angle_by_order(nx, ny, forward)
        theta = theta * math.pi / 180.
        phi = phi * math.pi / 180.
        
        if forward:
            diffracted_k = torch.tensor([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta)
            ], dtype=self.n_inc.dtype, device=self.device)
        else:
            diffracted_k = torch.tensor([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                -math.cos(theta)
            ], dtype=self.n_inc.dtype, device=self.device)
        
        n_norm = torch.tensor([0, 0, 1], dtype=self.n_inc.dtype, device=self.device)
        
        # Manual cross product: diffracted_k × n_norm
        e_TE = torch.tensor([
            diffracted_k[1] * n_norm[2] - diffracted_k[2] * n_norm[1],
            diffracted_k[2] * n_norm[0] - diffracted_k[0] * n_norm[2],
            diffracted_k[0] * n_norm[1] - diffracted_k[1] * n_norm[0]
        ], dtype=diffracted_k.dtype, device=self.device)
        
        if torch.sqrt(torch.vdot(e_TE, e_TE).real).item() == 0. or theta == 0:
            # Normal incidence: Define TE as x-polarized light
            e_TE = torch.tensor([1, 0, 0], dtype=self.k_inc.dtype, device=self.device)
        
        # Normalize using complex inner product
        norm_TE = torch.sqrt(torch.vdot(e_TE, e_TE).real)
        e_TE = e_TE / norm_TE
        
        # Manual cross product: diffracted_k × e_TE
        e_TM = torch.tensor([
            diffracted_k[1] * e_TE[2] - diffracted_k[2] * e_TE[1],
            diffracted_k[2] * e_TE[0] - diffracted_k[0] * e_TE[2],
            diffracted_k[0] * e_TE[1] - diffracted_k[1] * e_TE[0]
        ], dtype=e_TE.dtype, device=self.device)
        
        norm_TM = torch.sqrt(torch.vdot(e_TM, e_TM).real)
        e_TM = e_TM / norm_TM
        
        return e_TE, e_TM
    
    def update_wavelength(self, lamb0, epsilon_trn, mu_trn=1.0):
        """
        Update wavelength and recalculate k-vectors without recreating object.
        
        Args:
            lamb0: New free space wavelength
            epsilon_trn: Permittivity in transmission region
            mu_trn: Permeability in transmission region (default: 1.0)
        """
        self.lamb0 = lamb0
        self.k0 = 2 * math.pi / lamb0
        
        # Recalculate incident k-vector
        self.k_inc = self.k0 * self.n_inc * torch.tensor([
            math.sin(self.theta) * math.cos(self.phi),
            math.sin(self.theta) * math.sin(self.phi),
            math.cos(self.theta)
        ], dtype=self.n_inc.dtype, device=self.device)
        
        self.kx = self.k_inc[0]
        self.ky = self.k_inc[1]
        self.kz = self.k_inc[2]
        self.kx_norm = self.kx / self.k0
        self.ky_norm = self.ky / self.k0
        self.kz_norm = self.kz / self.k0
        
        # Recalculate k-matrices with new wavelength
        self._setup_k_matrices(torch.tensor(epsilon_trn), torch.tensor(mu_trn))
        
    def angle_by_order(self, nx, ny, forward=True):
        """
        Calculate diffraction angle for a specific order.
        
        Args:
            nx, ny: Diffraction order indices
            forward: True for transmission, False for reflection
            
        Returns:
            theta, phi: Angles in degrees
        """
        n_half = int((self.n_harmonics - 1) / 2)
        index = (-ny + n_half) * self.n_harmonics - nx + n_half
        
        Kx = torch.diag(self.Kx)
        Ky = torch.diag(self.Ky)
        kx = Kx[index]
        ky = Ky[index]
        
        if forward:
            Kz = torch.diag(self.Kz_norm_trn)
        else:
            Kz = -torch.diag(self.Kz_norm_ref)
        kz = Kz[index].real
        
        diffracted_phi = torch.atan2(ky, kx) * 180 / math.pi
        diffracted_theta = torch.atan2(torch.sqrt(kx**2 + ky**2), kz) * 180 / math.pi
        
        if diffracted_theta == 0.:
            diffracted_phi = self.phi * 180 / math.pi
            
        return diffracted_theta, diffracted_phi