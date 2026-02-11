import torch
import torch.linalg as la
from .environment import Environment
from .incident import Incident
from .utils import compute_diffraction_efficiency, extract_field_by_order


class RCWA:
    """
    RCWA (Rigorous Coupled-Wave Analysis) solver.
    
    Takes Environment (structure) and Incident (light) as inputs and performs simulation.
    """
    
    def __init__(self, environment, incident):
        """
        Initialize RCWA solver.
        
        Args:
            environment: Environment object (defines structure/layers)
            incident: Incident object (defines light properties)
        """
        self.env = environment
        self.inc = incident
        self.device = environment.device
        self.dtype = environment.dtype
        
        # Convolution storage
        self.epsilon_Cs = []
        self.mu_Cs = []
        
        # Solver storage
        self.Ws = []
        self.Vs = []
        self.Ps = []
        self.Qs = []
        self.Ss = []
        self.S_global = None
        self.W_ref = None
        self.W_trn = None
        
        # Results storage
        self.rs = None
        self.ts = None
        self.Rs = None
        self.Ts = None
    
    # ========== Convolution Methods ==========
    
    def _compute_convmat(self, img_tensor, epsilon_r, mu_r):
        """
        Compute convolution matrices for permittivity and permeability.
        
        Args:
            img_tensor: Binary pattern (1s and 0s) defining geometry
            epsilon_r: [epsilon_r_material, epsilon_r_background]
            mu_r: [mu_r_material, mu_r_background]
            
        Returns:
            epsilon_C, mu_C: Convolution matrices
        """
        # Homogeneous case - return scalars
        if epsilon_r[0] == epsilon_r[1] and mu_r[0] == mu_r[1]:
            return epsilon_r[0], mu_r[0]
        
        # Create material distribution
        epsilon_r_dist = img_tensor * (epsilon_r[0] - epsilon_r[1]) + epsilon_r[1]
        mu_r_dist = img_tensor * (mu_r[0] - mu_r[1]) + mu_r[1]
        
        # Fourier transform
        epsilon_ft = torch.fft.fft2(epsilon_r_dist)
        epsilon_ft = epsilon_ft / (self.env.resolution**2)  # Normalization
        epsilon_ft = epsilon_ft.to(self.dtype)
        
        mu_ft = torch.fft.fft2(mu_r_dist)
        mu_ft = mu_ft / (self.env.resolution**2)  # Normalization
        mu_ft = mu_ft.to(self.dtype)
        
        # Build convolution matrices
        harmonics_range = self.env.harmonics_range
        a, b = torch.meshgrid(harmonics_range, harmonics_range, indexing='ij')
        a = a.to(torch.int64).reshape([-1])
        b = b.to(torch.int64).reshape([-1])
        
        n_harmonics_sq = self.env.n_harmonics**2
        ind = torch.arange(n_harmonics_sq, device=self.device)
        a_ind, b_ind = torch.meshgrid(ind.to(torch.int64), ind.to(torch.int64), indexing='ij')
        
        epsilon_C = epsilon_ft[a[a_ind] - a[b_ind], b[a_ind] - b[b_ind]]
        mu_C = mu_ft[a[a_ind] - a[b_ind], b[a_ind] - b[b_ind]]
        
        # Store for later reference
        self.epsilon_Cs.append(epsilon_C)
        self.mu_Cs.append(mu_C)
        
        return epsilon_C, mu_C
    
    # ========== Solver Methods ==========
    
    def _solve_eigenmodes(self, epsilon_C, mu_C):
        """
        Solve eigenvalue problem to get W, V, and eigenvalues.
        
        Args:
            epsilon_C: Permittivity convolution matrix
            mu_C: Permeability convolution matrix
            
        Returns:
            W, V, lamb: Eigenvector matrix, V matrix, eigenvalues
        """
        Kx = self.inc.Kx.to(self.dtype)
        Ky = self.inc.Ky.to(self.dtype)
        I1 = self.env.I1.to(epsilon_C.dtype if epsilon_C.shape != torch.Size([]) else self.dtype)
        I2 = self.env.I2
        
        # Homogeneous layer (scalar case)
        if epsilon_C.shape == torch.Size([]) and mu_C.shape == torch.Size([]):
            epsilon_C_inv = epsilon_C**-1
            
            P11 = epsilon_C_inv * torch.matmul(Kx, Ky)
            P12 = mu_C * I1 - epsilon_C_inv * torch.matmul(Kx, Kx)
            P21 = epsilon_C_inv * torch.matmul(Ky, Ky) - mu_C * I1
            P22 = -epsilon_C_inv * torch.matmul(Ky, Kx)
            
            P1 = torch.cat((P11, P12), axis=1)
            P2 = torch.cat((P21, P22), axis=1)
            P = torch.cat((P1, P2), axis=0)
            
            Q = (epsilon_C / mu_C * P).to(self.dtype)
            
            W = I2
            
            Kx_diag = torch.diag(Kx)
            Ky_diag = torch.diag(Ky)
            Kz = torch.conj(torch.sqrt(
                (torch.conj(epsilon_C) * torch.conj(mu_C) - Kx_diag**2 - Ky_diag**2).to(self.dtype)
            ))
            jKz = Kz * 1j
            lamb = torch.cat((jKz, jKz), axis=0)
            lamb_mat = torch.diag(lamb)
            
        # Inhomogeneous layer (matrix case)
        else:
            epsilon_C_inv = la.inv(epsilon_C)
            mu_C_inv = la.inv(mu_C)
            
            P11 = torch.matmul(Kx, torch.matmul(epsilon_C_inv, Ky))
            P12 = mu_C - torch.matmul(Kx, torch.matmul(epsilon_C_inv, Kx))
            P21 = torch.matmul(Ky, torch.matmul(epsilon_C_inv, Ky)) - mu_C
            P22 = -torch.matmul(Ky, torch.matmul(epsilon_C_inv, Kx))
            
            P1 = torch.cat((P11, P12), axis=1)
            P2 = torch.cat((P21, P22), axis=1)
            P = torch.cat((P1, P2), axis=0)
            
            Q11 = torch.matmul(Kx, torch.matmul(mu_C_inv, Ky))
            Q12 = epsilon_C - torch.matmul(Kx, torch.matmul(mu_C_inv, Kx))
            Q21 = torch.matmul(Ky, torch.matmul(mu_C_inv, Ky)) - epsilon_C
            Q22 = -torch.matmul(Ky, torch.matmul(mu_C_inv, Kx))
            
            Q1 = torch.cat((Q11, Q12), axis=1)
            Q2 = torch.cat((Q21, Q22), axis=1)
            Q = torch.cat((Q1, Q2), axis=0)
            
            Omega_sq = torch.matmul(P, Q)
            
            lamb_sq, W = la.eig(Omega_sq)
            lamb = torch.sqrt(lamb_sq)
            lamb_mat = torch.diag(lamb)
        
        lamb_mat_inv = la.inv(lamb_mat)
        V = torch.matmul(torch.matmul(Q, W), lamb_mat_inv)
        
        # Store for reference
        self.Ws.append(W)
        self.Vs.append(V)
        self.Ps.append(P)
        self.Qs.append(Q)
        
        return W, V, lamb
    
    def _compute_layer_smatrix(self, W0, V0, Wi, Vi, Li, lamb_i, layer='layer'):
        """
        Compute S-matrix for a single layer.
        
        Args:
            W0, V0: Free space eigenmodes
            Wi, Vi: Layer eigenmodes
            Li: Layer thickness
            lamb_i: Layer eigenvalues
            layer: 'layer', 'reflection', or 'transmission'
            
        Returns:
            [S11, S12, S21, S22]: S-matrix components
        """
        if layer == 'layer':
            Wi_inv = la.inv(Wi)
            Vi_inv = la.inv(Vi)
            
            Ai = torch.matmul(Wi_inv, W0) + torch.matmul(Vi_inv, V0)
            Bi = torch.matmul(Wi_inv, W0) - torch.matmul(Vi_inv, V0)
            Xi = torch.diag(torch.exp(-lamb_i * self.inc.k0 * Li))
            
            Ai_inv = la.inv(Ai)
            BiAi_inv = torch.matmul(Bi, Ai_inv)
            XiBiAi_invXi = torch.matmul(torch.matmul(Xi, BiAi_inv), Xi)
            Inv = la.inv(Ai - torch.matmul(XiBiAi_invXi, Bi))
            
            S11 = torch.matmul(Inv, torch.matmul(XiBiAi_invXi, Ai) - Bi)
            S12 = torch.matmul(Inv, torch.matmul(Xi, Ai - torch.matmul(BiAi_inv, Bi)))
            S21 = S12
            S22 = S11
            
        else:
            assert layer in ['reflection', 'transmission'], 'Invalid layer type'
            W0_inv = la.inv(W0)
            V0_inv = la.inv(V0)
            
            Ai = torch.matmul(W0_inv, Wi) + torch.matmul(V0_inv, Vi)
            Bi = torch.matmul(W0_inv, Wi) - torch.matmul(V0_inv, Vi)
            
            Ai_inv = la.inv(Ai)
            
            if layer == 'reflection':
                S11 = -torch.matmul(Ai_inv, Bi)
                S12 = 2 * Ai_inv
                S21 = 0.5 * (Ai - torch.matmul(Bi, torch.matmul(Ai_inv, Bi)))
                S22 = torch.matmul(Bi, Ai_inv)
            else:  # transmission
                S11 = torch.matmul(Bi, Ai_inv)
                S12 = 0.5 * (Ai - torch.matmul(Bi, torch.matmul(Ai_inv, Bi)))
                S21 = 2 * Ai_inv
                S22 = -torch.matmul(Ai_inv, Bi)
        
        return [S11, S12, S21, S22]
    
    def _redheffer_star(self, SA, SB):
        """
        Redheffer star product of two S-matrices.
        
        Args:
            SA: First S-matrix [S11, S12, S21, S22]
            SB: Second S-matrix [S11, S12, S21, S22]
            
        Returns:
            Combined S-matrix [S11, S12, S21, S22]
        """
        SA11, SA12, SA21, SA22 = SA
        SB11, SB12, SB21, SB22 = SB
        
        I2 = self.env.I2
        
        I_SB11SA11_inv = la.inv(I2 - torch.matmul(SB11, SA22))
        I_SA22SB11_inv = la.inv(I2 - torch.matmul(SA22, SB11))
        
        D = torch.matmul(SA12, I_SB11SA11_inv)
        F = torch.matmul(SB21, I_SA22SB11_inv)
        
        S11 = SA11 + torch.matmul(D, torch.matmul(SB11, SA21))
        S12 = torch.matmul(D, SB12)
        S21 = torch.matmul(F, SA21)
        S22 = SB22 + torch.matmul(F, torch.matmul(SA22, SB12))
        
        return [S11, S12, S21, S22]
    
    # ========== Main Solve Methods ==========
    
    def set_polarization(self, amp_TE):
        """Update incident polarization (convenience method)."""
        self.inc.set_polarization(amp_TE)
    
    def solve_S_matrix(self):
        """
        Compute the global S-matrix by combining all layers.
        
        Uses the Redheffer star product to combine individual layer S-matrices.
        """
        # Create free space reference
        air_geometry = torch.zeros((self.env.resolution, self.env.resolution), 
                                   dtype=self.env.floatdtype, device=self.device)
        epsilon_r = torch.tensor([1.0, 1.0], dtype=self.dtype, device=self.device)
        mu_r = torch.tensor([1.0, 1.0], dtype=self.dtype, device=self.device)
        
        epsilon_C, mu_C = self._compute_convmat(air_geometry, epsilon_r, mu_r)
        W0, V0, _ = self._solve_eigenmodes(epsilon_C, mu_C)
        
        # Process each layer
        for i in range(self.env.n_layers):
            layer_params = self.env.get_layer(i)
            
            # Determine layer type
            layer_type = 'layer'
            if i == 0:
                layer_type = 'reflection'
            elif i == self.env.n_layers - 1:
                layer_type = 'transmission'
            
            # Compute convolution matrix for this layer
            epsilon_C, mu_C = self._compute_convmat(
                layer_params['geometry'],
                layer_params['epsilon_r'],
                layer_params['mu_r']
            )
            
            # Solve eigenmodes
            Wi, Vi, lamb_i = self._solve_eigenmodes(epsilon_C, mu_C)
            
            # Store boundary eigenmodes
            if layer_type == 'reflection':
                self.W_ref = Wi
            elif layer_type == 'transmission':
                self.W_trn = Wi
            
            # Compute layer S-matrix
            Si = self._compute_layer_smatrix(W0, V0, Wi, Vi, 
                                            layer_params['L'], lamb_i, 
                                            layer=layer_type)
            self.Ss.append(Si)
            
            # Combine with global S-matrix using Redheffer star product
            if layer_type == 'reflection':
                self.S_global = Si
            else:
                self.S_global = self._redheffer_star(self.S_global, Si)
    
    def solve_RT(self):
        """
        Solve for reflection and transmission fields and efficiencies.
        
        Must call solve_S_matrix() first.
        """
        # Setup incident field
        Ex_inc = self.inc.Polarization[0]
        Ey_inc = self.inc.Polarization[1]
        
        delta_col = torch.zeros(self.env.n_harmonics**2, dtype=self.dtype, device=self.device)
        n_half = int((self.env.n_harmonics - 1) / 2)
        delta_col[-self.inc.harmonics_range[0] * self.env.n_harmonics - self.inc.harmonics_range[0]] = 1
        
        s_inc = torch.cat((Ex_inc * delta_col, Ey_inc * delta_col), axis=0)
        c_inc = torch.matmul(la.inv(self.W_ref), s_inc)
        
        # Solve for scattered fields
        c_ref = torch.matmul(self.S_global[0], c_inc)
        c_trn = torch.matmul(self.S_global[2], c_inc)
        
        s_ref = torch.matmul(self.W_ref, c_ref)
        s_trn = torch.matmul(self.W_trn, c_trn)
        
        # Extract field components
        rx = s_ref[:self.env.n_harmonics**2]
        ry = s_ref[self.env.n_harmonics**2:]
        tx = s_trn[:self.env.n_harmonics**2]
        ty = s_trn[self.env.n_harmonics**2:]
        
        Kx = self.inc.Kx.to(self.dtype)
        Ky = self.inc.Ky.to(self.dtype)
        
        rz = -torch.matmul(la.inv(self.inc.Kz_norm_ref), 
                          (torch.matmul(Kx, rx) + torch.matmul(Ky, ry)))
        tz = -torch.matmul(la.inv(self.inc.Kz_norm_trn), 
                          (torch.matmul(Kx, tx) + torch.matmul(Ky, ty)))
        
        self.rs = [rx, ry, rz]
        self.ts = [tx, ty, tz]
        
        # Compute diffraction efficiencies
        self.Rs, self.Ts = compute_diffraction_efficiency(
            self.rs, self.ts,
            self.inc.Kz_norm_ref, self.inc.Kz_norm_trn,
            self.inc.kz_norm,
            self.inc.mu_ref, self.env.mu_rs[-1][0],
            self.env.n_harmonics
        )
    
    def solve(self):
        """
        Complete solve: compute S-matrix and solve for R/T.
        
        Returns:
            Rs, Ts: Reflection and transmission efficiencies
        """
        self.solve_S_matrix()
        self.solve_RT()
        return self.Rs, self.Ts
    
    # ========== Query Methods ==========
    
    def field_by_order(self, nx, ny):
        """
        Get field components for a specific diffraction order.
        
        Args:
            nx, ny: Diffraction order indices
            
        Returns:
            r_field: [rx, ry, rz] reflection field
            t_field: [tx, ty, tz] transmission field
        """
        return extract_field_by_order(
            self.rs, self.ts, nx, ny,
            self.inc.Kz_norm_ref, self.inc.Kz_norm_trn,
            self.inc.kz_norm,
            self.inc.mu_ref, self.env.mu_rs[-1][0],
            self.env.n_harmonics,
            self.device, self.dtype
        )
    
    def angle_by_order(self, nx, ny, forward=True):
        """
        Get diffraction angle for a specific order.
        
        Args:
            nx, ny: Diffraction order indices
            forward: True for transmission, False for reflection
            
        Returns:
            theta, phi: Angles in degrees
        """
        return self.inc.angle_by_order(nx, ny, forward)
    
    def transmission_polarization(self, nx, ny):
        """Get TE/TM polarization vectors for transmitted order (nx, ny)."""
        return self.inc.get_polarization_vectors(nx, ny, forward=True)
    
    def reflection_polarization(self, nx, ny):
        """Get TE/TM polarization vectors for reflected order (nx, ny)."""
        return self.inc.get_polarization_vectors(nx, ny, forward=False)
    
    def get_total_reflection(self):
        """Get total reflection (sum of all orders)."""
        return sum(self.Rs) if self.Rs else None
    
    def get_total_transmission(self):
        """Get total transmission (sum of all orders)."""
        return sum(self.Ts) if self.Ts else None
    
    def clear_cache(self):
        """Clear all cached computation results."""
        self.epsilon_Cs = []
        self.mu_Cs = []
        self.Ws = []
        self.Vs = []
        self.Ps = []
        self.Qs = []
        self.Ss = []
        self.S_global = None
        self.W_ref = None
        self.W_trn = None
        self.rs = None
        self.ts = None
        self.Rs = None
        self.Ts = None