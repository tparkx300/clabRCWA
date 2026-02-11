import torch


class Environment:
    """Manages lattice parameters, harmonics, and layer definitions."""
    
    def __init__(self, Lambda_x, Lambda_y, n_harmonics, resolution, device, dtype=torch.complex64):
        """
        Initialize the RCWA environment.
        
        Args:
            Lambda_x: Lattice period in x direction
            Lambda_y: Lattice period in y direction
            n_harmonics: Number of harmonics (must be odd)
            resolution: Grid resolution for geometries
            device: torch device
            dtype: Complex dtype (torch.complex64 or torch.complex128)
        """
        self.device = device
        self.dtype = dtype
        self.floatdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        
        self.Lambda_x = Lambda_x
        self.Lambda_y = Lambda_y
        self.n_harmonics = n_harmonics
        self.resolution = resolution
        
        # Harmonic indices
        self.harmonics_range = torch.arange(
            -(n_harmonics - 1) / 2, 
            (n_harmonics + 1) / 2, 
            1, 
            dtype=torch.int, 
            device=device
        )
        
        # Identity matrices
        self.I0 = torch.ones(n_harmonics**2, dtype=dtype, device=device)
        self.I1 = torch.eye(n_harmonics**2, dtype=dtype, device=device)
        self.I2 = torch.eye(2 * n_harmonics**2, dtype=dtype, device=device)
        
         # Layer data storage
        self.epsilon_rs = None
        self.mu_rs = None
        self.Ls = None
        self.layer_geometries = None
        self.n_layers = 0
        
        # Builder API storage
        self._ref_layer = None
        self._middle_layers = []
        self._trn_layer = None
        
    def set_ref_layer(self, epsilon_r, mu_r=1.0):
        """
        Set reflection region (always homogeneous, semi-infinite).
        
        Args:
            epsilon_r: Permittivity (scalar, homogeneous)
            mu_r: Permeability (default: 1.0)
        """
        self._ref_layer = {
            'epsilon_r': [epsilon_r, epsilon_r],
            'mu_r': [mu_r, mu_r],
            'L': 0.0,
            'geometry': 'homogeneous'
        }
        
    def add_layer(self, thickness, epsilon_material, epsilon_background=None, 
                  geometry_pattern=None, mu_material=1.0, mu_background=1.0):
        """
        Add a layer (patterned or homogeneous).
        
        Args:
            thickness: Layer thickness
            epsilon_material: Permittivity where geometry=1
            epsilon_background: Permittivity where geometry=0 (if None, homogeneous layer)
            geometry_pattern: Geometry tensor or None for homogeneous
            mu_material: Permeability where geometry=1 (default: 1.0)
            mu_background: Permeability where geometry=0 (default: 1.0)
            
        Examples:
            # Homogeneous layer
            env.add_layer(0.5, epsilon_material=2.25)
            
            # Patterned layer
            pattern = geometry.circle(resolution, period, radius, device=device)
            env.add_layer(0.5, epsilon_material=2.25, epsilon_background=1.0, 
                         geometry_pattern=pattern)
        """
        if epsilon_background is None:
            epsilon_background = epsilon_material
        
        if geometry_pattern is None:
            geometry_pattern = 'homogeneous'
            
        self._middle_layers.append({
            'epsilon_r': [epsilon_material, epsilon_background],
            'mu_r': [mu_material, mu_background],
            'L': thickness,
            'geometry': geometry_pattern
        })
        
    def set_trn_layer(self, epsilon_r, mu_r=1.0):
        """
        Set transmission region (always homogeneous, semi-infinite).
        
        Args:
            epsilon_r: Permittivity (scalar, homogeneous)
            mu_r: Permeability (default: 1.0)
        """
        self._trn_layer = {
            'epsilon_r': [epsilon_r, epsilon_r],
            'mu_r': [mu_r, mu_r],
            'L': 0.0,
            'geometry': 'homogeneous'
        }
        
    def build(self):
        """
        Build the final layer structure.
        
        Call this after setting ref, middle layers, and trn layers.
        """
        assert self._ref_layer is not None, "Must call set_ref_layer() first"
        assert self._trn_layer is not None, "Must call set_trn_layer() first"
        
        all_layers = [self._ref_layer] + self._middle_layers + [self._trn_layer]
        
        epsilon_rs = []
        mu_rs = []
        Ls = []
        geometries = []
        
        # Import geometry module
        from . import geometry as geom
        
        for layer in all_layers:
            epsilon_rs.append(layer['epsilon_r'])
            mu_rs.append(layer['mu_r'])
            Ls.append(layer['L'])
            
            # Create geometry tensor
            if layer['geometry'] == 'homogeneous':
                geometries.append(geom.homogeneous(self.resolution, device=self.device, dtype=self.floatdtype))
            else:
                geometries.append(layer['geometry'])
        
        # Stack and set layers using old method
        layer_geometries = torch.stack(geometries)
        self.set_layers(epsilon_rs, mu_rs, Ls, layer_geometries)
        
    def set_layers(self, epsilon_rs, mu_rs, Ls, layer_geometries):
        """
        Set layer materials and geometries.
        
        Args:
            epsilon_rs: List of permittivity values [epsilon_r1, epsilon_r2] for each layer
            mu_rs: List of permeability values [mu_r1, mu_r2] for each layer
            Ls: List of layer thicknesses (first and last must be 0.0)
            layer_geometries: Tensor of geometry patterns for each layer
        """
        assert len(epsilon_rs) >= 2, "Need at least 2 layers (reflection and transmission)"
        assert len(epsilon_rs) == len(mu_rs) == len(Ls), "Mismatched layer counts"
        assert Ls[0] == 0.0 and Ls[-1] == 0.0, "First and last layers must have L=0 (semi-infinite)"
        
        # Convert to tensors
        epsilon_rs = torch.as_tensor(epsilon_rs, device=self.device)
        mu_rs = torch.as_tensor(mu_rs, device=self.device)
        
        # Handle complex vs real values
        if torch.is_complex(epsilon_rs) or torch.is_complex(mu_rs):
            self.epsilon_rs = epsilon_rs.to(self.dtype)
            self.mu_rs = mu_rs.to(self.dtype)
        else:
            self.epsilon_rs = epsilon_rs.to(self.floatdtype)
            self.mu_rs = mu_rs.to(self.floatdtype)
        
        self.Ls = torch.as_tensor(Ls, dtype=self.dtype, device=self.device)
        self.layer_geometries = torch.as_tensor(layer_geometries, dtype=self.floatdtype, device=self.device)
        self.n_layers = self.epsilon_rs.shape[0]
        
    def get_layer(self, i):
        """Get parameters for layer i."""
        return {
            'epsilon_r': self.epsilon_rs[i],
            'mu_r': self.mu_rs[i],
            'L': self.Ls[i],
            'geometry': self.layer_geometries[i]
        }