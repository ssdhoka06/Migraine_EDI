"""
Mamba State-Space Model - Pure PyTorch Implementation
======================================================
This is a pure PyTorch implementation of the Mamba architecture
that works on Apple Silicon (M3) via MPS backend.

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
Paper: https://arxiv.org/abs/2312.00752

Author: Dhoka
Date: December 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba.
    
    This implements the selective scan mechanism that makes Mamba
    content-aware, unlike traditional SSMs.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Compute dt_rank
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Input projection: projects input to inner dimension
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 1D Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )
        
        # SSM Parameters projection
        # Projects to: dt (delta), B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias for stability
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # dt bias initialization (log scale for positivity)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter (diagonal, learned in log space for stability)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def _selective_scan(
        self,
        x: torch.Tensor,  # (batch, d_inner, seq_len)
        delta: torch.Tensor,  # (batch, d_inner, seq_len)
        A: torch.Tensor,  # (d_inner, d_state)
        B: torch.Tensor,  # (batch, d_state, seq_len)
        C: torch.Tensor,  # (batch, d_state, seq_len)
        D: torch.Tensor,  # (d_inner,)
    ) -> torch.Tensor:
        """
        Selective scan algorithm - the heart of Mamba.
        
        This is a pure PyTorch implementation that works on all backends.
        For production on CUDA, you'd use the optimized CUDA kernel.
        """
        batch, d_inner, seq_len = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B using delta
        # deltaA = exp(delta * A)
        deltaA = torch.exp(
            rearrange(delta, "b d l -> b d l 1") * 
            rearrange(A, "d n -> 1 d 1 n")
        )  # (batch, d_inner, seq_len, d_state)
        
        # deltaB = delta * B
        deltaB_x = (
            rearrange(delta, "b d l -> b d l 1") *
            rearrange(B, "b n l -> b 1 l n") *
            rearrange(x, "b d l -> b d l 1")
        )  # (batch, d_inner, seq_len, d_state)
        
        # Scan (recurrence)
        # h[t] = deltaA[t] * h[t-1] + deltaB[t] * x[t]
        # y[t] = C[t] @ h[t] + D * x[t]
        
        h = torch.zeros(
            (batch, d_inner, d_state),
            device=x.device,
            dtype=x.dtype
        )
        
        ys = []
        for t in range(seq_len):
            h = deltaA[:, :, t] * h + deltaB_x[:, :, t]
            y = torch.einsum("bdn,bn->bd", h, C[:, :, t])
            ys.append(y)
        
        y = torch.stack(ys, dim=2)  # (batch, d_inner, seq_len)
        
        # Add skip connection
        y = y + D.unsqueeze(0).unsqueeze(-1) * x
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mamba block.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection (split into x and z paths)
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Transpose for conv1d: (batch, d_inner, seq_len)
        x = rearrange(x, "b l d -> b d l")
        
        # 1D convolution for local context
        x = self.conv1d(x)[:, :, :seq_len]  # Trim to original length
        
        # Activation
        x = F.silu(x)
        
        # SSM parameters from x
        x_rearranged = rearrange(x, "b d l -> b l d")
        x_proj = self.x_proj(x_rearranged)  # (batch, seq_len, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(
            x_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # Compute delta (dt)
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Ensure positivity
        dt = rearrange(dt, "b l d -> b d l")
        
        # Transpose B and C
        B = rearrange(B, "b l n -> b n l")
        C = rearrange(C, "b l n -> b n l")
        
        # Get A from log space
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective scan
        y = self._selective_scan(x, dt, A, B, C, self.D)
        
        # Transpose back
        y = rearrange(y, "b d l -> b l d")
        
        # Gate with z path
        z = F.silu(z)
        y = y * z
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class MambaBlock(nn.Module):
    """
    A single Mamba block with residual connection and normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm = RMSNorm(d_model)
        self.mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm residual connection
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaBackbone(nn.Module):
    """
    Stack of Mamba blocks forming the backbone.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        return self.norm_f(x)


if __name__ == "__main__":
    # Test the implementation
    print("Testing Mamba implementation on available device...")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Test parameters
    batch_size = 4
    seq_len = 14  # 14 days
    d_model = 64
    
    # Create model
    model = MambaBackbone(
        d_model=d_model,
        n_layers=2,
        d_state=16,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("\n✓ Mamba backbone test passed!")