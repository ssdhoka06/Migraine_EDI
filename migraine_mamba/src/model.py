"""
MigraineMamba - Complete Model Architecture (Phase 2.3)
========================================================
Clinically-interpretable Mamba model for migraine prediction.

Architecture Overview:
1. Feature Embedding - Groups features clinically
2. Positional Encoding - Learnable day positions
3. Mamba Backbone - Temporal pattern learning (SSM)
4. Clinical Attention - 4-head interpretable attention
5. Temporal Aggregation - Recency-weighted pooling
6. Clinical Knowledge Integration - Engineered features
7. Prediction Heads - Attack probability, severity, triggers

Target: ~800K parameters, trainable on M3 MacBook Air

Author: Dhoka
Date: December 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Import from local mamba_ssm module
from mamba_ssm import MambaBackbone, RMSNorm


@dataclass
class MigraineModelConfig:
    """Configuration for MigraineMamba model."""
    
    # Input dimensions (matching synthetic data)
    n_continuous_features: int = 8   # sleep_hours, stress_level, pressure, etc.
    n_binary_features: int = 6       # had_breakfast, bright_light, etc.
    seq_len: int = 14                # 14-day lookback window
    
    # Embedding dimensions
    d_model: int = 64                # Main hidden dimension
    d_embed_continuous: int = 32     # Continuous feature embedding
    d_embed_binary: int = 16         # Binary feature embedding
    
    # Mamba configuration
    n_mamba_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    
    # Attention configuration
    n_attention_heads: int = 4
    
    # Clinical knowledge features
    n_clinical_features: int = 8     # Engineered features
    
    # Regularization
    dropout: float = 0.3
    
    # Output
    n_trigger_classes: int = 7       # Number of trigger types


class FeatureEmbedding(nn.Module):
    """
    Layer 1: Feature Embedding
    
    Groups features clinically and embeds each group separately.
    Maintains semantic meaning for interpretation.
    """
    
    def __init__(self, config: MigraineModelConfig):
        super().__init__()
        self.config = config
        
        # Continuous features: sleep_hours, stress_level, pressure, etc.
        self.continuous_embed = nn.Sequential(
            nn.Linear(config.n_continuous_features, config.d_embed_continuous),
            nn.LayerNorm(config.d_embed_continuous),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
        )
        
        # Binary features: had_breakfast, bright_light, etc.
        self.binary_embed = nn.Sequential(
            nn.Linear(config.n_binary_features, config.d_embed_binary),
            nn.LayerNorm(config.d_embed_binary),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
        )
        
        # Menstrual cycle embedding (0-27 + special tokens)
        self.menstrual_embed = nn.Embedding(
            num_embeddings=30,  # 0-27 + padding + male indicator
            embedding_dim=8,
            padding_idx=29,
        )
        
        # Day of week embedding
        self.dow_embed = nn.Embedding(
            num_embeddings=7,
            embedding_dim=8,
        )
        
        # Total embedding dimension
        self.total_embed_dim = (
            config.d_embed_continuous + 
            config.d_embed_binary + 
            8 + 8  # menstrual + dow
        )
        
        # Project to model dimension
        self.projection = nn.Linear(self.total_embed_dim, config.d_model)
    
    def forward(
        self,
        continuous_features: torch.Tensor,
        binary_features: torch.Tensor,
        menstrual_day: torch.Tensor,
        day_of_week: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            continuous_features: (batch, seq_len, n_continuous)
            binary_features: (batch, seq_len, n_binary)
            menstrual_day: (batch, seq_len)
            day_of_week: (batch, seq_len)
            
        Returns:
            embedded: (batch, seq_len, d_model)
        """
        # Embed each feature group
        cont_emb = self.continuous_embed(continuous_features)
        bin_emb = self.binary_embed(binary_features)
        
        # Handle menstrual day (-1 for males → use index 29)
        menstrual_idx = menstrual_day.clone()
        menstrual_idx[menstrual_idx < 0] = 29
        menstrual_idx = menstrual_idx.clamp(0, 29)
        menstrual_emb = self.menstrual_embed(menstrual_idx.long())
        
        # Day of week
        dow_emb = self.dow_embed(day_of_week.long())
        
        # Concatenate all
        combined = torch.cat([cont_emb, bin_emb, menstrual_emb, dow_emb], dim=-1)
        
        return self.projection(combined)


class PositionalEncoding(nn.Module):
    """
    Layer 2: Learnable Positional Encoding
    
    Adds learnable position embeddings for days 1-14.
    """
    
    def __init__(self, config: MigraineModelConfig):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.seq_len, config.d_model) * 0.02
        )
        self.dropout = nn.Dropout(config.dropout * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        return self.dropout(x)


class ClinicalAttention(nn.Module):
    """
    Layer 4: Clinical Attention Mechanism
    
    Multi-head attention with clinical interpretation:
    - Head 1: Sleep-attack relationships
    - Head 2: Weather-attack relationships
    - Head 3: Stress-attack relationships
    - Head 4: Hormonal-attack relationships
    """
    
    def __init__(self, config: MigraineModelConfig):
        super().__init__()
        
        self.n_heads = config.n_attention_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_attention_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        self.attention_weights = attention.detach()
        
        # Apply attention
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention
        return output, None


class TemporalAggregation(nn.Module):
    """
    Layer 5: Temporal Aggregation
    
    Combines 14-day sequence into single summary using
    recency-weighted pooling (recent days matter more).
    """
    
    def __init__(self, config: MigraineModelConfig):
        super().__init__()
        
        # Initialize with recency bias: exp(-0.1 * (14 - day))
        initial_weights = torch.exp(
            -0.1 * torch.arange(config.seq_len, 0, -1).float()
        )
        self.agg_weights = nn.Parameter(initial_weights)
        
        # Attention-based aggregation query
        self.attention_proj = nn.Linear(config.d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            aggregated: (batch, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Recency-weighted mean
        weights = F.softmax(self.agg_weights[:seq_len], dim=0)
        weighted_sum = torch.einsum("bld,l->bd", x, weights)
        
        # Attention-based aggregation
        attn_scores = self.attention_proj(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attention_agg = torch.einsum("bld,bl->bd", x, attn_weights)
        
        # Combine both
        return 0.5 * weighted_sum + 0.5 * attention_agg


class ClinicalKnowledgeIntegration(nn.Module):
    """
    Layer 6: Clinical Knowledge Integration
    
    Adds engineered features based on medical knowledge:
    - Refractory period flag
    - Menstrual high-risk phase
    - Attack frequency trend
    - Trigger accumulation score
    """
    
    def __init__(self, config: MigraineModelConfig):
        super().__init__()
        
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.n_clinical_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        
        self.combine = nn.Linear(config.d_model + 32, config.d_model)
    
    def forward(
        self,
        aggregated: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            aggregated: (batch, d_model)
            clinical_features: (batch, n_clinical)
        Returns:
            combined: (batch, d_model)
        """
        clinical_emb = self.clinical_proj(clinical_features)
        combined = torch.cat([aggregated, clinical_emb], dim=-1)
        return self.combine(combined)


class PredictionHeads(nn.Module):
    """
    Layer 7: Prediction Heads
    
    Three output heads:
    1. Attack Probability: P(attack in next 24h)
    2. Attack Severity: Expected severity if attack
    3. Trigger Attribution: Importance scores
    
    Plus SSL head for masked reconstruction.
    """
    
    def __init__(self, config: MigraineModelConfig):
        super().__init__()
        
        # Attack probability head
        self.attack_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(32, 1),
        )
        
        # Severity head
        self.severity_head = nn.Sequential(
            nn.Linear(config.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        # Trigger attribution head
        self.trigger_head = nn.Sequential(
            nn.Linear(config.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, config.n_trigger_classes),
        )
        
        # Reconstruction head for SSL
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.n_continuous_features + config.n_binary_features),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        attack_logits = self.attack_head(x).squeeze(-1)
        outputs = {"attack_logits": attack_logits}
        
        if return_all:
            severity = self.severity_head(x).squeeze(-1)
            severity = torch.sigmoid(severity) * 9 + 1  # Scale to 1-10
            outputs["severity"] = severity
            
            trigger_logits = self.trigger_head(x)
            outputs["trigger_importance"] = F.softmax(trigger_logits, dim=-1)
            outputs["trigger_logits"] = trigger_logits
        
        return outputs
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction for SSL pre-training."""
        return self.reconstruction_head(x)


class MigraineMamba(nn.Module):
    """
    Complete MigraineMamba Model
    
    Combines all layers for migraine prediction.
    
    Input: 14 days of patient data
    Output: Attack probability, severity, trigger attribution
    
    Expected parameters: ~800K
    """
    
    def __init__(self, config: Optional[MigraineModelConfig] = None):
        super().__init__()
        
        self.config = config or MigraineModelConfig()
        
        # Layer 1: Feature Embedding
        self.feature_embedding = FeatureEmbedding(self.config)
        
        # Layer 2: Positional Encoding
        self.positional_encoding = PositionalEncoding(self.config)
        
        # Layer 3: Mamba Backbone
        self.mamba = MambaBackbone(
            d_model=self.config.d_model,
            n_layers=self.config.n_mamba_layers,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
            expand=self.config.expand,
            dropout=self.config.dropout,
        )
        
        # Layer 4: Clinical Attention
        self.clinical_attention = ClinicalAttention(self.config)
        self.attention_norm = RMSNorm(self.config.d_model)
        
        # Layer 5: Temporal Aggregation
        self.temporal_aggregation = TemporalAggregation(self.config)
        
        # Layer 6: Clinical Knowledge Integration
        self.clinical_integration = ClinicalKnowledgeIntegration(self.config)
        
        # Layer 7: Prediction Heads
        self.prediction_heads = PredictionHeads(self.config)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        continuous_features: torch.Tensor,
        binary_features: torch.Tensor,
        menstrual_day: torch.Tensor,
        day_of_week: torch.Tensor,
        clinical_features: torch.Tensor,
        return_attention: bool = False,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            continuous_features: (batch, seq_len, n_continuous)
            binary_features: (batch, seq_len, n_binary)
            menstrual_day: (batch, seq_len)
            day_of_week: (batch, seq_len)
            clinical_features: (batch, n_clinical)
            
        Returns:
            Dictionary with predictions
        """
        # Layer 1: Embed features
        embedded = self.feature_embedding(
            continuous_features, binary_features, menstrual_day, day_of_week
        )
        
        # Layer 2: Positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Layer 3: Mamba temporal processing
        mamba_output = self.mamba(embedded)
        
        # Layer 4: Clinical attention (+ residual)
        attended, attention_weights = self.clinical_attention(
            mamba_output, return_attention=return_attention
        )
        attended = self.attention_norm(attended + mamba_output)
        
        # Layer 5: Temporal aggregation
        aggregated = self.temporal_aggregation(attended)
        
        # Layer 6: Clinical knowledge integration
        combined = self.clinical_integration(aggregated, clinical_features)
        
        # Layer 7: Predictions
        outputs = self.prediction_heads(combined, return_all=return_all)
        
        if return_attention and attention_weights is not None:
            outputs["attention_weights"] = attention_weights
        
        return outputs
    
    def forward_ssl(
        self,
        continuous_features: torch.Tensor,
        binary_features: torch.Tensor,
        menstrual_day: torch.Tensor,
        day_of_week: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised learning.
        
        Args:
            mask: (batch, seq_len) boolean mask of positions to reconstruct
            
        Returns:
            reconstructed: (batch, seq_len, n_features)
            mamba_output: (batch, seq_len, d_model)
        """
        # Embed features
        embedded = self.feature_embedding(
            continuous_features, binary_features, menstrual_day, day_of_week
        )
        
        # Positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Apply mask (zero out masked positions)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            embedded = embedded * (1 - mask_expanded)
        
        # Mamba processing
        mamba_output = self.mamba(embedded)
        
        # Reconstruct
        reconstructed = self.prediction_heads.reconstruct(mamba_output)
        
        return reconstructed, mamba_output
    
    def get_attention_interpretation(self) -> Dict[str, torch.Tensor]:
        """Get interpretable attention weights."""
        head_names = ["sleep", "weather", "stress", "hormonal"]
        attention = self.clinical_attention.attention_weights
        
        if attention is None:
            return {}
        
        return {
            name: attention[:, i, :, :]
            for i, name in enumerate(head_names)
            if i < attention.size(1)
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            "feature_embedding": sum(p.numel() for p in self.feature_embedding.parameters()),
            "positional_encoding": sum(p.numel() for p in self.positional_encoding.parameters()),
            "mamba_backbone": sum(p.numel() for p in self.mamba.parameters()),
            "clinical_attention": sum(p.numel() for p in self.clinical_attention.parameters()),
            "temporal_aggregation": sum(p.numel() for p in self.temporal_aggregation.parameters()),
            "clinical_integration": sum(p.numel() for p in self.clinical_integration.parameters()),
            "prediction_heads": sum(p.numel() for p in self.prediction_heads.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


def create_model(
    n_continuous: int = 8,
    n_binary: int = 6,
    seq_len: int = 14,
    d_model: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
) -> MigraineMamba:
    """Factory function to create MigraineMamba model."""
    config = MigraineModelConfig(
        n_continuous_features=n_continuous,
        n_binary_features=n_binary,
        seq_len=seq_len,
        d_model=d_model,
        n_mamba_layers=n_layers,
        dropout=dropout,
    )
    return MigraineMamba(config)


def test_model():
    """Test the complete MigraineMamba model."""
    print("=" * 60)
    print("Testing MigraineMamba Model (Phase 2.3)")
    print("=" * 60)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Create model
    config = MigraineModelConfig()
    model = MigraineMamba(config).to(device)
    
    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter counts by component:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Create dummy input
    batch_size = 4
    seq_len = 14
    
    continuous = torch.randn(batch_size, seq_len, 8).to(device)
    binary = torch.randint(0, 2, (batch_size, seq_len, 6)).float().to(device)
    menstrual = torch.randint(-1, 28, (batch_size, seq_len)).to(device)
    dow = torch.randint(0, 7, (batch_size, seq_len)).to(device)
    clinical = torch.randn(batch_size, 8).to(device)
    
    print(f"\nInput shapes:")
    print(f"  continuous: {continuous.shape}")
    print(f"  binary: {binary.shape}")
    print(f"  menstrual: {menstrual.shape}")
    print(f"  day_of_week: {dow.shape}")
    print(f"  clinical: {clinical.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            continuous, binary, menstrual, dow, clinical,
            return_attention=True,
            return_all=True
        )
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test SSL forward
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
    mask[:, [3, 7, 11]] = True
    
    reconstructed, mamba_out = model.forward_ssl(
        continuous, binary, menstrual, dow, mask
    )
    
    print(f"\nSSL output shapes:")
    print(f"  reconstructed: {reconstructed.shape}")
    print(f"  mamba_output: {mamba_out.shape}")
    
    print("\n" + "=" * 60)
    print("✓ MigraineMamba test PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_model()