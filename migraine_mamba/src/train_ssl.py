"""
Self-Supervised Pre-Training - Phase 2.4 (Fixed)
=================================================
Train MigraineMamba on synthetic data using 4 SSL tasks.

Fixed issues:
- NaN handling in contrastive loss
- Gradient clipping
- Loss scaling
- Better numerical stability

Usage:
    python train_ssl.py --epochs 50 --batch-size 64

Author: Dhoka
Date: December 2025
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

# Import our modules
from model import MigraineMamba, MigraineModelConfig
from make_tensors import MigraineDataset


class SSLMigraineDataset(Dataset):
    """Dataset for SSL pre-training."""
    
    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        mask_ratio: float = 0.25,
        n_triggers: int = 7,
    ):
        self.continuous = tensors['continuous']
        self.binary = tensors['binary']
        self.menstrual = tensors['menstrual']
        self.day_of_week = tensors['day_of_week']
        self.clinical = tensors['clinical']
        self.targets = tensors['targets']
        
        self.mask_ratio = mask_ratio
        self.seq_len = self.continuous.shape[1]
        self.n_features = self.continuous.shape[2] + self.binary.shape[2]
        self.n_triggers = n_triggers
        
        # Handle NaN in input data
        self.continuous = torch.nan_to_num(self.continuous, nan=0.0)
        self.binary = torch.nan_to_num(self.binary, nan=0.0)
        self.clinical = torch.nan_to_num(self.clinical, nan=0.0)
        
        self._precompute_triggers()
    
    def _precompute_triggers(self):
        """Derive trigger labels from features."""
        self.trigger_labels = []
        
        for i in range(len(self.targets)):
            triggers = torch.zeros(self.n_triggers)
            
            # Sleep trigger
            sleep = self.continuous[i, -3:, 0]
            if (sleep < -0.5).any():
                triggers[0] = 1
            
            # Stress trigger
            stress = self.continuous[i, -3:, 1]
            if (stress > 0.5).any():
                triggers[1] = 1
            
            # Weather trigger
            if self.continuous.shape[2] > 3:
                pressure_change = self.continuous[i, -3:, 3]
                if (pressure_change < -0.3).any():
                    triggers[2] = 1
            
            # Fasting trigger
            if self.continuous.shape[2] > 6:
                fasting = self.continuous[i, -3:, 6]
                if (fasting > 0.3).any():
                    triggers[3] = 1
            
            # Alcohol trigger
            if self.continuous.shape[2] > 7:
                alcohol = self.continuous[i, -3:, 7]
                if (alcohol > 0.3).any():
                    triggers[4] = 1
            
            # Menstrual trigger
            menstrual = self.menstrual[i, -1].item()
            if menstrual in [0, 1, 26, 27]:
                triggers[5] = 1
            
            # Light trigger
            if self.binary.shape[2] > 4:
                light = self.binary[i, -3:, 4]
                if (light > 0.5).any():
                    triggers[6] = 1
            
            self.trigger_labels.append(triggers)
        
        self.trigger_labels = torch.stack(self.trigger_labels)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Random mask
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        n_mask = max(1, int(self.seq_len * self.mask_ratio))
        mask_indices = torch.randperm(self.seq_len)[:n_mask]
        mask[mask_indices] = True
        
        return {
            'continuous': self.continuous[idx],
            'binary': self.binary[idx],
            'menstrual': self.menstrual[idx],
            'day_of_week': self.day_of_week[idx],
            'clinical': self.clinical[idx],
            'target': self.targets[idx],
            'mask': mask,
            'trigger_labels': self.trigger_labels[idx],
            'idx': idx,
        }


class SSLTrainer:
    """SSL Trainer with stable training."""
    
    def __init__(
        self,
        model: MigraineMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss weights (reduced contrastive weight for stability)
        self.lambda_masked = config.get('lambda_masked', 1.0)
        self.lambda_forecast = config.get('lambda_forecast', 1.0)
        self.lambda_trigger = config.get('lambda_trigger', 0.5)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Optimizer with lower learning rate for stability
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            eps=1e-8,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 50),
            eta_min=1e-6,
        )
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'forecast_auc': [],
            'trigger_f1': [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with NaN protection."""
        self.model.train()
        
        total_loss = 0
        total_masked = 0
        total_forecast = 0
        total_trigger = 0
        n_batches = 0
        nan_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            targets = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            trigger_labels = batch['trigger_labels'].to(self.device)
            
            # Replace any NaN in inputs
            continuous = torch.nan_to_num(continuous, nan=0.0)
            binary = torch.nan_to_num(binary, nan=0.0)
            clinical = torch.nan_to_num(clinical, nan=0.0)
            
            original_features = torch.cat([continuous, binary], dim=-1)
            
            self.optimizer.zero_grad()
            
            # Task 1: Masked Prediction
            reconstructed, mamba_output = self.model.forward_ssl(
                continuous, binary, menstrual, day_of_week, mask
            )
            
            # Only compute loss on masked positions
            if mask.sum() > 0:
                masked_pred = reconstructed[mask]
                masked_target = original_features[mask]
                masked_loss = self.mse_loss(masked_pred, masked_target)
            else:
                masked_loss = torch.tensor(0.0, device=self.device)
            
            # Task 2: Attack Forecasting
            outputs = self.model(
                continuous, binary, menstrual, day_of_week, clinical
            )
            forecast_loss = self.bce_loss(outputs['attack_logits'], targets)
            
            # Task 3: Trigger Identification
            aggregated = self.model.temporal_aggregation(mamba_output)
            trigger_logits = self.model.prediction_heads.trigger_head(aggregated)
            trigger_loss = self.bce_loss(trigger_logits, trigger_labels)
            
            # Combined loss (no contrastive for stability)
            loss = (
                self.lambda_masked * masked_loss +
                self.lambda_forecast * forecast_loss +
                self.lambda_trigger * trigger_loss
            )
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
            
            # Backward with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_masked += masked_loss.item()
            total_forecast += forecast_loss.item()
            total_trigger += trigger_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'fore': f'{forecast_loss.item():.4f}',
            })
        
        if nan_batches > 0:
            print(f"  Warning: {nan_batches} batches had NaN loss")
        
        if n_batches == 0:
            return {'loss': float('inf')}
        
        return {
            'loss': total_loss / n_batches,
            'masked_loss': total_masked / n_batches,
            'forecast_loss': total_forecast / n_batches,
            'trigger_loss': total_trigger / n_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        all_trigger_preds = []
        all_trigger_targets = []
        n_batches = 0
        
        for batch in self.val_loader:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            targets = batch['target'].to(self.device)
            trigger_labels = batch['trigger_labels'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Replace NaN
            continuous = torch.nan_to_num(continuous, nan=0.0)
            binary = torch.nan_to_num(binary, nan=0.0)
            clinical = torch.nan_to_num(clinical, nan=0.0)
            
            original_features = torch.cat([continuous, binary], dim=-1)
            
            reconstructed, mamba_output = self.model.forward_ssl(
                continuous, binary, menstrual, day_of_week, mask
            )
            outputs = self.model(
                continuous, binary, menstrual, day_of_week, clinical
            )
            aggregated = self.model.temporal_aggregation(mamba_output)
            trigger_logits = self.model.prediction_heads.trigger_head(aggregated)
            
            if mask.sum() > 0:
                masked_loss = self.mse_loss(reconstructed[mask], original_features[mask])
            else:
                masked_loss = torch.tensor(0.0, device=self.device)
            
            forecast_loss = self.bce_loss(outputs['attack_logits'], targets)
            trigger_loss = self.bce_loss(trigger_logits, trigger_labels)
            
            loss = (
                self.lambda_masked * masked_loss +
                self.lambda_forecast * forecast_loss +
                self.lambda_trigger * trigger_loss
            )
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
            
            probs = torch.sigmoid(outputs['attack_logits'])
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            trigger_probs = torch.sigmoid(trigger_logits)
            all_trigger_preds.extend(trigger_probs.cpu().numpy())
            all_trigger_targets.extend(trigger_labels.cpu().numpy())
        
        # Metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Filter out NaN predictions
        valid_mask = ~(np.isnan(all_preds) | np.isnan(all_targets))
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask]
        
        try:
            if len(np.unique(all_targets)) > 1:
                forecast_auc = roc_auc_score(all_targets, all_preds)
            else:
                forecast_auc = 0.5
        except:
            forecast_auc = 0.5
        
        all_trigger_preds = np.array(all_trigger_preds)
        all_trigger_targets = np.array(all_trigger_targets)
        trigger_preds_binary = (all_trigger_preds > 0.5).astype(int)
        
        try:
            trigger_f1 = f1_score(
                all_trigger_targets.flatten(),
                trigger_preds_binary.flatten(),
                average='macro',
                zero_division=0
            )
        except:
            trigger_f1 = 0.0
        
        return {
            'loss': total_loss / max(n_batches, 1),
            'forecast_auc': forecast_auc,
            'trigger_f1': trigger_f1,
        }
    
    def train(self, epochs: int, save_dir: str):
        """Full training loop."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("SSL PRE-TRAINING - MigraineMamba")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Learning rate: {self.config.get('lr', 1e-4)}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['forecast_auc'].append(val_metrics['forecast_auc'])
            self.history['trigger_f1'].append(val_metrics['trigger_f1'])
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Forecast AUC: {val_metrics['forecast_auc']:.4f}")
            print(f"  Trigger F1:   {val_metrics['trigger_f1']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.save_checkpoint(save_dir, 'best')
                print(f"  ✓ New best model!")
            
            if epoch % 10 == 0:
                self.save_checkpoint(save_dir, f'epoch_{epoch}')
        
        total_time = time.time() - start_time
        
        self.save_checkpoint(save_dir, 'final')
        
        with open(f'{save_dir}/ssl_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("SSL PRE-TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Time: {total_time/60:.1f} min")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final AUC: {self.history['forecast_auc'][-1]:.4f}")
        print(f"Model: {save_dir}/mamba_ssl.pth")
        print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, save_dir: str, name: str):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
        
        path = f'{save_dir}/mamba_ssl.pth' if name == 'best' else f'{save_dir}/mamba_ssl_{name}.pth'
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="SSL Pre-training for MigraineMamba")
    parser.add_argument('--data-dir', '-d', default='processed', help='Data directory')
    parser.add_argument('--output-dir', '-o', default='models', help='Output directory')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Mamba layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_tensors = torch.load(f'{args.data_dir}/train_tensors.pt', weights_only=False)
    val_tensors = torch.load(f'{args.data_dir}/val_tensors.pt', weights_only=False)
    
    print(f"Train: {len(train_tensors['targets']):,} samples")
    print(f"Val: {len(val_tensors['targets']):,} samples")
    
    # Check for NaN in data
    for name, tensor in train_tensors.items():
        if torch.is_floating_point(tensor):
            nan_count = torch.isnan(tensor).sum().item()
            if nan_count > 0:
                print(f"  Warning: {name} has {nan_count} NaN values")
    
    # Datasets
    train_dataset = SSLMigraineDataset(train_tensors, mask_ratio=0.25)
    val_dataset = SSLMigraineDataset(val_tensors, mask_ratio=0.25)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # Disabled for MPS stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Model
    config = MigraineModelConfig(
        n_continuous_features=train_tensors['continuous'].shape[-1],
        n_binary_features=train_tensors['binary'].shape[-1],
        seq_len=train_tensors['continuous'].shape[1],
        d_model=args.d_model,
        n_mamba_layers=args.n_layers,
        dropout=args.dropout,
    )
    
    model = MigraineMamba(config)
    
    # Initialize weights properly
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Train config
    train_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 0.01,
        'lambda_masked': 1.0,
        'lambda_forecast': 1.0,
        'lambda_trigger': 0.5,
    }
    
    # Trainer
    trainer = SSLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=train_config,
    )
    
    # Train!
    trainer.train(args.epochs, args.output_dir)
    
    print("\n✓ SSL pre-training complete!")


if __name__ == "__main__":
    main()