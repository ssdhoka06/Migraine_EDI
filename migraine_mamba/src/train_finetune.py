"""
Phase 2.5: Fine-Tuning for Attack Prediction
=============================================
Load SSL pre-trained MigraineMamba and fine-tune specifically
for the attack prediction task with class balancing.

Usage:
    python train_finetune.py \
        --data-dir /path/to/processed \
        --pretrained ../models/mamba_ssl.pth \
        --output-dir ../models \
        --epochs 20

Author: Dhoka
Date: December 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)

from model import MigraineMamba, MigraineModelConfig


class FinetuneDataset(Dataset):
    """Dataset for fine-tuning on attack prediction."""
    
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.continuous = torch.nan_to_num(tensors['continuous'], nan=0.0)
        self.binary = torch.nan_to_num(tensors['binary'], nan=0.0)
        self.menstrual = tensors['menstrual']
        self.day_of_week = tensors['day_of_week']
        self.clinical = torch.nan_to_num(tensors['clinical'], nan=0.0)
        self.targets = tensors['targets']
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'continuous': self.continuous[idx],
            'binary': self.binary[idx],
            'menstrual': self.menstrual[idx],
            'day_of_week': self.day_of_week[idx],
            'clinical': self.clinical[idx],
            'target': self.targets[idx],
        }


class FinetuneTrainer:
    """Fine-tuning trainer with class balancing and early stopping."""
    
    def __init__(
        self,
        model: MigraineMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config: dict,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Class weights for imbalanced data
        pos_weight = config.get('pos_weight', 1.0)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # Optimizer - lower LR for fine-tuning
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 3e-5),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Scheduler - removed 'verbose' parameter
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
        }
        
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                continuous, binary, menstrual, day_of_week, clinical
            )
            
            loss = self.criterion(outputs['attack_logits'], targets)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            probs = torch.sigmoid(outputs['attack_logits'])
            all_preds.extend(probs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute training AUC
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        try:
            train_auc = roc_auc_score(all_targets, all_preds)
        except:
            train_auc = 0.5
        
        return {
            'loss': total_loss / max(n_batches, 1),
            'auc': train_auc,
        }
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, name: str = "Val") -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        all_severities = []
        all_trigger_scores = []
        n_batches = 0
        
        for batch in loader:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            targets = batch['target'].to(self.device)
            
            outputs = self.model(
                continuous, binary, menstrual, day_of_week, clinical
            )
            
            loss = self.criterion(outputs['attack_logits'], targets)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1
            
            probs = torch.sigmoid(outputs['attack_logits'])
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Only collect if these outputs exist
            if 'severity' in outputs:
                all_severities.extend(outputs['severity'].cpu().numpy())
            if 'trigger_importance' in outputs:
                all_trigger_scores.extend(outputs['trigger_importance'].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Filter NaN
        valid = ~(np.isnan(all_preds) | np.isnan(all_targets))
        all_preds = all_preds[valid]
        all_targets = all_targets[valid]
        
        # Metrics
        try:
            auc = roc_auc_score(all_targets, all_preds)
            ap = average_precision_score(all_targets, all_preds)
        except:
            auc, ap = 0.5, 0.0
        
        # Find optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.2, 0.8, 0.05):
            preds_binary = (all_preds > thresh).astype(int)
            f1 = f1_score(all_targets, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        preds_binary = (all_preds > best_thresh).astype(int)
        precision = precision_score(all_targets, preds_binary, zero_division=0)
        recall = recall_score(all_targets, preds_binary, zero_division=0)
        
        return {
            'loss': total_loss / max(n_batches, 1),
            'auc': auc,
            'ap': ap,
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thresh,
            'severities': np.array(all_severities) if all_severities else np.array([]),
            'trigger_scores': np.array(all_trigger_scores) if all_trigger_scores else np.array([]),
        }
    
    def train(self, epochs: int, save_dir: str, patience: int = 7):
        """Full training loop with early stopping."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("FINE-TUNING - MigraineMamba")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Class weight (pos): {self.config.get('pos_weight', 1.0):.2f}")
        print(f"Learning rate: {self.config.get('lr', 3e-5)}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "Val")
            
            # Get current learning rate before scheduler step
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.scheduler.step(val_metrics['auc'])
            
            # Check if learning rate changed
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"  Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train AUC: {train_metrics['auc']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val AUC:   {val_metrics['auc']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")
            print(f"  Optimal threshold: {val_metrics['threshold']:.2f}")
            
            # Check for improvement
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(save_dir, 'best')
                print(f"  ✓ New best model! AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{patience})")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Load best model
        best_path = f'{save_dir}/mamba_finetuned.pth'
        checkpoint = torch.load(best_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, "Test")
        
        print(f"\nTest Results (Best model from epoch {self.best_epoch}):")
        print(f"  AUC:       {test_metrics['auc']:.4f}")
        print(f"  AP:        {test_metrics['ap']:.4f}")
        print(f"  F1:        {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  Threshold: {test_metrics['threshold']:.2f}")
        
        # Save final results
        results = {
            'training_time_min': total_time / 60,
            'best_epoch': self.best_epoch,
            'best_val_auc': self.best_val_auc,
            'test_auc': test_metrics['auc'],
            'test_ap': test_metrics['ap'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'optimal_threshold': test_metrics['threshold'],
            'history': self.history,
        }
        
        with open(f'{save_dir}/finetune_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Trigger importance analysis
        if len(test_metrics['trigger_scores']) > 0:
            self._analyze_triggers(test_metrics['trigger_scores'], save_dir)
        else:
            print("\n⚠ No trigger scores available - model may not output trigger_importance")
        
        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETE!")
        print("=" * 60)
        print(f"Time: {total_time/60:.1f} min")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Model: {save_dir}/mamba_finetuned.pth")
        print(f"Results: {save_dir}/finetune_results.json")
        print("=" * 60)
        
        return results
    
    def _analyze_triggers(self, trigger_scores: np.ndarray, save_dir: str):
        """Analyze trigger importance from model predictions."""
        trigger_names = [
            'Sleep', 'Stress', 'Weather', 'Fasting',
            'Alcohol', 'Menstrual', 'Light'
        ]
        
        mean_importance = trigger_scores.mean(axis=0)
        
        print("\n📊 Trigger Importance (Model Learned):")
        sorted_idx = np.argsort(mean_importance)[::-1]
        for i, idx in enumerate(sorted_idx):
            if idx < len(trigger_names):
                bar = "█" * int(mean_importance[idx] * 20)
                print(f"  {i+1}. {trigger_names[idx]:12} {mean_importance[idx]:.3f} {bar}")
        
        # Save
        trigger_analysis = {
            'trigger_names': trigger_names,
            'mean_importance': mean_importance.tolist(),
            'ranking': [trigger_names[i] for i in sorted_idx if i < len(trigger_names)],
        }
        
        with open(f'{save_dir}/trigger_analysis.json', 'w') as f:
            json.dump(trigger_analysis, f, indent=2)
    
    def save_checkpoint(self, save_dir: str, name: str):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
        }
        
        path = f'{save_dir}/mamba_finetuned.pth' if name == 'best' else f'{save_dir}/mamba_finetuned_{name}.pth'
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MigraineMamba")
    parser.add_argument('--data-dir', '-d', required=True, help='Data directory')
    parser.add_argument('--pretrained', '-p', required=True, help='Pre-trained SSL checkpoint')
    parser.add_argument('--output-dir', '-o', default='models', help='Output directory')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Max epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--pos-weight', type=float, default=None, help='Positive class weight (auto if not set)')
    
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
    test_tensors = torch.load(f'{args.data_dir}/test_tensors.pt', weights_only=False)
    
    print(f"Train: {len(train_tensors['targets']):,} samples")
    print(f"Val: {len(val_tensors['targets']):,} samples")
    print(f"Test: {len(test_tensors['targets']):,} samples")
    
    # Compute class weights
    pos_rate = train_tensors['targets'].float().mean().item()
    neg_rate = 1 - pos_rate
    auto_pos_weight = neg_rate / pos_rate if pos_rate > 0 else 1.0
    pos_weight = args.pos_weight if args.pos_weight else auto_pos_weight
    
    print(f"\nClass distribution:")
    print(f"  Positive (attack): {pos_rate*100:.1f}%")
    print(f"  Negative (no attack): {neg_rate*100:.1f}%")
    print(f"  Using pos_weight: {pos_weight:.2f}")
    
    # Datasets
    train_dataset = FinetuneDataset(train_tensors)
    val_dataset = FinetuneDataset(val_tensors)
    test_dataset = FinetuneDataset(test_tensors)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model from {args.pretrained}...")
    checkpoint = torch.load(args.pretrained, weights_only=False, map_location=device)
    
    # Recreate model config
    config = MigraineModelConfig(
        n_continuous_features=train_tensors['continuous'].shape[-1],
        n_binary_features=train_tensors['binary'].shape[-1],
        seq_len=train_tensors['continuous'].shape[1],
        d_model=64,
        n_mamba_layers=2,
        dropout=0.3,
    )
    
    model = MigraineMamba(config)
    
    # Load weights (handle potential key mismatches)
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    
    # Filter out mismatched keys
    matched_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    
    print(f"  Loaded {len(matched_dict)}/{len(model_dict)} layers from pre-trained model")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Training config
    train_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 0.01,
        'pos_weight': pos_weight,
    }
    
    # Trainer
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=train_config,
    )
    
    # Train!
    results = trainer.train(args.epochs, args.output_dir, patience=args.patience)
    
    print("\n✓ Fine-tuning complete!")
    print(f"\n📁 Output files:")
    print(f"   {args.output_dir}/mamba_finetuned.pth")
    print(f"   {args.output_dir}/finetune_results.json")
    print(f"   {args.output_dir}/trigger_analysis.json")


if __name__ == "__main__":
    main()