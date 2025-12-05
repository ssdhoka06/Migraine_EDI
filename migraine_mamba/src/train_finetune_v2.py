"""
Phase 2.5b: Optimized Fine-Tuning
==================================
Retrain with settings optimized for better trigger differentiation.

Key changes:
- Lower pos_weight (1.5 instead of 4.84)
- Higher learning rate (1e-4)
- Focal loss option for hard examples
- More epochs with patience

Usage:
    python train_finetune_v2.py \
        --data-dir /Users/sachidhoka/Desktop/Migraine_EDI/processed \
        --pretrained ../models/mamba_ssl.pth \
        --output-dir ../models_v2 \
        --epochs 30

Author: Dhoka
Date: December 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from model import MigraineMamba, MigraineModelConfig


class FocalLoss(nn.Module):
    """
    Focal Loss - focuses on hard examples.
    Better than BCE for imbalanced data when we want discrimination.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class FinetuneDataset(Dataset):
    def __init__(self, tensors):
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


class OptimizedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Loss function choice
        loss_type = config.get('loss_type', 'focal')
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print("  Using Focal Loss (α=0.25, γ=2.0)")
        else:
            pos_weight = config.get('pos_weight', 1.5)
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(device)
            )
            print(f"  Using BCE Loss (pos_weight={pos_weight})")
        
        # Optimizer with higher LR
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=1,
            eta_min=1e-6,
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_auc': [],
            'val_f1': [], 'val_precision': [], 'val_recall': [],
            'val_spread': [],  # Track prediction spread
        }
        
        self.best_val_auc = 0.0
        self.best_val_spread = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(continuous, binary, menstrual, day_of_week, clinical)
            loss = self.criterion(outputs['attack_logits'], targets)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs['attack_logits'])
            all_preds.extend(probs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        self.scheduler.step()
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate spread (std of predictions)
        pred_spread = np.std(all_preds)
        
        try:
            train_auc = roc_auc_score(all_targets, all_preds)
        except:
            train_auc = 0.5
        
        return {
            'loss': total_loss / len(self.train_loader),
            'auc': train_auc,
            'spread': pred_spread,
        }
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in loader:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            targets = batch['target'].to(self.device)
            
            outputs = self.model(continuous, binary, menstrual, day_of_week, clinical)
            loss = self.criterion(outputs['attack_logits'], targets)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            probs = torch.sigmoid(outputs['attack_logits'])
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Prediction spread - we WANT this to be high
        pred_spread = np.std(all_preds)
        pred_range = np.max(all_preds) - np.min(all_preds)
        
        # Metrics
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
        
        # Find best threshold
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds_bin = (all_preds > thresh).astype(int)
            f1 = f1_score(all_targets, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        preds_bin = (all_preds > best_thresh).astype(int)
        precision = precision_score(all_targets, preds_bin, zero_division=0)
        recall = recall_score(all_targets, preds_bin, zero_division=0)
        
        return {
            'loss': total_loss / len(loader),
            'auc': auc,
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thresh,
            'spread': pred_spread,
            'range': pred_range,
            'mean_pred': np.mean(all_preds),
        }
    
    def train(self, epochs, save_dir, patience=10):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("OPTIMIZED FINE-TUNING - MigraineMamba v2")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.config.get('lr', 1e-4)}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            
            # Record
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_spread'].append(val_metrics['spread'])
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc']:.4f} | Spread: {train_metrics['spread']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | Spread: {val_metrics['spread']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f} | Range: [{val_metrics['mean_pred']-val_metrics['range']/2:.2f}, {val_metrics['mean_pred']+val_metrics['range']/2:.2f}]")
            
            # Save best by AUC * spread (we want both high AUC AND good spread)
            combined_score = val_metrics['auc'] * (1 + val_metrics['spread'])
            
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_val_spread = val_metrics['spread']
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(save_dir, 'best')
                print(f"  ✓ New best! AUC: {self.best_val_auc:.4f}, Spread: {self.best_val_spread:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping at epoch {epoch}")
                    break
        
        # Final test evaluation
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)
        
        checkpoint = torch.load(f'{save_dir}/mamba_finetuned_v2.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader)
        
        print(f"\nTest Results:")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  Prediction Spread: {test_metrics['spread']:.4f}")
        print(f"  Prediction Range: {test_metrics['range']:.4f}")
        
        # Save results
        results = {
            'best_epoch': self.best_epoch,
            'best_val_auc': self.best_val_auc,
            'test_auc': test_metrics['auc'],
            'test_f1': test_metrics['f1'],
            'test_spread': test_metrics['spread'],
            'history': self.history,
        }
        
        with open(f'{save_dir}/finetune_v2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Model saved to {save_dir}/mamba_finetuned_v2.pth")
        
        return results
    
    def save_checkpoint(self, save_dir, name):
        path = f'{save_dir}/mamba_finetuned_v2.pth' if name == 'best' else f'{save_dir}/mamba_finetuned_v2_{name}.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_auc': self.best_val_auc,
        }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', required=True)
    parser.add_argument('--pretrained', '-p', required=True)
    parser.add_argument('--output-dir', '-o', default='../models_v2')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch-size', '-b', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pos-weight', type=float, default=1.5)
    parser.add_argument('--loss', choices=['bce', 'focal'], default='focal')
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load data
    print(f"\nLoading data...")
    train_tensors = torch.load(f'{args.data_dir}/train_tensors.pt', weights_only=False)
    val_tensors = torch.load(f'{args.data_dir}/val_tensors.pt', weights_only=False)
    test_tensors = torch.load(f'{args.data_dir}/test_tensors.pt', weights_only=False)
    
    print(f"  Train: {len(train_tensors['targets']):,}")
    print(f"  Val: {len(val_tensors['targets']):,}")
    print(f"  Test: {len(test_tensors['targets']):,}")
    
    # Datasets
    train_loader = DataLoader(FinetuneDataset(train_tensors), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FinetuneDataset(val_tensors), batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(FinetuneDataset(test_tensors), batch_size=args.batch_size, num_workers=0)
    
    # Load model
    print(f"\nLoading pre-trained model...")
    config = MigraineModelConfig(
        n_continuous_features=8, n_binary_features=6, seq_len=14,
        d_model=64, n_mamba_layers=2, dropout=0.3,
    )
    model = MigraineMamba(config)
    
    checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded from {args.pretrained}")
    
    # Train config
    train_config = {
        'lr': args.lr,
        'weight_decay': 0.01,
        'pos_weight': args.pos_weight,
        'loss_type': args.loss,
    }
    
    # Train
    trainer = OptimizedTrainer(model, train_loader, val_loader, test_loader, device, train_config)
    trainer.train(args.epochs, args.output_dir, patience=args.patience)
    
    print("\n✅ Optimized fine-tuning complete!")
    print(f"\nNext: test with test_prediction_v2.py --model {args.output_dir}/mamba_finetuned_v2.pth")


if __name__ == "__main__":
    main()