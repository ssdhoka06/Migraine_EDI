"""
Phase 3.2: Weekly Personalization Pipeline
==========================================
Runs every Sunday to fine-tune models for individual users.

Process:
1. Check eligibility (4+ days logged this week)
2. Prepare user's historical data
3. Initialize from generic model
4. Fine-tune with frozen backbone
5. Validate on held-out data
6. Save if improved

Author: Dhoka
Date: December 2025
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import MigraineMamba, MigraineModelConfig


class UserDataset(Dataset):
    """Dataset for a single user's historical data."""
    
    def __init__(self, sequences: List[Dict], labels: List[int]):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'continuous': torch.tensor(seq['continuous'], dtype=torch.float32),
            'binary': torch.tensor(seq['binary'], dtype=torch.float32),
            'menstrual': torch.tensor(seq['menstrual'], dtype=torch.int64),
            'day_of_week': torch.tensor(seq['day_of_week'], dtype=torch.int64),
            'clinical': torch.tensor(seq['clinical'], dtype=torch.float32),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class WeeklyPersonalizer:
    """
    Weekly personalization pipeline.
    
    Fine-tunes the generic model for each eligible user.
    """
    
    def __init__(
        self,
        generic_model_path: str,
        user_data_dir: str,
        output_dir: str = None,
        device: str = None,
    ):
        self.generic_model_path = Path(generic_model_path)
        self.user_data_dir = Path(user_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.user_data_dir
        
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.model_config = MigraineModelConfig(
            n_continuous_features=8,
            n_binary_features=6,
            seq_len=14,
            d_model=64,
            n_mamba_layers=2,
            dropout=0.3,
        )
        
        # Feature names
        self.continuous_features = [
            'sleep_hours', 'stress_level', 'barometric_pressure', 'pressure_change',
            'temperature', 'humidity', 'hours_fasting', 'alcohol_drinks'
        ]
        self.binary_features = [
            'had_breakfast', 'had_lunch', 'had_dinner', 'had_snack',
            'bright_light_exposure', 'sleep_quality'
        ]
        
        # Normalization
        self.means = np.array([7.0, 5.0, 1013.0, 0.0, 20.0, 50.0, 6.0, 0.5])
        self.stds = np.array([1.5, 2.5, 10.0, 5.0, 10.0, 20.0, 4.0, 1.0])
        self.stds = np.where(self.stds == 0, 1.0, self.stds)
        
        print(f"✓ WeeklyPersonalizer initialized on {self.device}")
    
    def run_weekly_update(self):
        """
        Main entry point - run weekly update for all eligible users.
        """
        print("\n" + "=" * 60)
        print("🔄 WEEKLY PERSONALIZATION - " + datetime.now().strftime("%Y-%m-%d"))
        print("=" * 60)
        
        # Find all users
        users = self._get_all_users()
        print(f"\nFound {len(users)} users")
        
        results = {
            'date': datetime.now().isoformat(),
            'users_processed': 0,
            'users_updated': 0,
            'users_skipped': 0,
            'details': [],
        }
        
        for user_id in users:
            print(f"\n--- Processing: {user_id} ---")
            
            result = self._process_user(user_id)
            results['details'].append(result)
            
            if result['status'] == 'updated':
                results['users_updated'] += 1
                results['users_processed'] += 1
            elif result['status'] == 'no_improvement':
                results['users_processed'] += 1
            else:
                results['users_skipped'] += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 WEEKLY SUMMARY")
        print("=" * 60)
        print(f"   Users processed: {results['users_processed']}")
        print(f"   Models updated:  {results['users_updated']}")
        print(f"   Skipped:         {results['users_skipped']}")
        
        # Save results
        results_path = self.output_dir / "weekly_results" / f"{datetime.now().strftime('%Y%m%d')}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _get_all_users(self) -> List[str]:
        """Get list of all user IDs."""
        users = []
        for user_dir in self.user_data_dir.iterdir():
            if user_dir.is_dir() and (user_dir / "state.json").exists():
                users.append(user_dir.name)
        return users
    
    def _process_user(self, user_id: str) -> Dict:
        """Process a single user for personalization."""
        result = {
            'user_id': user_id,
            'status': 'skipped',
            'reason': '',
            'old_auc': None,
            'new_auc': None,
        }
        
        # Step 1: Check eligibility
        state_path = self.user_data_dir / user_id / "state.json"
        with open(state_path) as f:
            state = json.load(f)
        
        if state['days_logged'] < 30:
            result['reason'] = f"Not enough days ({state['days_logged']}/30)"
            print(f"   ⏭️ Skipped: {result['reason']}")
            return result
        
        # Check days logged this week
        logs_this_week = self._count_logs_this_week(user_id)
        if logs_this_week < 4:
            result['reason'] = f"Not enough logs this week ({logs_this_week}/4)"
            print(f"   ⏭️ Skipped: {result['reason']}")
            return result
        
        print(f"   ✓ Eligible: {state['days_logged']} days, {logs_this_week} this week")
        
        # Step 2: Prepare data
        train_data, val_data = self._prepare_user_data(user_id)
        
        if len(train_data['labels']) < 10:
            result['reason'] = "Not enough training sequences"
            print(f"   ⏭️ Skipped: {result['reason']}")
            return result
        
        print(f"   ✓ Data: {len(train_data['labels'])} train, {len(val_data['labels'])} val")
        
        # Step 3: Load generic model
        model = self._load_generic_model()
        
        # Step 4: Evaluate before training
        old_auc = self._evaluate(model, val_data)
        result['old_auc'] = old_auc
        print(f"   📊 Before: AUC = {old_auc:.4f}")
        
        # Step 5: Fine-tune
        model = self._fine_tune(model, train_data, val_data)
        
        # Step 6: Evaluate after training
        new_auc = self._evaluate(model, val_data)
        result['new_auc'] = new_auc
        print(f"   📊 After:  AUC = {new_auc:.4f}")
        
        # Step 7: Save if improved
        if new_auc > old_auc:
            self._save_user_model(user_id, model, state)
            result['status'] = 'updated'
            print(f"   ✅ Model updated! (+{(new_auc - old_auc)*100:.1f}%)")
        else:
            result['status'] = 'no_improvement'
            result['reason'] = "New model not better"
            print(f"   ⚠️ No improvement, keeping old model")
        
        return result
    
    def _count_logs_this_week(self, user_id: str) -> int:
        """Count how many days were logged in the last 7 days."""
        logs_dir = self.user_data_dir / user_id / "logs"
        if not logs_dir.exists():
            return 0
        
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        count = 0
        
        for log_file in logs_dir.glob("*.json"):
            date_str = log_file.stem
            if date_str >= week_ago:
                count += 1
        
        return count
    
    def _prepare_user_data(
        self,
        user_id: str,
        lookback_days: int = 60,
        val_days: int = 7,
    ) -> Tuple[Dict, Dict]:
        """
        Prepare sliding window sequences from user's logs.
        
        Returns:
            train_data: {'sequences': [...], 'labels': [...]}
            val_data: {'sequences': [...], 'labels': [...]}
        """
        logs_dir = self.user_data_dir / user_id / "logs"
        
        # Load all logs
        all_logs = {}
        for log_file in sorted(logs_dir.glob("*.json")):
            with open(log_file) as f:
                log = json.load(f)
                all_logs[log['date']] = log
        
        # Sort by date
        sorted_dates = sorted(all_logs.keys())
        
        # Only use last N days
        if len(sorted_dates) > lookback_days:
            sorted_dates = sorted_dates[-lookback_days:]
        
        # Create sliding windows
        sequences = []
        labels = []
        dates = []
        
        for i in range(14, len(sorted_dates)):
            window_dates = sorted_dates[i-14:i]
            target_date = sorted_dates[i]
            
            # Check if all days have logs and target has outcome
            target_log = all_logs.get(target_date)
            if target_log is None or target_log.get('attack_occurred') is None:
                continue
            
            # Create sequence
            seq = self._create_sequence_from_logs(
                [all_logs.get(d, {}) for d in window_dates]
            )
            
            sequences.append(seq)
            labels.append(1 if target_log['attack_occurred'] else 0)
            dates.append(target_date)
        
        # Split: last val_days for validation
        split_idx = len(sequences) - val_days
        
        train_data = {
            'sequences': sequences[:split_idx],
            'labels': labels[:split_idx],
        }
        
        val_data = {
            'sequences': sequences[split_idx:],
            'labels': labels[split_idx:],
        }
        
        return train_data, val_data
    
    def _create_sequence_from_logs(self, logs: List[Dict]) -> Dict:
        """Create a 14-day sequence from log entries."""
        default_continuous = [7.0, 5.0, 1013.0, 0.0, 22.0, 50.0, 4.0, 0.0]
        default_binary = [1, 1, 1, 0, 0, 1]
        
        continuous_seq = np.zeros((14, 8))
        binary_seq = np.zeros((14, 6))
        menstrual_seq = np.full(14, -1)
        
        for i, log in enumerate(logs):
            features = log.get('features', {})
            
            continuous_seq[i] = [
                features.get(f, default_continuous[j])
                for j, f in enumerate(self.continuous_features)
            ]
            binary_seq[i] = [
                features.get(f, default_binary[j])
                for j, f in enumerate(self.binary_features)
            ]
            menstrual_seq[i] = features.get('menstrual_cycle_day', -1)
        
        # Normalize
        continuous_seq = (continuous_seq - self.means) / self.stds
        
        # Clinical features
        clinical = np.zeros(8)
        if logs:
            last_features = logs[-1].get('features', {})
            clinical[6] = max(0, 7 - last_features.get('sleep_hours', 7)) / 3
            clinical[7] = last_features.get('stress_level', 5) / 10
        
        return {
            'continuous': continuous_seq.astype(np.float32),
            'binary': binary_seq.astype(np.float32),
            'menstrual': menstrual_seq.astype(np.int64),
            'day_of_week': (np.arange(14) % 7).astype(np.int64),
            'clinical': clinical.astype(np.float32),
        }
    
    def _load_generic_model(self) -> MigraineMamba:
        """Load the generic pre-trained model."""
        model = MigraineMamba(self.model_config)
        
        checkpoint = torch.load(
            self.generic_model_path,
            map_location=self.device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _fine_tune(
        self,
        model: MigraineMamba,
        train_data: Dict,
        val_data: Dict,
        epochs: int = 10,
        lr: float = 1e-5,
        batch_size: int = 16,
    ) -> MigraineMamba:
        """
        Fine-tune model on user's data.
        
        Strategy:
        - Freeze: Embedding + Mamba backbone (keep general patterns)
        - Train: Attention + prediction heads (adapt to user)
        """
        model.train()
        
        # Freeze backbone layers
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            # Freeze embedding and mamba layers
            if 'embedding' in name or 'mamba' in name.lower():
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        
        print(f"   🔒 Frozen: {frozen_count:,} params | 🔓 Trainable: {trainable_count:,}")
        
        # Create datasets
        train_dataset = UserDataset(train_data['sequences'], train_data['labels'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            shuffle=True,
        )
        
        # Optimizer (only for trainable params)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        
        # Loss
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                continuous = batch['continuous'].to(self.device)
                binary = batch['binary'].to(self.device)
                menstrual = batch['menstrual'].to(self.device)
                day_of_week = batch['day_of_week'].to(self.device)
                clinical = batch['clinical'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = model(continuous, binary, menstrual, day_of_week, clinical)
                loss = criterion(outputs['attack_logits'], targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 5 == 0:
                val_auc = self._evaluate(model, val_data)
                print(f"   Epoch {epoch+1}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}")
        
        model.eval()
        return model
    
    @torch.no_grad()
    def _evaluate(self, model: MigraineMamba, data: Dict) -> float:
        """Evaluate model on data, return AUC."""
        model.eval()
        
        if len(data['labels']) == 0:
            return 0.5
        
        dataset = UserDataset(data['sequences'], data['labels'])
        loader = DataLoader(dataset, batch_size=32)
        
        all_preds = []
        all_targets = []
        
        for batch in loader:
            continuous = batch['continuous'].to(self.device)
            binary = batch['binary'].to(self.device)
            menstrual = batch['menstrual'].to(self.device)
            day_of_week = batch['day_of_week'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            
            outputs = model(continuous, binary, menstrual, day_of_week, clinical)
            probs = torch.sigmoid(outputs['attack_logits'])
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(batch['target'].numpy())
        
        try:
            if len(set(all_targets)) < 2:
                return 0.5
            return roc_auc_score(all_targets, all_preds)
        except:
            return 0.5
    
    def _save_user_model(self, user_id: str, model: MigraineMamba, state: Dict):
        """Save personalized model for user."""
        model_dir = self.user_data_dir / user_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Increment week counter
        week = state.get('personalization_week', 0) + 1
        
        # Save model
        model_path = model_dir / f"week_{week:03d}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'week': week,
            'date': datetime.now().isoformat(),
        }, model_path)
        
        # Update state
        state['personalization_week'] = week
        state['model_version'] = f"personalized_week_{week}"
        
        state_path = self.user_data_dir / user_id / "state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Keep only last 4 versions
        model_files = sorted(model_dir.glob("week_*.pth"))
        for old_model in model_files[:-4]:
            old_model.unlink()
        
        print(f"   💾 Saved: {model_path.name}")


def main():
    """Run weekly personalization."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='../models_v2/mamba_finetuned_v2.pth')
    parser.add_argument('--user-data', '-u', default='../demo_users')
    args = parser.parse_args()
    
    personalizer = WeeklyPersonalizer(
        generic_model_path=args.model,
        user_data_dir=args.user_data,
    )
    
    personalizer.run_weekly_update()


if __name__ == "__main__":
    main()