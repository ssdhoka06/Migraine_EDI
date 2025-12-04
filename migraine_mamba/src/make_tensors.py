"""
Data Preprocessing - Make Tensors
=================================
Converts synthetic migraine CSV data into PyTorch tensors
for training the MigraineMamba model.

Creates:
- 14-day sliding window sequences
- Normalized continuous features
- Binary features
- Clinical knowledge features
- Train/validation/test splits (by patient, no leakage)

Usage:
    python make_tensors.py --data data/migraine_synthetic.csv --output processed

Author: Dhoka
Date: December 2025
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    
    seq_len: int = 14
    pred_horizon: int = 1
    test_size: float = 0.1
    val_size: float = 0.1
    
    # Feature columns (matching synthetic data generator)
    continuous_features: List[str] = field(default_factory=lambda: [
        'sleep_hours',
        'stress_level',
        'barometric_pressure',
        'pressure_change',
        'temperature',
        'humidity',
        'hours_fasting',
        'alcohol_drinks',
    ])
    
    binary_features: List[str] = field(default_factory=lambda: [
        'had_breakfast',
        'had_lunch',
        'had_dinner',
        'had_snack',
        'bright_light_exposure',
        'sleep_quality',
    ])


class MigraineDataProcessor:
    """Processes raw synthetic migraine data into model-ready tensors."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.scaler = StandardScaler()
        self.scaler_fitted = False
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load raw CSV data."""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df):,} records from {df['patient_id'].nunique():,} patients")
        
        # Print column info
        print(f"\nAvailable columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns[:20]):
            print(f"  {col}")
        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more")
        
        return df
    
    def _compute_clinical_features(
        self,
        patient_df: pd.DataFrame,
        seq_end_idx: int
    ) -> np.ndarray:
        """
        Compute engineered clinical features for a sequence.
        
        Features (8 total):
        1. Refractory flag (attack in last 48h)
        2. Menstrual high-risk (days 0-1 or 26-27)
        3. Attack count last 7 days (normalized)
        4. Attack count last 14 days (normalized)
        5. Trigger accumulation (sleep dep + stress)
        6. Weekend flag
        7. Sleep deficit accumulation
        8. Stress trend
        """
        seq_start = max(0, seq_end_idx - self.config.seq_len + 1)
        seq_data = patient_df.iloc[seq_start:seq_end_idx + 1]
        extended_start = max(0, seq_end_idx - 14)
        extended_data = patient_df.iloc[extended_start:seq_end_idx + 1]
        
        features = np.zeros(8, dtype=np.float32)
        
        # 1. Refractory flag
        last_2_days = seq_data.tail(2)
        if 'attack' in last_2_days.columns:
            features[0] = float(last_2_days['attack'].sum() > 0)
        
        # 2. Menstrual high-risk phase
        if 'menstrual_cycle_day' in seq_data.columns:
            last_cycle_day = seq_data['menstrual_cycle_day'].iloc[-1]
            if last_cycle_day >= 0:
                high_risk = last_cycle_day in [0, 1, 26, 27]
                features[1] = float(high_risk)
        
        # 3. Attack count last 7 days
        last_7 = seq_data.tail(7)
        if 'attack' in last_7.columns:
            features[2] = last_7['attack'].sum() / 7.0
        
        # 4. Attack count last 14 days
        if 'attack' in extended_data.columns:
            features[3] = extended_data['attack'].sum() / 14.0
        
        # 5. Trigger accumulation
        last_3 = seq_data.tail(3)
        low_sleep_days = 0
        high_stress_days = 0
        if 'sleep_hours' in last_3.columns:
            low_sleep_days = (last_3['sleep_hours'] < 6).sum()
        if 'stress_level' in last_3.columns:
            high_stress_days = (last_3['stress_level'] > 7).sum()
        features[4] = (low_sleep_days + high_stress_days) / 6.0
        
        # 6. Weekend flag
        if 'day_of_week' in seq_data.columns:
            dow = seq_data['day_of_week'].iloc[-1]
            features[5] = float(dow in [5, 6])
        
        # 7. Sleep deficit
        if 'sleep_hours' in last_7.columns:
            avg_sleep = last_7['sleep_hours'].mean()
            features[6] = max(-1, min(1, (avg_sleep - 7) / 2))
        
        # 8. Stress trend
        if 'stress_level' in last_7.columns and len(last_7) >= 2:
            first_half = last_7['stress_level'].iloc[:len(last_7)//2].mean()
            second_half = last_7['stress_level'].iloc[len(last_7)//2:].mean()
            features[7] = (second_half - first_half) / 5.0
        
        return features
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
        verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Create sliding window sequences from patient data."""
        if verbose:
            print("\nCreating sequences...")
        
        all_continuous = []
        all_binary = []
        all_menstrual = []
        all_dow = []
        all_clinical = []
        all_targets = []
        all_patient_ids = []
        
        patient_ids = df['patient_id'].unique()
        iterator = tqdm(patient_ids, desc="Processing patients") if verbose else patient_ids
        
        for patient_id in iterator:
            patient_df = df[df['patient_id'] == patient_id].sort_values('day').reset_index(drop=True)
            n_days = len(patient_df)
            
            if n_days < self.config.seq_len + self.config.pred_horizon:
                continue
            
            for end_idx in range(self.config.seq_len - 1, n_days - self.config.pred_horizon):
                start_idx = end_idx - self.config.seq_len + 1
                seq_data = patient_df.iloc[start_idx:end_idx + 1]
                
                # Continuous features
                cont_cols = [c for c in self.config.continuous_features if c in seq_data.columns]
                continuous = seq_data[cont_cols].values.astype(np.float32)
                
                # Pad if missing
                if len(cont_cols) < len(self.config.continuous_features):
                    pad_width = len(self.config.continuous_features) - len(cont_cols)
                    continuous = np.pad(continuous, ((0, 0), (0, pad_width)))
                
                # Binary features
                bin_cols = [c for c in self.config.binary_features if c in seq_data.columns]
                binary = seq_data[bin_cols].values.astype(np.float32)
                
                if len(bin_cols) < len(self.config.binary_features):
                    pad_width = len(self.config.binary_features) - len(bin_cols)
                    binary = np.pad(binary, ((0, 0), (0, pad_width)))
                
                # Menstrual cycle day
                if 'menstrual_cycle_day' in seq_data.columns:
                    menstrual = seq_data['menstrual_cycle_day'].values.astype(np.int32)
                else:
                    menstrual = np.full(self.config.seq_len, -1, dtype=np.int32)
                
                # Day of week
                if 'day_of_week' in seq_data.columns:
                    dow = seq_data['day_of_week'].values.astype(np.int32)
                else:
                    dow = np.zeros(self.config.seq_len, dtype=np.int32)
                
                # Clinical features
                clinical = self._compute_clinical_features(patient_df, end_idx)
                
                # Target
                if include_target:
                    target_idx = end_idx + self.config.pred_horizon
                    target = patient_df.iloc[target_idx]['attack']
                else:
                    target = 0
                
                all_continuous.append(continuous)
                all_binary.append(binary)
                all_menstrual.append(menstrual)
                all_dow.append(dow)
                all_clinical.append(clinical)
                all_targets.append(target)
                all_patient_ids.append(patient_id)
        
        if verbose:
            print(f"✓ Created {len(all_targets):,} sequences")
        
        features = {
            'continuous': np.array(all_continuous),
            'binary': np.array(all_binary),
            'menstrual': np.array(all_menstrual),
            'day_of_week': np.array(all_dow),
            'clinical': np.array(all_clinical),
            'patient_ids': np.array(all_patient_ids),
        }
        
        return features, np.array(all_targets)
    
    def normalize_features(
        self,
        features: Dict[str, np.ndarray],
        fit: bool = True
    ) -> Dict[str, np.ndarray]:
        """Normalize continuous features."""
        continuous = features['continuous']
        original_shape = continuous.shape
        
        flat = continuous.reshape(-1, original_shape[-1])
        
        # Handle NaN
        nan_mask = np.isnan(flat)
        flat[nan_mask] = 0
        
        if fit:
            self.scaler.fit(flat)
            self.scaler_fitted = True
        
        normalized = self.scaler.transform(flat)
        normalized[nan_mask] = 0
        
        features['continuous'] = normalized.reshape(original_shape)
        return features
    
    def split_data(
        self,
        features: Dict[str, np.ndarray],
        targets: np.ndarray,
    ) -> Tuple:
        """Split by patient to prevent data leakage."""
        patient_ids = features['patient_ids']
        unique_patients = np.unique(patient_ids)
        
        train_val_patients, test_patients = train_test_split(
            unique_patients,
            test_size=self.config.test_size,
            random_state=42
        )
        
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=self.config.val_size / (1 - self.config.test_size),
            random_state=42
        )
        
        def get_split(patients):
            mask = np.isin(patient_ids, patients)
            split_features = {k: v[mask] for k, v in features.items()}
            return split_features, targets[mask]
        
        train_f, train_t = get_split(train_patients)
        val_f, val_t = get_split(val_patients)
        test_f, test_t = get_split(test_patients)
        
        print(f"\nData split (by patient, no leakage):")
        print(f"  Train: {len(train_t):,} sequences ({len(train_patients):,} patients)")
        print(f"  Val:   {len(val_t):,} sequences ({len(val_patients):,} patients)")
        print(f"  Test:  {len(test_t):,} sequences ({len(test_patients):,} patients)")
        
        return (train_f, train_t), (val_f, val_t), (test_f, test_t)
    
    def to_tensors(
        self,
        features: Dict[str, np.ndarray],
        targets: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return {
            'continuous': torch.tensor(features['continuous'], dtype=torch.float32),
            'binary': torch.tensor(features['binary'], dtype=torch.float32),
            'menstrual': torch.tensor(features['menstrual'], dtype=torch.long),
            'day_of_week': torch.tensor(features['day_of_week'], dtype=torch.long),
            'clinical': torch.tensor(features['clinical'], dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
        }
    
    def save_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        output_path: str,
        prefix: str = "train"
    ):
        """Save tensors to disk."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_path) / f"{prefix}_tensors.pt"
        torch.save(tensors, save_path)
        print(f"✓ Saved {prefix} tensors to {save_path}")
    
    def save_scaler_config(self, output_path: str):
        """Save scaler parameters."""
        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted")
        
        config = {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist(),
            'feature_names': self.config.continuous_features,
        }
        
        save_path = Path(output_path) / "scaler_config.json"
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved scaler config to {save_path}")
    
    def process_and_save(
        self,
        data_path: str,
        output_path: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Complete pipeline."""
        # Load
        df = self.load_data(data_path)
        
        # Create sequences
        features, targets = self.create_sequences(df)
        
        # Normalize
        features = self.normalize_features(features, fit=True)
        
        # Split
        (train_f, train_t), (val_f, val_t), (test_f, test_t) = self.split_data(
            features, targets
        )
        
        # Convert to tensors
        train_tensors = self.to_tensors(train_f, train_t)
        val_tensors = self.to_tensors(val_f, val_t)
        test_tensors = self.to_tensors(test_f, test_t)
        
        # Save
        self.save_tensors(train_tensors, output_path, "train")
        self.save_tensors(val_tensors, output_path, "val")
        self.save_tensors(test_tensors, output_path, "test")
        self.save_scaler_config(output_path)
        
        # Print class balance
        print(f"\nClass balance (attack rate):")
        print(f"  Train: {train_t.mean():.2%}")
        print(f"  Val:   {val_t.mean():.2%}")
        print(f"  Test:  {test_t.mean():.2%}")
        
        return {
            'train': train_tensors,
            'val': val_tensors,
            'test': test_tensors,
        }


class MigraineDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for MigraineMamba."""
    
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.continuous = tensors['continuous']
        self.binary = tensors['binary']
        self.menstrual = tensors['menstrual']
        self.day_of_week = tensors['day_of_week']
        self.clinical = tensors['clinical']
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


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,  # MPS compatibility
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create DataLoaders from saved tensors."""
    train_tensors = torch.load(Path(data_dir) / "train_tensors.pt")
    val_tensors = torch.load(Path(data_dir) / "val_tensors.pt")
    test_tensors = torch.load(Path(data_dir) / "test_tensors.pt")
    
    train_dataset = MigraineDataset(train_tensors)
    val_dataset = MigraineDataset(val_tensors)
    test_dataset = MigraineDataset(test_tensors)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process migraine data into tensors")
    parser.add_argument(
        "--data", "-d",
        default="data/migraine_synthetic.csv",
        help="Path to synthetic data CSV"
    )
    parser.add_argument(
        "--output", "-o",
        default="processed",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    processor = MigraineDataProcessor()
    all_tensors = processor.process_and_save(args.data, args.output)
    
    print("\n" + "=" * 60)
    print("Data processing complete!")
    print("=" * 60)
    print(f"\nTensor shapes:")
    for split_name, tensors in all_tensors.items():
        print(f"\n{split_name}:")
        for key, tensor in tensors.items():
            print(f"  {key}: {tensor.shape}")