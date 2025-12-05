"""
Final Validation Script - Realistic Scenarios
==============================================
Tests the model with realistic multi-day trigger patterns.

Key insight: Migraine triggers typically build up over 2-3 days,
not just the last day. This script tests with realistic patterns.

Author: Dhoka
Date: December 2025
"""

import torch
import json
import numpy as np
from pathlib import Path

from model import MigraineMamba, MigraineModelConfig


CONTINUOUS_FEATURES = [
    'sleep_hours', 'stress_level', 'barometric_pressure', 'pressure_change',
    'temperature', 'humidity', 'hours_fasting', 'alcohol_drinks',
]

BINARY_FEATURES = [
    'had_breakfast', 'had_lunch', 'had_dinner', 'had_snack',
    'bright_light_exposure', 'sleep_quality',
]

HEALTHY_DAY = {
    'sleep_hours': 7.5, 'stress_level': 3.0, 'barometric_pressure': 1013.0,
    'pressure_change': 0.0, 'temperature': 22.0, 'humidity': 50.0,
    'hours_fasting': 4.0, 'alcohol_drinks': 0.0,
    'had_breakfast': 1, 'had_lunch': 1, 'had_dinner': 1, 'had_snack': 0,
    'bright_light_exposure': 0, 'sleep_quality': 1,
}

BAD_DAY = {
    'sleep_hours': 4.5, 'stress_level': 8.0, 'barometric_pressure': 1005.0,
    'pressure_change': -8.0, 'temperature': 22.0, 'humidity': 70.0,
    'hours_fasting': 14.0, 'alcohol_drinks': 3.0,
    'had_breakfast': 0, 'had_lunch': 1, 'had_dinner': 1, 'had_snack': 0,
    'bright_light_exposure': 1, 'sleep_quality': 0,
}


def load_model(model_path: str, device: str):
    """Load model."""
    config = MigraineModelConfig(
        n_continuous_features=8, n_binary_features=6, seq_len=14,
        d_model=64, n_mamba_layers=2, dropout=0.3,
    )
    model = MigraineMamba(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def create_14day_sequence(day_pattern: list, scaler_mean=None, scaler_std=None):
    """
    Create a 14-day sequence from a pattern specification.
    
    day_pattern: list of 14 dicts, each specifying that day's values
    """
    if scaler_mean is None:
        scaler_mean = np.array([7.0, 5.0, 1013.0, 0.0, 20.0, 50.0, 6.0, 0.5])
    if scaler_std is None:
        scaler_std = np.array([1.5, 2.5, 10.0, 5.0, 10.0, 20.0, 4.0, 1.0])
    
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)
    
    continuous_seq = np.zeros((14, 8))
    binary_seq = np.zeros((14, 6))
    menstrual_seq = np.full(14, -1)
    day_of_week_seq = np.arange(14) % 7
    
    for i, day in enumerate(day_pattern):
        continuous_seq[i] = [day.get(f, HEALTHY_DAY[f]) for f in CONTINUOUS_FEATURES]
        binary_seq[i] = [day.get(f, HEALTHY_DAY[f]) for f in BINARY_FEATURES]
        menstrual_seq[i] = day.get('menstrual_cycle_day', -1)
    
    # Normalize
    continuous_seq = (continuous_seq - scaler_mean) / scaler_std
    
    # Clinical features (based on last few days)
    clinical = np.zeros(8)
    last_day = day_pattern[-1]
    
    # Trigger accumulation from last 3 days
    triggers = 0
    for day in day_pattern[-3:]:
        if day.get('sleep_hours', 7.5) < 6: triggers += 0.5
        if day.get('stress_level', 3) > 6: triggers += 0.5
        if day.get('alcohol_drinks', 0) > 1: triggers += 0.5
        if day.get('sleep_quality', 1) == 0: triggers += 0.5
    clinical[4] = min(triggers / 4.0, 1.0)
    
    clinical[6] = max(0, 7.0 - last_day.get('sleep_hours', 7.5)) / 3.0
    clinical[7] = last_day.get('stress_level', 3) / 10.0
    
    return {
        'continuous': continuous_seq.astype(np.float32),
        'binary': binary_seq.astype(np.float32),
        'menstrual': menstrual_seq.astype(np.int64),
        'day_of_week': day_of_week_seq.astype(np.int64),
        'clinical': clinical.astype(np.float32),
    }


def predict(model, sequence, device):
    """Get raw prediction."""
    continuous = torch.tensor(sequence['continuous']).unsqueeze(0).to(device)
    binary = torch.tensor(sequence['binary']).unsqueeze(0).to(device)
    menstrual = torch.tensor(sequence['menstrual']).unsqueeze(0).to(device)
    day_of_week = torch.tensor(sequence['day_of_week']).unsqueeze(0).to(device)
    clinical = torch.tensor(sequence['clinical']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(continuous, binary, menstrual, day_of_week, clinical)
        prob = torch.sigmoid(outputs['attack_logits']).item()
    
    return prob


def run_realistic_validation(model, device):
    """
    Test with REALISTIC multi-day scenarios.
    """
    print("\n" + "=" * 70)
    print("🧪 REALISTIC MULTI-DAY VALIDATION")
    print("=" * 70)
    print("\nKey insight: Triggers build up over days, not just the last day.\n")
    
    results = []
    
    # ============================================
    # Scenario 1: 14 healthy days (baseline)
    # ============================================
    pattern1 = [HEALTHY_DAY.copy() for _ in range(14)]
    seq1 = create_14day_sequence(pattern1)
    prob1 = predict(model, seq1, device)
    
    print("-" * 70)
    print("1️⃣  ALL HEALTHY (14 days)")
    print("-" * 70)
    print(f"   📊 Risk: {prob1*100:.1f}%")
    results.append(("All Healthy", prob1))
    
    # ============================================
    # Scenario 2: 13 healthy + 1 bad (last day only)
    # ============================================
    pattern2 = [HEALTHY_DAY.copy() for _ in range(13)] + [BAD_DAY.copy()]
    seq2 = create_14day_sequence(pattern2)
    prob2 = predict(model, seq2, device)
    
    print("\n" + "-" * 70)
    print("2️⃣  ONE BAD DAY (last day only)")
    print("-" * 70)
    print(f"   📊 Risk: {prob2*100:.1f}% (Δ = +{(prob2-prob1)*100:.1f}%)")
    results.append(("1 Bad Day (last)", prob2))
    
    # ============================================
    # Scenario 3: 11 healthy + 3 bad days (building up)
    # ============================================
    pattern3 = [HEALTHY_DAY.copy() for _ in range(11)]
    # Gradual worsening
    for i in range(3):
        bad = HEALTHY_DAY.copy()
        factor = (i + 1) / 3.0
        bad['sleep_hours'] = 7.5 - factor * 3.0  # 7.5 -> 6.5 -> 5.5 -> 4.5
        bad['stress_level'] = 3.0 + factor * 5.0  # 3 -> 4.7 -> 6.3 -> 8
        bad['alcohol_drinks'] = factor * 3.0
        bad['sleep_quality'] = 0 if factor > 0.5 else 1
        pattern3.append(bad)
    
    seq3 = create_14day_sequence(pattern3)
    prob3 = predict(model, seq3, device)
    
    print("\n" + "-" * 70)
    print("3️⃣  THREE BAD DAYS (gradual decline)")
    print("-" * 70)
    print("   Days 12-14: Progressively worse sleep, stress, alcohol")
    print(f"   📊 Risk: {prob3*100:.1f}% (Δ = +{(prob3-prob1)*100:.1f}%)")
    results.append(("3 Bad Days (gradual)", prob3))
    
    # ============================================
    # Scenario 4: 7 healthy + 7 bad (full week of triggers)
    # ============================================
    pattern4 = [HEALTHY_DAY.copy() for _ in range(7)] + [BAD_DAY.copy() for _ in range(7)]
    seq4 = create_14day_sequence(pattern4)
    prob4 = predict(model, seq4, device)
    
    print("\n" + "-" * 70)
    print("4️⃣  FULL WEEK BAD (days 8-14)")
    print("-" * 70)
    print("   7 days of: sleep 4.5h, stress 8, alcohol 3, poor sleep quality")
    print(f"   📊 Risk: {prob4*100:.1f}% (Δ = +{(prob4-prob1)*100:.1f}%)")
    results.append(("7 Bad Days", prob4))
    
    # ============================================
    # Scenario 5: All bad days (worst case)
    # ============================================
    pattern5 = [BAD_DAY.copy() for _ in range(14)]
    seq5 = create_14day_sequence(pattern5)
    prob5 = predict(model, seq5, device)
    
    print("\n" + "-" * 70)
    print("5️⃣  ALL BAD (14 days of triggers)")
    print("-" * 70)
    print("   Every day: sleep 4.5h, stress 8, alcohol 3, poor sleep quality")
    print(f"   📊 Risk: {prob5*100:.1f}% (Δ = +{(prob5-prob1)*100:.1f}%)")
    results.append(("All Bad (14 days)", prob5))
    
    # ============================================
    # Scenario 6: Menstrual trigger + stress
    # ============================================
    pattern6 = [HEALTHY_DAY.copy() for _ in range(14)]
    for i in range(12, 14):  # Last 2 days
        pattern6[i]['stress_level'] = 7.0
        pattern6[i]['sleep_hours'] = 6.0
        pattern6[i]['menstrual_cycle_day'] = i - 11  # Day 1, 2 of cycle
    
    seq6 = create_14day_sequence(pattern6)
    prob6 = predict(model, seq6, device)
    
    print("\n" + "-" * 70)
    print("6️⃣  MENSTRUAL + STRESS (last 2 days)")
    print("-" * 70)
    print("   Menstrual day 1-2 + elevated stress + reduced sleep")
    print(f"   📊 Risk: {prob6*100:.1f}% (Δ = +{(prob6-prob1)*100:.1f}%)")
    results.append(("Menstrual + Stress", prob6))
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("📋 SUMMARY - RAW MODEL PREDICTIONS")
    print("=" * 70)
    
    print("\n   Scenario                      Risk    vs Baseline")
    print("   " + "-" * 50)
    
    baseline = results[0][1]
    for name, prob in results:
        diff = prob - baseline
        bar_len = int(prob * 40)
        bar = "█" * bar_len
        indicator = "🔴" if prob > 0.5 else "🟡" if prob > 0.35 else "🟢"
        print(f"   {indicator} {name:<25} {prob*100:5.1f}%   {diff*100:+5.1f}%  |{bar}")
    
    # Spread analysis
    probs = [r[1] for r in results]
    spread = max(probs) - min(probs)
    
    print(f"\n   Prediction Range: {min(probs)*100:.1f}% - {max(probs)*100:.1f}%")
    print(f"   Spread: {spread*100:.1f}%")
    
    # Validation
    print("\n" + "-" * 70)
    print("   Validation Checks:")
    
    passed = 0
    
    if prob1 < 0.40:
        print(f"   ✅ Baseline under 40% ({prob1*100:.1f}%)")
        passed += 1
    else:
        print(f"   ⚠️ Baseline too high ({prob1*100:.1f}%)")
    
    if prob5 > prob1 + 0.10:
        print(f"   ✅ Worst case clearly higher (+{(prob5-prob1)*100:.1f}%)")
        passed += 1
    else:
        print(f"   ❌ Worst case not different enough (+{(prob5-prob1)*100:.1f}%)")
    
    if prob3 > prob2:
        print(f"   ✅ 3 bad days > 1 bad day (temporal pattern)")
        passed += 1
    else:
        print(f"   ⚠️ 3 bad days not > 1 bad day")
    
    if spread > 0.15:
        print(f"   ✅ Good prediction spread ({spread*100:.1f}%)")
        passed += 1
    else:
        print(f"   ⚠️ Narrow spread ({spread*100:.1f}%)")
    
    # Monotonicity check
    if prob1 < prob2 < prob3 < prob4 < prob5:
        print(f"   ✅ Monotonic increase with severity")
        passed += 1
    else:
        print(f"   ⚠️ Non-monotonic (some ordering issues)")
    
    print(f"\n   Tests Passed: {passed}/5")
    
    if passed >= 4:
        print("   🎉 MODEL VALIDATES WELL!")
    elif passed >= 3:
        print("   ✓ MODEL ACCEPTABLE")
    else:
        print("   ⚠️ MODEL NEEDS IMPROVEMENT")
    
    print("=" * 70)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='../models_v2/mamba_finetuned_v2.pth')
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    model = load_model(args.model, device)
    print(f"✓ Model loaded from {args.model}")
    
    run_realistic_validation(model, device)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()