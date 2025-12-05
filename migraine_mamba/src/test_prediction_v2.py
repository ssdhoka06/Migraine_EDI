"""
Improved Migraine Risk Prediction - Calibrated Version
=======================================================
Fixes:
1. Calibrated probability output (accounts for base rate)
2. More realistic test sequences (gradual trigger buildup)
3. Better risk stratification

Usage:
    python test_prediction_v2.py --quick

Author: Dhoka
Date: December 2025
"""

import torch
import json
import argparse
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

HEALTHY_DEFAULTS = {
    'sleep_hours': 7.5, 'stress_level': 3.0, 'barometric_pressure': 1013.0,
    'pressure_change': 0.0, 'temperature': 22.0, 'humidity': 50.0,
    'hours_fasting': 4.0, 'alcohol_drinks': 0.0,
    'had_breakfast': 1, 'had_lunch': 1, 'had_dinner': 1, 'had_snack': 0,
    'bright_light_exposure': 0, 'sleep_quality': 1,
    'menstrual_cycle_day': -1, 'day_of_week': 3,
}


def load_system(model_path: str, scaler_path: str, device: str):
    """Load model and scaler."""
    print(f"🔧 Loading system on {device}...")
    
    if Path(scaler_path).exists():
        with open(scaler_path, "r") as f:
            scaler_config = json.load(f)
    else:
        scaler_config = {
            'continuous_mean': [7.0, 5.0, 1013.0, 0.0, 20.0, 50.0, 6.0, 0.5],
            'continuous_std': [1.5, 2.5, 10.0, 5.0, 10.0, 20.0, 4.0, 1.0],
        }
    
    config = MigraineModelConfig(
        n_continuous_features=8, n_binary_features=6, seq_len=14,
        d_model=64, n_mamba_layers=2, dropout=0.3,
    )
    
    model = MigraineMamba(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded")
    return model, scaler_config


def create_realistic_sequence(
    current_day: dict, 
    scaler_config: dict, 
    history_type: str = "healthy",
    seq_len: int = 14
):
    """
    Create sequence with REALISTIC history patterns.
    
    history_type options:
    - "healthy": All good days (original behavior)
    - "gradual_decline": Progressively worsening (more realistic for triggers)
    - "mixed": Some good, some bad days
    - "stressed_week": Last 3-5 days with elevated stress
    """
    means = np.array(scaler_config.get('continuous_mean', [7.0, 5.0, 1013.0, 0.0, 20.0, 50.0, 6.0, 0.5]))
    stds = np.array(scaler_config.get('continuous_std', [1.5, 2.5, 10.0, 5.0, 10.0, 20.0, 4.0, 1.0]))
    stds = np.where(stds == 0, 1.0, stds)
    
    # Initialize with healthy values
    healthy_continuous = np.array([HEALTHY_DEFAULTS[f] for f in CONTINUOUS_FEATURES])
    healthy_binary = np.array([HEALTHY_DEFAULTS[f] for f in BINARY_FEATURES])
    
    continuous_seq = np.tile(healthy_continuous, (seq_len, 1))
    binary_seq = np.tile(healthy_binary, (seq_len, 1))
    menstrual_seq = np.full(seq_len, current_day.get('menstrual_cycle_day', -1))
    day_of_week_seq = np.arange(seq_len) % 7
    
    # Apply history pattern
    if history_type == "gradual_decline":
        # Last 3 days get progressively worse
        for i in range(3):
            day_idx = seq_len - 3 + i
            factor = (i + 1) / 3.0  # 0.33, 0.67, 1.0
            
            # Interpolate between healthy and current
            for j, feat in enumerate(CONTINUOUS_FEATURES):
                target = current_day.get(feat, healthy_continuous[j])
                continuous_seq[day_idx, j] = healthy_continuous[j] + factor * (target - healthy_continuous[j])
            
            for j, feat in enumerate(BINARY_FEATURES):
                if current_day.get(feat, healthy_binary[j]) != healthy_binary[j]:
                    if factor > 0.5:
                        binary_seq[day_idx, j] = current_day.get(feat, healthy_binary[j])
    
    elif history_type == "stressed_week":
        # Last 5 days with elevated stress
        for i in range(5):
            day_idx = seq_len - 5 + i
            continuous_seq[day_idx, 1] = 5.0 + i * 0.5  # Stress: 5, 5.5, 6, 6.5, 7
            continuous_seq[day_idx, 0] = 7.0 - i * 0.3  # Sleep declining
    
    elif history_type == "mixed":
        # Random variation
        np.random.seed(42)
        for i in range(seq_len - 1):
            continuous_seq[i, 0] += np.random.normal(0, 0.5)  # Sleep variation
            continuous_seq[i, 1] += np.random.normal(0, 1.0)  # Stress variation
            continuous_seq[i, 1] = np.clip(continuous_seq[i, 1], 1, 10)
    
    # Always set last day to current input
    continuous_seq[-1] = np.array([current_day.get(f, healthy_continuous[i]) for i, f in enumerate(CONTINUOUS_FEATURES)])
    binary_seq[-1] = np.array([current_day.get(f, healthy_binary[i]) for i, f in enumerate(BINARY_FEATURES)])
    
    # Normalize
    continuous_seq = (continuous_seq - means) / stds
    
    # Clinical features
    clinical = compute_clinical_features(current_day, menstrual_seq[-1])
    
    return {
        'continuous': continuous_seq.astype(np.float32),
        'binary': binary_seq.astype(np.float32),
        'menstrual': menstrual_seq.astype(np.int64),
        'day_of_week': day_of_week_seq.astype(np.int64),
        'clinical': clinical.astype(np.float32),
    }


def compute_clinical_features(current_day: dict, menstrual_day: int) -> np.ndarray:
    """Compute clinical features with trigger accumulation."""
    clinical = np.zeros(8)
    
    clinical[0] = 0  # refractory
    clinical[1] = 1 if menstrual_day in [0, 1, 2, 26, 27] else 0
    clinical[2] = 0  # attacks_3d
    clinical[3] = 0  # attacks_7d
    
    # Trigger accumulation - MORE SENSITIVE
    triggers = 0
    if current_day.get('sleep_hours', 7.5) < 6: triggers += 1.5
    if current_day.get('sleep_hours', 7.5) < 5: triggers += 1.0  # Extra penalty
    if current_day.get('stress_level', 3) > 6: triggers += 1.5
    if current_day.get('stress_level', 3) > 8: triggers += 1.0  # Extra penalty
    if abs(current_day.get('pressure_change', 0)) > 5: triggers += 1.0
    if current_day.get('hours_fasting', 4) > 10: triggers += 1.0
    if current_day.get('alcohol_drinks', 0) > 2: triggers += 1.5
    if current_day.get('sleep_quality', 1) == 0: triggers += 1.0
    if current_day.get('bright_light_exposure', 0) == 1: triggers += 0.5
    
    clinical[4] = min(triggers / 8.0, 1.0)  # Normalized, capped at 1
    
    clinical[5] = 1 if current_day.get('day_of_week', 3) in [5, 6] else 0
    clinical[6] = max(0, 7.0 - current_day.get('sleep_hours', 7.5)) / 3.0
    clinical[7] = current_day.get('stress_level', 3) / 10.0
    
    return clinical


def predict_calibrated(model, sequence: dict, device: str, base_rate: float = 0.17) -> dict:
    """
    Predict with calibrated output.
    
    The model was trained with pos_weight=4.84, which shifts predictions.
    We recalibrate to give more interpretable probabilities.
    """
    continuous = torch.tensor(sequence['continuous']).unsqueeze(0).to(device)
    binary = torch.tensor(sequence['binary']).unsqueeze(0).to(device)
    menstrual = torch.tensor(sequence['menstrual']).unsqueeze(0).to(device)
    day_of_week = torch.tensor(sequence['day_of_week']).unsqueeze(0).to(device)
    clinical = torch.tensor(sequence['clinical']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(continuous, binary, menstrual, day_of_week, clinical)
        logit = outputs['attack_logits'].item()
        raw_prob = torch.sigmoid(outputs['attack_logits']).item()
    
    # Calibration: map raw probability to calibrated probability
    # Using Platt scaling approximation based on base rate
    # This shifts the "neutral" point from 0.5 to base_rate
    
    # Method: Use logit difference from base rate
    base_logit = np.log(base_rate / (1 - base_rate))  # ~ -1.58 for 17%
    
    # Calibrated logit: shift and scale
    calibrated_logit = (logit - 0.0) * 1.5  # Amplify differences
    calibrated_prob = 1 / (1 + np.exp(-calibrated_logit))
    
    # Alternative: Risk ratio approach
    # How much higher/lower than base rate?
    risk_ratio = raw_prob / 0.5  # ratio vs neutral point
    risk_based_prob = base_rate * risk_ratio
    risk_based_prob = np.clip(risk_based_prob, 0.01, 0.99)
    
    return {
        'raw_probability': raw_prob,
        'calibrated_probability': calibrated_prob,
        'risk_ratio_probability': risk_based_prob,
        'logit': logit,
    }


def run_validation_v2(model, scaler_config, device):
    """
    Improved validation with:
    1. Calibrated probabilities
    2. Realistic history sequences
    3. Better differentiation
    """
    
    print("\n" + "=" * 70)
    print("🧪 IMPROVED VALIDATION SCENARIOS (v2)")
    print("=" * 70)
    
    BASE_RATE = 0.17  # 17% attack rate in training data
    
    # ========================================
    # SCENARIO 1: Perfect Health
    # ========================================
    print("\n" + "-" * 70)
    print("1️⃣  PERFECT HEALTH BASELINE")
    print("-" * 70)
    
    scenario1 = {**HEALTHY_DEFAULTS}
    
    # Test with different history types
    print("\n   Testing with different history patterns:")
    
    for history_type in ["healthy", "mixed"]:
        seq = create_realistic_sequence(scenario1, scaler_config, history_type)
        pred = predict_calibrated(model, seq, device, BASE_RATE)
        print(f"   • {history_type:15} → Raw: {pred['raw_probability']*100:5.1f}% | "
              f"Calibrated: {pred['calibrated_probability']*100:5.1f}% | "
              f"Risk-based: {pred['risk_ratio_probability']*100:5.1f}%")
    
    # Use healthy history as baseline
    seq = create_realistic_sequence(scenario1, scaler_config, "healthy")
    baseline_pred = predict_calibrated(model, seq, device, BASE_RATE)
    baseline_raw = baseline_pred['raw_probability']
    baseline_cal = baseline_pred['risk_ratio_probability']
    
    print(f"\n   📊 BASELINE: {baseline_cal*100:.1f}% (calibrated)")
    
    # ========================================
    # SCENARIO 2: Lifestyle Disaster
    # ========================================
    print("\n" + "-" * 70)
    print("2️⃣  LIFESTYLE DISASTER (with gradual buildup)")
    print("-" * 70)
    print("   Sleep: 4h | Stress: 9 | Alcohol: 5 | Poor sleep quality")
    print("   History: 3-day gradual decline (more realistic)")
    
    scenario2 = {
        **HEALTHY_DEFAULTS,
        'sleep_hours': 4.0,
        'stress_level': 9.0,
        'alcohol_drinks': 5.0,
        'sleep_quality': 0,
        'had_breakfast': 0,
    }
    
    # Compare history types
    print("\n   With different history patterns:")
    for history_type in ["healthy", "gradual_decline", "stressed_week"]:
        seq = create_realistic_sequence(scenario2, scaler_config, history_type)
        pred = predict_calibrated(model, seq, device, BASE_RATE)
        print(f"   • {history_type:15} → Raw: {pred['raw_probability']*100:5.1f}% | "
              f"Calibrated: {pred['calibrated_probability']*100:5.1f}% | "
              f"Risk-based: {pred['risk_ratio_probability']*100:5.1f}%")
    
    # Use gradual decline (most realistic)
    seq = create_realistic_sequence(scenario2, scaler_config, "gradual_decline")
    disaster_pred = predict_calibrated(model, seq, device, BASE_RATE)
    disaster_raw = disaster_pred['raw_probability']
    disaster_cal = disaster_pred['risk_ratio_probability']
    
    print(f"\n   📊 DISASTER: {disaster_cal*100:.1f}% (calibrated)")
    print(f"   📈 vs Baseline: +{(disaster_cal - baseline_cal)*100:.1f}% | "
          f"Ratio: {disaster_cal/baseline_cal:.2f}x")
    
    # ========================================
    # SCENARIO 3: Subtle Warning Signs
    # ========================================
    print("\n" + "-" * 70)
    print("3️⃣  SUBTLE WARNING SIGNS")
    print("-" * 70)
    print("   Normal sleep duration but POOR quality")
    print("   Slight stress elevation + Weather change")
    
    scenario3 = {
        **HEALTHY_DEFAULTS,
        'sleep_hours': 7.5,
        'sleep_quality': 0,
        'stress_level': 5.0,
        'pressure_change': -8.0,
    }
    
    seq = create_realistic_sequence(scenario3, scaler_config, "mixed")
    subtle_pred = predict_calibrated(model, seq, device, BASE_RATE)
    subtle_raw = subtle_pred['raw_probability']
    subtle_cal = subtle_pred['risk_ratio_probability']
    
    print(f"\n   📊 SUBTLE: {subtle_cal*100:.1f}% (calibrated)")
    print(f"   📈 vs Baseline: +{(subtle_cal - baseline_cal)*100:.1f}%")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY (Calibrated)")
    print("=" * 70)
    
    print(f"""
    Using Risk-Ratio Calibration (base rate = {BASE_RATE*100:.0f}%)
    
    Scenario                 Raw      Calibrated   vs Baseline
    ──────────────────────────────────────────────────────────
    1. Perfect Health       {baseline_raw*100:5.1f}%     {baseline_cal*100:5.1f}%      (baseline)
    2. Lifestyle Disaster   {disaster_raw*100:5.1f}%     {disaster_cal*100:5.1f}%      +{(disaster_cal-baseline_cal)*100:.1f}% ({disaster_cal/baseline_cal:.1f}x)
    3. Subtle Warning       {subtle_raw*100:5.1f}%     {subtle_cal*100:5.1f}%      +{(subtle_cal-baseline_cal)*100:.1f}% ({subtle_cal/baseline_cal:.1f}x)
    """)
    
    # Visual comparison
    print("    Risk Distribution (Calibrated):")
    max_risk = max(baseline_cal, disaster_cal, subtle_cal)
    scale = 40 / max_risk if max_risk > 0 else 1
    
    for name, risk in [("Healthy", baseline_cal), ("Disaster", disaster_cal), ("Subtle", subtle_cal)]:
        bar = "█" * int(risk * scale)
        level = "🔴" if risk > 0.4 else "🟡" if risk > 0.25 else "🟢"
        print(f"    {level} {name:10} {risk*100:5.1f}% |{bar}")
    
    # Validation checks
    print("\n" + "-" * 70)
    print("    Validation Checks:")
    
    checks_passed = 0
    
    # Check 1: Baseline should be near base rate
    if baseline_cal < 0.30:
        print(f"    ✅ Baseline reasonable ({baseline_cal*100:.1f}% < 30%)")
        checks_passed += 1
    else:
        print(f"    ⚠️ Baseline still high ({baseline_cal*100:.1f}%)")
    
    # Check 2: Disaster should be significantly higher
    if disaster_cal > baseline_cal * 1.5:
        print(f"    ✅ Disaster shows clear increase ({disaster_cal/baseline_cal:.1f}x baseline)")
        checks_passed += 1
    else:
        print(f"    ❌ Disaster increase too small ({disaster_cal/baseline_cal:.1f}x)")
    
    # Check 3: Subtle should be between baseline and disaster
    if baseline_cal < subtle_cal < disaster_cal:
        print(f"    ✅ Subtle risk properly ordered")
        checks_passed += 1
    else:
        print(f"    ⚠️ Subtle risk ordering issue")
    
    # Check 4: Good spread
    spread = disaster_cal - baseline_cal
    if spread > 0.15:
        print(f"    ✅ Good risk spread ({spread*100:.1f}%)")
        checks_passed += 1
    else:
        print(f"    ⚠️ Risk spread narrow ({spread*100:.1f}%)")
    
    print(f"\n    Checks Passed: {checks_passed}/4")
    print("=" * 70)
    
    return {
        'baseline_raw': baseline_raw,
        'baseline_calibrated': baseline_cal,
        'disaster_raw': disaster_raw,
        'disaster_calibrated': disaster_cal,
        'subtle_raw': subtle_raw,
        'subtle_calibrated': subtle_cal,
    }


def run_trigger_sensitivity(model, scaler_config, device):
    """Test sensitivity to individual triggers."""
    
    print("\n" + "=" * 70)
    print("🎯 TRIGGER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nTesting how much each trigger affects risk (isolated)\n")
    
    BASE_RATE = 0.17
    
    # Baseline
    baseline = {**HEALTHY_DEFAULTS}
    seq = create_realistic_sequence(baseline, scaler_config, "healthy")
    baseline_pred = predict_calibrated(model, seq, device, BASE_RATE)
    baseline_risk = baseline_pred['risk_ratio_probability']
    
    # Individual triggers
    triggers = {
        "Sleep 4h": {'sleep_hours': 4.0},
        "Sleep 5h": {'sleep_hours': 5.0},
        "Sleep 6h": {'sleep_hours': 6.0},
        "Stress 6": {'stress_level': 6.0},
        "Stress 8": {'stress_level': 8.0},
        "Stress 10": {'stress_level': 10.0},
        "Alcohol 2": {'alcohol_drinks': 2.0},
        "Alcohol 4": {'alcohol_drinks': 4.0},
        "Poor sleep quality": {'sleep_quality': 0},
        "Weather -5hPa": {'pressure_change': -5.0},
        "Weather -10hPa": {'pressure_change': -10.0},
        "Menstrual day 1": {'menstrual_cycle_day': 1},
        "Bright light": {'bright_light_exposure': 1},
        "Fasting 12h": {'hours_fasting': 12.0},
    }
    
    results = []
    
    for name, changes in triggers.items():
        scenario = {**HEALTHY_DEFAULTS, **changes}
        seq = create_realistic_sequence(scenario, scaler_config, "healthy")
        pred = predict_calibrated(model, seq, device, BASE_RATE)
        risk = pred['risk_ratio_probability']
        diff = risk - baseline_risk
        ratio = risk / baseline_risk if baseline_risk > 0 else 1
        results.append((name, risk, diff, ratio))
    
    # Sort by impact
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Baseline risk: {baseline_risk*100:.1f}%\n")
    print(f"{'Trigger':<22} {'Risk':>8} {'Change':>10} {'Ratio':>8}")
    print("-" * 50)
    
    for name, risk, diff, ratio in results:
        indicator = "🔺" if diff > 0.02 else "➖" if diff > -0.02 else "🔻"
        print(f"{indicator} {name:<20} {risk*100:6.1f}%  {diff*100:+7.1f}%   {ratio:6.2f}x")
    
    print("\n" + "=" * 70)
    
    # Identify strongest triggers
    print("\n📊 Strongest Triggers (by impact):")
    for name, risk, diff, ratio in results[:5]:
        bar = "█" * int(diff * 100)
        print(f"   {name:<22} +{diff*100:.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='../models/mamba_finetuned.pth')
    parser.add_argument('--scaler', '-s', default='/Users/sachidhoka/Desktop/Migraine_EDI/processed/scaler_config.json')
    parser.add_argument('--quick', '-q', action='store_true', help='Run improved validation')
    parser.add_argument('--sensitivity', action='store_true', help='Run trigger sensitivity analysis')
    
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    model, scaler_config = load_system(args.model, args.scaler, device)
    
    if args.quick:
        run_validation_v2(model, scaler_config, device)
    elif args.sensitivity:
        run_trigger_sensitivity(model, scaler_config, device)
    else:
        # Run both
        run_validation_v2(model, scaler_config, device)
        run_trigger_sensitivity(model, scaler_config, device)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()