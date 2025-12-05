"""
Validation Script - Matching Exact Data Generation Constraints
================================================================
Tests using the EXACT trigger thresholds from the synthetic data generation.

Based on peer-reviewed constraints:
- Sleep deprivation: <6 hours → OR 3.98
- Menstrual days 0-1: OR 2.04
- Alcohol 5+ drinks: OR 2.08
- Stress DECLINE: OR 1.92 (not high stress!)
- Barometric pressure drop >10mb: OR 1.27
- Nighttime snack: OR 0.60 (protective)
- Refractory period: 48-72 hours (no attack after recent attack)

Author: Dhoka
Date: December 2025
"""

import torch
import numpy as np
from pathlib import Path

from model import MigraineMamba, MigraineModelConfig


# Feature definitions matching your data generation
CONTINUOUS_FEATURES = [
    'sleep_hours',           # Normal(7.0, 1.5), trigger <6h
    'stress_level',          # 1-10, trigger is DECLINE not high
    'barometric_pressure',   # ~1013 hPa
    'pressure_change',       # Trigger: drop >10 mb
    'temperature',
    'humidity',
    'hours_fasting',         # Trigger: >6 hours
    'alcohol_drinks',        # Trigger: 5+ drinks
]

BINARY_FEATURES = [
    'had_breakfast',
    'had_lunch', 
    'had_dinner',
    'had_snack',             # Nighttime snack is PROTECTIVE (OR 0.60)
    'bright_light_exposure',
    'sleep_quality',
]


def load_model(model_path: str, device: str):
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


def create_day(
    sleep_hours=7.0,
    stress_level=5.0,
    barometric_pressure=1013.0,
    pressure_change=0.0,
    temperature=22.0,
    humidity=50.0,
    hours_fasting=4.0,
    alcohol_drinks=0.0,
    had_breakfast=1,
    had_lunch=1,
    had_dinner=1,
    had_snack=0,
    bright_light_exposure=0,
    sleep_quality=1,
    menstrual_cycle_day=-1,
):
    return {
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'barometric_pressure': barometric_pressure,
        'pressure_change': pressure_change,
        'temperature': temperature,
        'humidity': humidity,
        'hours_fasting': hours_fasting,
        'alcohol_drinks': alcohol_drinks,
        'had_breakfast': had_breakfast,
        'had_lunch': had_lunch,
        'had_dinner': had_dinner,
        'had_snack': had_snack,
        'bright_light_exposure': bright_light_exposure,
        'sleep_quality': sleep_quality,
        'menstrual_cycle_day': menstrual_cycle_day,
    }


def create_sequence(day_list, scaler_mean=None, scaler_std=None):
    """Create 14-day sequence from list of day dicts."""
    if scaler_mean is None:
        scaler_mean = np.array([7.0, 5.0, 1013.0, 0.0, 20.0, 50.0, 6.0, 0.5])
    if scaler_std is None:
        scaler_std = np.array([1.5, 2.5, 10.0, 5.0, 10.0, 20.0, 4.0, 1.0])
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)
    
    continuous_seq = np.zeros((14, 8))
    binary_seq = np.zeros((14, 6))
    menstrual_seq = np.zeros(14, dtype=np.int64)
    
    for i, day in enumerate(day_list):
        continuous_seq[i] = [
            day['sleep_hours'], day['stress_level'], day['barometric_pressure'],
            day['pressure_change'], day['temperature'], day['humidity'],
            day['hours_fasting'], day['alcohol_drinks']
        ]
        binary_seq[i] = [
            day['had_breakfast'], day['had_lunch'], day['had_dinner'],
            day['had_snack'], day['bright_light_exposure'], day['sleep_quality']
        ]
        menstrual_seq[i] = day.get('menstrual_cycle_day', -1)
    
    continuous_seq = (continuous_seq - scaler_mean) / scaler_std
    
    # Clinical features
    clinical = np.zeros(8)
    last_days = day_list[-3:]
    
    # Trigger accumulation
    triggers = 0
    for d in last_days:
        if d['sleep_hours'] < 6: triggers += 1
        if d['pressure_change'] < -10: triggers += 1
        if d['alcohol_drinks'] >= 5: triggers += 1
        if d['hours_fasting'] > 6: triggers += 1
    clinical[4] = triggers / 12.0
    
    clinical[6] = max(0, 7.0 - day_list[-1]['sleep_hours']) / 3.0
    clinical[7] = day_list[-1]['stress_level'] / 10.0
    
    # Menstrual high risk
    last_menstrual = menstrual_seq[-1]
    if last_menstrual in [0, 1, 26, 27]:
        clinical[1] = 1.0
    
    return {
        'continuous': continuous_seq.astype(np.float32),
        'binary': binary_seq.astype(np.float32),
        'menstrual': menstrual_seq,
        'day_of_week': (np.arange(14) % 7).astype(np.int64),
        'clinical': clinical.astype(np.float32),
    }


def predict(model, sequence, device):
    continuous = torch.tensor(sequence['continuous']).unsqueeze(0).to(device)
    binary = torch.tensor(sequence['binary']).unsqueeze(0).to(device)
    menstrual = torch.tensor(sequence['menstrual']).unsqueeze(0).to(device)
    day_of_week = torch.tensor(sequence['day_of_week']).unsqueeze(0).to(device)
    clinical = torch.tensor(sequence['clinical']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(continuous, binary, menstrual, day_of_week, clinical)
        prob = torch.sigmoid(outputs['attack_logits']).item()
    return prob


def run_constraint_matched_validation(model, device):
    """
    Test scenarios that EXACTLY match the data generation constraints.
    """
    print("\n" + "=" * 70)
    print("🧪 CONSTRAINT-MATCHED VALIDATION")
    print("=" * 70)
    print("\nUsing EXACT trigger thresholds from synthetic data generation:\n")
    print("   • Sleep deprivation: <6 hours → OR 3.98")
    print("   • Alcohol: 5+ drinks → OR 2.08")
    print("   • Pressure drop: >10 mb → OR 1.27")
    print("   • Menstrual days 0-1 → OR 2.04")
    print("   • Nighttime snack → OR 0.60 (protective)")
    print()
    
    results = []
    
    # ============================================
    # BASELINE: Perfect healthy 14 days
    # ============================================
    baseline_day = create_day(
        sleep_hours=7.5,    # Normal sleep
        stress_level=5.0,   # Moderate stress
        pressure_change=0,  # No weather change
        alcohol_drinks=0,   # No alcohol
        hours_fasting=4,    # Regular meals
        had_snack=1,        # Has protective snack
    )
    baseline_pattern = [baseline_day.copy() for _ in range(14)]
    seq = create_sequence(baseline_pattern)
    prob = predict(model, seq, device)
    
    print("-" * 70)
    print("1️⃣  BASELINE (All protective factors)")
    print("-" * 70)
    print("   Sleep 7.5h, no alcohol, regular meals, has snack (protective)")
    print(f"   📊 Risk: {prob*100:.1f}%")
    baseline_risk = prob
    results.append(("Baseline (protected)", prob))
    
    # ============================================
    # SLEEP DEPRIVATION: <6 hours (OR 3.98)
    # ============================================
    sleep_deprived = [create_day(sleep_hours=7.5) for _ in range(11)]
    # Last 3 days: severe sleep deprivation
    for _ in range(3):
        sleep_deprived.append(create_day(
            sleep_hours=4.5,  # Well under 6h threshold
            sleep_quality=0,
        ))
    
    seq = create_sequence(sleep_deprived)
    prob = predict(model, seq, device)
    
    print("\n" + "-" * 70)
    print("2️⃣  SLEEP DEPRIVATION (<6h for 3 days) - OR 3.98")
    print("-" * 70)
    print("   Days 12-14: Sleep 4.5 hours (threshold <6h)")
    print(f"   📊 Risk: {prob*100:.1f}% (Δ = +{(prob-baseline_risk)*100:.1f}%)")
    print(f"   Expected: ~{baseline_risk * 3.98 * 100:.0f}% if OR applied directly")
    results.append(("Sleep <6h (OR 3.98)", prob))
    
    # ============================================
    # ALCOHOL 5+ DRINKS (OR 2.08)
    # ============================================
    alcohol_pattern = [create_day() for _ in range(12)]
    # Last 2 days: heavy drinking (effect on next day per your spec)
    alcohol_pattern.append(create_day(alcohol_drinks=6))  # Day 13
    alcohol_pattern.append(create_day(alcohol_drinks=5))  # Day 14
    
    seq = create_sequence(alcohol_pattern)
    prob = predict(model, seq, device)
    
    print("\n" + "-" * 70)
    print("3️⃣  ALCOHOL 5+ DRINKS (2 days) - OR 2.08")
    print("-" * 70)
    print("   Days 13-14: 5-6 drinks (threshold ≥5)")
    print(f"   📊 Risk: {prob*100:.1f}% (Δ = +{(prob-baseline_risk)*100:.1f}%)")
    results.append(("Alcohol 5+ (OR 2.08)", prob))
    
    # ============================================
    # MENSTRUAL DAYS 0-1 (OR 2.04)
    # ============================================
    menstrual_pattern = [create_day(menstrual_cycle_day=i+14) for i in range(12)]
    # Days 13-14: Menstrual days 0-1 (highest risk)
    menstrual_pattern.append(create_day(menstrual_cycle_day=0))
    menstrual_pattern.append(create_day(menstrual_cycle_day=1))
    
    seq = create_sequence(menstrual_pattern)
    prob = predict(model, seq, device)
    
    print("\n" + "-" * 70)
    print("4️⃣  MENSTRUAL DAYS 0-1 - OR 2.04")
    print("-" * 70)
    print("   Days 13-14: Menstrual cycle days 0-1")
    print(f"   📊 Risk: {prob*100:.1f}% (Δ = +{(prob-baseline_risk)*100:.1f}%)")
    results.append(("Menstrual 0-1 (OR 2.04)", prob))
    
    # ============================================
    # BAROMETRIC PRESSURE DROP >10mb (OR 1.27)
    # ============================================
    weather_pattern = [create_day(barometric_pressure=1013, pressure_change=0) for _ in range(11)]
    # Last 3 days: pressure dropping
    weather_pattern.append(create_day(barometric_pressure=1008, pressure_change=-5))
    weather_pattern.append(create_day(barometric_pressure=1000, pressure_change=-8))
    weather_pattern.append(create_day(barometric_pressure=990, pressure_change=-12))  # >10mb drop
    
    seq = create_sequence(weather_pattern)
    prob = predict(model, seq, device)
    
    print("\n" + "-" * 70)
    print("5️⃣  PRESSURE DROP >10mb - OR 1.27")
    print("-" * 70)
    print("   Day 14: Pressure drop of 12mb (threshold >10mb)")
    print(f"   📊 Risk: {prob*100:.1f}% (Δ = +{(prob-baseline_risk)*100:.1f}%)")
    results.append(("Pressure >10mb (OR 1.27)", prob))
    
    # ============================================
    # COMBINED: Sleep + Alcohol + Menstrual
    # ============================================
    combined_pattern = [create_day() for _ in range(11)]
    for i in range(3):
        combined_pattern.append(create_day(
            sleep_hours=4.5,
            alcohol_drinks=5,
            menstrual_cycle_day=i,  # Days 0, 1, 2
            sleep_quality=0,
            had_snack=0,  # Remove protective factor
        ))
    
    seq = create_sequence(combined_pattern)
    prob = predict(model, seq, device)
    
    print("\n" + "-" * 70)
    print("6️⃣  COMBINED TRIGGERS (Sleep + Alcohol + Menstrual)")
    print("-" * 70)
    print("   Days 12-14: Sleep 4.5h + 5 drinks + menstrual days 0-2")
    print(f"   📊 Risk: {prob*100:.1f}% (Δ = +{(prob-baseline_risk)*100:.1f}%)")
    print(f"   Expected multiplicative: ~{baseline_risk * 3.98 * 2.08 * 2.04 * 100:.0f}% (capped at 100%)")
    results.append(("Combined (multiplicative)", prob))
    
    # ============================================
    # PROTECTIVE: Nighttime snack (OR 0.60)
    # ============================================
    # Start with some risk factors, then add protective snack
    risky_no_snack = [create_day(
        sleep_hours=5.5,
        stress_level=7,
        had_snack=0,  # No protective snack
    ) for _ in range(14)]
    
    seq = create_sequence(risky_no_snack)
    prob_no_snack = predict(model, seq, device)
    
    risky_with_snack = [create_day(
        sleep_hours=5.5,
        stress_level=7,
        had_snack=1,  # WITH protective snack
    ) for _ in range(14)]
    
    seq = create_sequence(risky_with_snack)
    prob_with_snack = predict(model, seq, device)
    
    print("\n" + "-" * 70)
    print("7️⃣  PROTECTIVE FACTOR: Nighttime Snack (OR 0.60)")
    print("-" * 70)
    print(f"   Without snack: {prob_no_snack*100:.1f}%")
    print(f"   With snack:    {prob_with_snack*100:.1f}%")
    print(f"   Reduction:     {(prob_no_snack - prob_with_snack)*100:.1f}%")
    if prob_with_snack < prob_no_snack:
        print("   ✅ Snack is protective (as expected)")
    else:
        print("   ⚠️ Snack not showing protective effect")
    results.append(("No snack (risk)", prob_no_snack))
    results.append(("With snack (protected)", prob_with_snack))
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("📋 SUMMARY - TRIGGER RESPONSE")
    print("=" * 70)
    
    print(f"\n   {'Scenario':<35} {'Risk':>8} {'vs Base':>10} {'Expected OR':>12}")
    print("   " + "-" * 65)
    
    expected_ors = {
        "Baseline (protected)": 1.0,
        "Sleep <6h (OR 3.98)": 3.98,
        "Alcohol 5+ (OR 2.08)": 2.08,
        "Menstrual 0-1 (OR 2.04)": 2.04,
        "Pressure >10mb (OR 1.27)": 1.27,
        "Combined (multiplicative)": 16.8,  # 3.98 * 2.08 * 2.04
    }
    
    for name, prob in results:
        diff = prob - baseline_risk
        ratio = prob / baseline_risk if baseline_risk > 0 else 1
        expected = expected_ors.get(name, "?")
        
        bar = "█" * int(prob * 40)
        indicator = "🔴" if prob > 0.5 else "🟡" if prob > 0.35 else "🟢"
        
        exp_str = f"{expected:.2f}" if isinstance(expected, float) else expected
        print(f"   {indicator} {name:<33} {prob*100:5.1f}%  {diff*100:+6.1f}%      {exp_str}")
    
    # Check if model responds in correct direction
    print("\n" + "-" * 70)
    print("   Validation:")
    
    passed = 0
    
    # Check each trigger increases risk
    for name, prob in results[1:6]:  # Skip baseline
        if prob > baseline_risk:
            print(f"   ✅ {name}: Increases risk")
            passed += 1
        else:
            print(f"   ❌ {name}: Does NOT increase risk")
    
    # Check combined is highest
    combined_prob = results[5][1]
    if combined_prob == max(r[1] for r in results[:6]):
        print(f"   ✅ Combined triggers = highest risk")
        passed += 1
    else:
        print(f"   ⚠️ Combined is not highest")
    
    print(f"\n   Directional Tests Passed: {passed}/6")
    
    if passed >= 5:
        print("   🎉 MODEL RESPONDS CORRECTLY TO TRIGGERS!")
    elif passed >= 3:
        print("   ✓ MODEL PARTIALLY RESPONDS")
    else:
        print("   ⚠️ MODEL NOT RESPONDING TO KNOWN TRIGGERS")
    
    print("=" * 70)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='../models_v2/mamba_finetuned_v2.pth')
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    model = load_model(args.model, device)
    print(f"✓ Loaded: {args.model}\n")
    
    run_constraint_matched_validation(model, device)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()