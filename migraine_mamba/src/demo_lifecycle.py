"""
Phase 3 Demo: Complete User Lifecycle Simulation
=================================================
Demonstrates the full personalization journey:
- Days 1-14: Foundation model
- Days 15-30: Generic Mamba
- Days 31+: Personalized model

Run this to see the system in action!

Usage:
    python demo_lifecycle.py

Author: Dhoka
Date: December 2025
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from personalization import PersonalizationSystem, UserPhase
from weekly_personalization import WeeklyPersonalizer


def simulate_user_journey(
    user_id: str = "demo_patient_001",
    n_days: int = 45,
    attack_rate: float = 0.15,
    models_dir: str = "../models_v2",
    output_dir: str = "../demo_output",
):
    """
    Simulate a complete user journey through all phases.
    
    Generates realistic daily logs and shows how predictions
    evolve as the user accumulates data.
    """
    
    print("\n" + "=" * 70)
    print("🧠 MIGRAINE PERSONALIZATION - USER LIFECYCLE DEMO")
    print("=" * 70)
    
    # Clean start
    demo_dir = Path(output_dir)
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True)
    
    # Initialize system
    print("\n📱 Initializing personalization system...")
    system = PersonalizationSystem(
        models_dir=models_dir,
        user_data_dir=str(demo_dir / "users"),
    )
    
    # Create user
    print(f"\n👤 Creating user: {user_id}")
    state = system.get_or_create_user(user_id)
    
    # Track results
    journey = {
        'user_id': user_id,
        'days': [],
        'phase_transitions': [],
    }
    
    print(f"\n🗓️ Simulating {n_days} days of usage...\n")
    print("-" * 70)
    
    base_date = datetime.now() - timedelta(days=n_days)
    current_phase = None
    
    for day in range(n_days):
        date = base_date + timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")
        
        # Generate daily features with some patterns
        features = generate_daily_features(day, date.weekday())
        
        # Determine if attack occurred (with realistic patterns)
        attack_prob = calculate_attack_probability(features, attack_rate)
        attack_occurred = np.random.random() < attack_prob
        
        # Log the day
        system.update_user_log(user_id, date_str, features, attack_occurred)
        
        # Get prediction for this day
        prediction = system.predict(user_id, features)
        
        # Track phase transitions
        phase = system.get_user_phase(user_id)
        if phase != current_phase:
            journey['phase_transitions'].append({
                'day': day + 1,
                'from': current_phase.value if current_phase else None,
                'to': phase.value,
            })
            current_phase = phase
            
            print(f"\n🔄 PHASE TRANSITION at Day {day + 1}: → {phase.value.upper()}")
            print("-" * 70)
        
        # Record day results
        day_result = {
            'day': day + 1,
            'date': date_str,
            'phase': phase.value,
            'features': {
                'sleep': features['sleep_hours'],
                'stress': features['stress_level'],
                'alcohol': features['alcohol_drinks'],
            },
            'attack_occurred': attack_occurred,
            'predicted_risk': prediction.risk_probability,
            'risk_level': prediction.risk_level,
            'model': prediction.model_version,
        }
        journey['days'].append(day_result)
        
        # Print key days
        if day in [0, 6, 13, 14, 20, 29, 30, 37, n_days-1] or attack_occurred:
            print_day_summary(day_result, prediction, attack_occurred)
    
    # Run weekly personalization
    print("\n" + "=" * 70)
    print("🔄 RUNNING WEEKLY PERSONALIZATION")
    print("=" * 70)
    
    personalizer = WeeklyPersonalizer(
        generic_model_path=f"{models_dir}/mamba_finetuned_v2.pth",
        user_data_dir=str(demo_dir / "users"),
    )
    
    personalization_result = personalizer.run_weekly_update()
    
    # Final prediction with personalized model
    print("\n" + "=" * 70)
    print("📊 FINAL PERSONALIZED PREDICTION")
    print("=" * 70)
    
    # High-risk scenario
    high_risk_features = {
        'sleep_hours': 4.5,
        'stress_level': 8,
        'barometric_pressure': 1000,
        'pressure_change': -12,
        'temperature': 22,
        'humidity': 65,
        'hours_fasting': 10,
        'alcohol_drinks': 4,
        'had_breakfast': 0,
        'had_lunch': 1,
        'had_dinner': 1,
        'had_snack': 0,
        'bright_light_exposure': 1,
        'sleep_quality': 0,
        'menstrual_cycle_day': 1,
    }
    
    final_prediction = system.predict(user_id, high_risk_features)
    
    print(f"\n   🔮 High-Risk Scenario Prediction:")
    print(f"   ─────────────────────────────────")
    print(f"   Risk: {final_prediction.risk_probability*100:.1f}% ({final_prediction.risk_level})")
    print(f"   Model: {final_prediction.model_version}")
    print(f"   Confidence: {final_prediction.confidence*100:.0f}%")
    
    print(f"\n   🎯 Your Personal Triggers:")
    sorted_triggers = sorted(
        final_prediction.trigger_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (trigger, score) in enumerate(sorted_triggers[:3], 1):
        bar = "█" * int(score * 25)
        print(f"   {i}. {trigger:<12} {score*100:5.1f}% {bar}")
    
    print(f"\n   💡 Recommendations:")
    for rec in final_prediction.recommendations:
        print(f"   • {rec}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📈 JOURNEY SUMMARY")
    print("=" * 70)
    
    state = system.get_or_create_user(user_id)
    
    print(f"\n   User: {user_id}")
    print(f"   Days logged: {state.days_logged}")
    print(f"   Total attacks: {state.total_attacks}")
    print(f"   Current phase: {state.current_phase}")
    print(f"   Model version: {state.model_version}")
    
    # Prediction accuracy analysis
    predictions = [d['predicted_risk'] for d in journey['days']]
    actuals = [1 if d['attack_occurred'] else 0 for d in journey['days']]
    
    # Simple accuracy: high predictions when attacks happened
    attack_days = [i for i, a in enumerate(actuals) if a == 1]
    if attack_days:
        avg_pred_on_attack = np.mean([predictions[i] for i in attack_days])
        avg_pred_no_attack = np.mean([predictions[i] for i in range(len(actuals)) if actuals[i] == 0])
        
        print(f"\n   Prediction Analysis:")
        print(f"   • Avg risk on attack days: {avg_pred_on_attack*100:.1f}%")
        print(f"   • Avg risk on normal days: {avg_pred_no_attack*100:.1f}%")
        print(f"   • Discrimination: {(avg_pred_on_attack - avg_pred_no_attack)*100:+.1f}%")
    
    # Phase breakdown
    print(f"\n   Phase Distribution:")
    phase_counts = {}
    for d in journey['days']:
        phase = d['phase']
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    for phase, count in phase_counts.items():
        bar = "█" * (count * 2)
        print(f"   • {phase:15} {count:3} days {bar}")
    
    # Save journey
    journey_path = demo_dir / "journey.json"
    with open(journey_path, 'w') as f:
        json.dump(journey, f, indent=2, default=str)
    
    print(f"\n   📁 Results saved to: {demo_dir}")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    
    return journey


def generate_daily_features(day: int, weekday: int) -> dict:
    """Generate realistic daily features with patterns."""
    
    # Base values with weekly patterns
    base_sleep = 7.0 if weekday < 5 else 7.5  # More sleep on weekends
    base_stress = 5.5 if weekday < 5 else 4.0  # Less stress on weekends
    
    # Add some autocorrelation (bad days tend to cluster)
    if day > 0 and np.random.random() < 0.3:
        # 30% chance of "carry-over" from previous day
        sleep_adj = np.random.choice([-1, 0, 1], p=[0.5, 0.3, 0.2])
    else:
        sleep_adj = 0
    
    features = {
        'sleep_hours': max(3, min(10, base_sleep + np.random.normal(0, 1.2) + sleep_adj)),
        'stress_level': max(1, min(10, base_stress + np.random.normal(0, 2))),
        'barometric_pressure': 1013 + np.random.normal(0, 8),
        'pressure_change': np.random.normal(0, 5),
        'temperature': 22 + np.random.normal(0, 5),
        'humidity': 50 + np.random.normal(0, 15),
        'hours_fasting': max(0, 4 + np.random.exponential(3)),
        'alcohol_drinks': max(0, int(np.random.exponential(0.5))) if weekday >= 4 else 0,
        'had_breakfast': 1 if np.random.random() > 0.15 else 0,
        'had_lunch': 1 if np.random.random() > 0.1 else 0,
        'had_dinner': 1,
        'had_snack': 1 if np.random.random() > 0.5 else 0,
        'bright_light_exposure': 1 if np.random.random() > 0.7 else 0,
        'sleep_quality': 1 if np.random.random() > 0.25 else 0,
        'menstrual_cycle_day': day % 28 if np.random.random() > 0.5 else -1,
    }
    
    return features


def calculate_attack_probability(features: dict, base_rate: float) -> float:
    """Calculate attack probability based on features and clinical ORs."""
    prob = base_rate
    
    # Apply odds ratios
    if features['sleep_hours'] < 6:
        prob *= 3.98
    if features['alcohol_drinks'] >= 5:
        prob *= 2.08
    if features.get('menstrual_cycle_day', -1) in [0, 1]:
        prob *= 2.04
    if features['pressure_change'] < -10:
        prob *= 1.27
    if features['stress_level'] > 7:
        prob *= 1.5
    if features['had_snack'] == 1:
        prob *= 0.60
    if features['sleep_quality'] == 0:
        prob *= 1.3
    
    return min(prob, 0.85)


def print_day_summary(day_result: dict, prediction, attack: bool):
    """Print summary for a single day."""
    attack_icon = "🔴 ATTACK" if attack else "🟢"
    
    print(f"\n   Day {day_result['day']:2d} ({day_result['date']}) - {day_result['phase'].upper()}")
    print(f"   ├─ Sleep: {day_result['features']['sleep']:.1f}h | "
          f"Stress: {day_result['features']['stress']:.0f}/10 | "
          f"Alcohol: {day_result['features']['alcohol']}")
    print(f"   ├─ Prediction: {prediction.risk_probability*100:.0f}% ({prediction.risk_level})")
    print(f"   ├─ Model: {prediction.model_version}")
    print(f"   └─ Outcome: {attack_icon}")


def main():
    """Run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate user lifecycle")
    parser.add_argument('--days', '-d', type=int, default=45, help='Days to simulate')
    parser.add_argument('--models', '-m', default='../models_v2', help='Models directory')
    parser.add_argument('--output', '-o', default='../demo_output', help='Output directory')
    
    args = parser.parse_args()
    
    simulate_user_journey(
        n_days=args.days,
        models_dir=args.models,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()