"""
Comprehensive Validation Script for Synthetic Migraine Data
============================================================
Validates against all peer-reviewed clinical constraints.
Generates publication-ready figures and reports.

Author: Dhoka
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from scipy import stats


class DataValidator:
    """Validates synthetic migraine data against clinical literature."""
    
    def __init__(self, data_path: str, config_path: str = 'config.json'):
        """Initialize validator with data and config."""
        self.df = pd.read_csv(data_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.validation_results = {}
        self.figures = []
        
    def validate_all(self, save_dir: str = 'validation_output') -> Dict:
        """Run all validation checks and generate report."""
        Path(save_dir).mkdir(exist_ok=True)
        
        print("=" * 70)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("MigraineMamba Enhanced - Synthetic Data Validation")
        print("=" * 70)
        
        # Run all validations
        self._validate_refractory_period()
        self._validate_phenotype_distribution()
        self._validate_attack_frequencies()
        self._validate_feature_distributions()
        self._validate_trigger_odds_ratios()
        self._validate_menstrual_patterns()
        self._validate_gender_differences()
        self._validate_temporal_patterns()
        self._validate_prodrome_patterns()
        self._validate_trigger_sensitivity_prevalence()
        
        # Generate summary
        self._generate_summary_report(save_dir)
        
        # Save results
        with open(f'{save_dir}/validation_results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            clean_results = self._clean_for_json(self.validation_results)
            json.dump(clean_results, f, indent=2)
        
        print(f"\n✓ Validation results saved to {save_dir}/validation_results.json")
        
        return self.validation_results
    
    def _clean_for_json(self, obj):
        """Convert numpy types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    def _validate_refractory_period(self):
        """CRITICAL: Zero tolerance for refractory violations (48h)."""
        print("\n" + "=" * 70)
        print("1. REFRACTORY PERIOD VALIDATION (48-72 hours)")
        print("   Source: Peng et al. 2020")
        print("=" * 70)
        
        violations = 0
        total_gaps = 0
        gap_distribution = []
        violation_patients = []
        
        for patient_id in self.df['patient_id'].unique():
            patient_df = self.df[self.df['patient_id'] == patient_id].sort_values('day')
            attack_days = patient_df[patient_df['attack'] == 1]['day'].values
            
            if len(attack_days) > 1:
                gaps = np.diff(attack_days)
                gap_distribution.extend(gaps)
                total_gaps += len(gaps)
                
                patient_violations = np.sum(gaps < 2)
                violations += patient_violations
                if patient_violations > 0:
                    violation_patients.append(patient_id)
        
        violation_rate = violations / total_gaps if total_gaps > 0 else 0
        
        print(f"\n   Total attack gaps analyzed: {total_gaps:,}")
        print(f"   Refractory violations (<48h): {violations}")
        print(f"   Violation rate: {violation_rate:.4%}")
        
        if len(gap_distribution) > 0:
            print(f"\n   Gap Statistics:")
            print(f"   - Mean: {np.mean(gap_distribution):.1f} days")
            print(f"   - Median: {np.median(gap_distribution):.1f} days")
            print(f"   - Min: {np.min(gap_distribution)} days")
            print(f"   - Max: {np.max(gap_distribution)} days")
        
        passed = violations == 0
        print(f"\n   Status: {'✓ PASS - Zero refractory violations' if passed else '✗ CRITICAL FAIL'}")
        
        if not passed:
            print(f"   Patients with violations: {violation_patients[:10]}...")
        
        self.validation_results['refractory'] = {
            'violations': violations,
            'total_gaps': total_gaps,
            'violation_rate': violation_rate,
            'mean_gap': np.mean(gap_distribution) if gap_distribution else 0,
            'pass': passed
        }
    
    def _validate_phenotype_distribution(self):
        """Validate phenotype prevalence matches literature."""
        print("\n" + "=" * 70)
        print("2. PHENOTYPE DISTRIBUTION")
        print("   Sources: Mungoven 2021, Katsarava 2011")
        print("=" * 70)
        
        patient_phenotypes = self.df.groupby('patient_id')['phenotype'].first()
        actual_dist = patient_phenotypes.value_counts(normalize=True).to_dict()
        
        expected_dist = {k: v['prevalence'] for k, v in self.config['phenotypes'].items()}
        
        print(f"\n   {'Phenotype':<18} {'Expected':<12} {'Actual':<12} {'Diff':<10} {'Status'}")
        print("   " + "-" * 58)
        
        all_pass = True
        tolerance = self.config['validation_targets']['phenotype_tolerance']
        
        for phenotype in ['chronic', 'high_episodic', 'moderate', 'low']:
            expected = expected_dist.get(phenotype, 0)
            actual = actual_dist.get(phenotype, 0)
            diff = actual - expected
            
            status = "✓ PASS" if abs(diff) < tolerance else "✗ FAIL"
            if abs(diff) >= tolerance:
                all_pass = False
            
            print(f"   {phenotype:<18} {expected:<12.2%} {actual:<12.2%} "
                  f"{diff:+9.2%} {status}")
        
        self.validation_results['phenotypes'] = {
            'distribution': actual_dist,
            'expected': expected_dist,
            'tolerance': tolerance,
            'pass': all_pass
        }
    
    def _validate_attack_frequencies(self):
        """Validate attacks per month by phenotype."""
        print("\n" + "=" * 70)
        print("3. ATTACK FREQUENCY BY PHENOTYPE")
        print("=" * 70)
        
        n_months = self.df['day'].max() / 30
        tolerance = self.config['validation_targets']['attack_frequency_tolerance']
        
        print(f"\n   {'Phenotype':<18} {'Expected Range':<18} {'Actual Mean':<12} {'Status'}")
        print("   " + "-" * 58)
        
        results = {}
        all_pass = True
        
        for phenotype in ['chronic', 'high_episodic', 'moderate', 'low']:
            phenotype_df = self.df[self.df['phenotype'] == phenotype]
            
            if len(phenotype_df) == 0:
                continue
            
            attacks_per_patient = phenotype_df.groupby('patient_id')['attack'].sum()
            avg_attacks_per_month = attacks_per_patient.mean() / n_months
            std_attacks = attacks_per_patient.std() / n_months
            
            expected_range = self.config['phenotypes'][phenotype]['attacks_per_month_range']
            expected_str = f"{expected_range[0]}-{expected_range[1]}"
            
            # Allow 10% margin outside range
            in_range = (expected_range[0] * (1 - tolerance) <= avg_attacks_per_month <= 
                       expected_range[1] * (1 + tolerance))
            status = "✓ PASS" if in_range else "⚠ CHECK"
            
            if not in_range:
                all_pass = False
            
            print(f"   {phenotype:<18} {expected_str:<18} {avg_attacks_per_month:<12.1f} {status}")
            
            results[phenotype] = {
                'mean': avg_attacks_per_month,
                'std': std_attacks,
                'expected_range': expected_range,
                'in_range': in_range
            }
        
        self.validation_results['attack_frequency'] = {
            'by_phenotype': results,
            'pass': all_pass
        }
    
    def _validate_feature_distributions(self):
        """Validate feature distributions match literature."""
        print("\n" + "=" * 70)
        print("4. FEATURE DISTRIBUTIONS")
        print("=" * 70)
        
        results = {}
        
        # Sleep
        sleep_mean = self.df['sleep_hours'].mean()
        sleep_std = self.df['sleep_hours'].std()
        expected_mean = self.config['features']['sleep']['mean_hours']
        expected_std = self.config['features']['sleep']['std_hours']
        
        sleep_pass = abs(sleep_mean - expected_mean) < 0.3
        
        print(f"\n   Sleep Duration:")
        print(f"   - Expected: {expected_mean:.1f} ± {expected_std:.1f} hours")
        print(f"   - Actual:   {sleep_mean:.2f} ± {sleep_std:.2f} hours")
        print(f"   - Status:   {'✓ PASS' if sleep_pass else '⚠ CHECK'}")
        
        results['sleep'] = {
            'mean': sleep_mean,
            'std': sleep_std,
            'expected_mean': expected_mean,
            'pass': sleep_pass
        }
        
        # Stress
        stress_mean = self.df['stress_level'].mean()
        stress_std = self.df['stress_level'].std()
        
        print(f"\n   Stress Level:")
        print(f"   - Mean: {stress_mean:.2f} (scale 1-10)")
        print(f"   - Std:  {stress_std:.2f}")
        print(f"   - Range: [{self.df['stress_level'].min()}, {self.df['stress_level'].max()}]")
        
        results['stress'] = {'mean': stress_mean, 'std': stress_std}
        
        # Barometric pressure
        pressure_mean = self.df['barometric_pressure'].mean()
        pressure_std = self.df['barometric_pressure'].std()
        
        print(f"\n   Barometric Pressure:")
        print(f"   - Mean: {pressure_mean:.1f} mb")
        print(f"   - Std:  {pressure_std:.1f} mb")
        
        results['pressure'] = {'mean': pressure_mean, 'std': pressure_std}
        
        # Attack duration by gender
        print(f"\n   Attack Duration:")
        for gender in ['F', 'M']:
            gender_attacks = self.df[(self.df['gender'] == gender) & (self.df['attack'] == 1)]
            if len(gender_attacks) > 0:
                mean_dur = gender_attacks['duration_hours'].mean()
                print(f"   - {gender}: {mean_dur:.1f} hours")
        
        self.validation_results['features'] = results
    
    def _validate_trigger_odds_ratios(self):
        """Validate trigger odds ratios match literature."""
        print("\n" + "=" * 70)
        print("5. TRIGGER ODDS RATIOS")
        print("=" * 70)
        
        results = {}
        
        def calculate_or(exposed_attacks, exposed_total, unexposed_attacks, unexposed_total):
            """Calculate odds ratio with safety checks."""
            if exposed_total == 0 or unexposed_total == 0:
                return None
            
            exposed_noattack = exposed_total - exposed_attacks
            unexposed_noattack = unexposed_total - unexposed_attacks
            
            if exposed_noattack == 0 or unexposed_noattack == 0:
                return None
            
            odds_exposed = exposed_attacks / exposed_noattack
            odds_unexposed = unexposed_attacks / unexposed_noattack
            
            if odds_unexposed == 0:
                return None
            
            return odds_exposed / odds_unexposed
        
        # Sleep trigger (OR 3.98, Duan et al. 2022)
        # Only check patients sensitive to sleep
        sleep_sensitive = self.df[self.df['sensitive_sleep'] == 1]
        
        if len(sleep_sensitive) > 0:
            threshold = self.config['triggers']['sleep']['threshold_hours']
            deprived = sleep_sensitive[sleep_sensitive['sleep_hours'] < threshold]
            normal = sleep_sensitive[sleep_sensitive['sleep_hours'] >= threshold]
            
            sleep_or = calculate_or(
                deprived['attack'].sum(), len(deprived),
                normal['attack'].sum(), len(normal)
            )
            
            if sleep_or:
                expected = self.config['triggers']['sleep']['odds_ratio']
                expected_range = self.config['validation_targets']['sleep_or_range']
                in_range = expected_range[0] <= sleep_or <= expected_range[1]
                
                print(f"\n   Sleep Deprivation (<{threshold}h):")
                print(f"   - Expected OR: {expected} (Duan et al. 2022)")
                print(f"   - Actual OR:   {sleep_or:.2f}")
                print(f"   - Sensitive patients only: {len(sleep_sensitive.groupby('patient_id')):,}")
                print(f"   - Status: {'✓ PASS' if in_range else '⚠ CHECK'}")
                
                results['sleep'] = {
                    'or': sleep_or,
                    'expected': expected,
                    'pass': in_range
                }
        
        # Weather trigger (OR 1.27, Kimoto et al. 2011)
        weather_sensitive = self.df[self.df['sensitive_weather'] == 1]
        
        if len(weather_sensitive) > 0:
            threshold = self.config['triggers']['weather']['threshold_mb']
            dropped = weather_sensitive[weather_sensitive['pressure_change'] < -threshold]
            normal = weather_sensitive[weather_sensitive['pressure_change'] >= -threshold]
            
            weather_or = calculate_or(
                dropped['attack'].sum(), len(dropped),
                normal['attack'].sum(), len(normal)
            )
            
            if weather_or:
                expected = self.config['triggers']['weather']['odds_ratio']
                expected_range = self.config['validation_targets']['weather_or_range']
                in_range = expected_range[0] <= weather_or <= expected_range[1]
                
                print(f"\n   Pressure Drop (>{threshold}mb):")
                print(f"   - Expected OR: {expected} (Kimoto et al. 2011)")
                print(f"   - Actual OR:   {weather_or:.2f}")
                print(f"   - Status: {'✓ PASS' if in_range else '⚠ CHECK'}")
                
                results['weather'] = {
                    'or': weather_or,
                    'expected': expected,
                    'pass': in_range
                }
        
        self.validation_results['triggers'] = results
    
    def _validate_menstrual_patterns(self):
        """Validate menstrual cycle effects (Stewart et al. 2000)."""
        print("\n" + "=" * 70)
        print("6. MENSTRUAL CYCLE PATTERNS (Females)")
        print("   Source: Stewart et al. 2000")
        print("=" * 70)
        
        # Only check females sensitive to menstrual trigger
        female_sensitive = self.df[
            (self.df['gender'] == 'F') & 
            (self.df['sensitive_menstrual'] == 1)
        ]
        
        if len(female_sensitive) == 0:
            print("\n   No menstrual-sensitive female patients in dataset")
            return
        
        results = {}
        
        # High-risk days (0-1): OR 2.04
        high_risk_days = self.config['triggers']['menstrual']['high_risk_days']
        high_risk = female_sensitive[female_sensitive['menstrual_cycle_day'].isin(high_risk_days)]
        other_days = female_sensitive[~female_sensitive['menstrual_cycle_day'].isin(high_risk_days)]
        
        if len(high_risk) > 0 and len(other_days) > 0:
            hr_attack = high_risk['attack'].mean()
            other_attack = other_days['attack'].mean()
            
            if other_attack > 0 and hr_attack > 0 and hr_attack < 1 and other_attack < 1:
                menstrual_or = (hr_attack / (1 - hr_attack)) / (other_attack / (1 - other_attack))
                
                expected = self.config['triggers']['menstrual']['odds_ratio_days_0_1']
                expected_range = self.config['validation_targets']['menstrual_or_range']
                in_range = expected_range[0] <= menstrual_or <= expected_range[1]
                
                print(f"\n   Days 0-1 (Menstruation):")
                print(f"   - Expected OR: {expected} (Stewart et al. 2000)")
                print(f"   - Actual OR:   {menstrual_or:.2f}")
                print(f"   - Attack rate: {hr_attack:.2%} (days 0-1) vs {other_attack:.2%} (other)")
                print(f"   - Status: {'✓ PASS' if in_range else '⚠ CHECK'}")
                
                results['days_0_1'] = {
                    'or': menstrual_or,
                    'expected': expected,
                    'pass': in_range
                }
        
        # Pre-menstrual days (26-27)
        pre_days = self.config['triggers']['menstrual']['pre_menstrual_days']
        pre_menstrual = female_sensitive[female_sensitive['menstrual_cycle_day'].isin(pre_days)]
        
        if len(pre_menstrual) > 0:
            pre_attack = pre_menstrual['attack'].mean()
            print(f"\n   Days 26-27 (Pre-menstrual):")
            print(f"   - Attack rate: {pre_attack:.2%}")
            
            results['pre_menstrual'] = {'attack_rate': pre_attack}
        
        # Ovulation (protective)
        ov_days = self.config['triggers']['menstrual']['ovulation_days']
        ovulation = female_sensitive[female_sensitive['menstrual_cycle_day'].isin(ov_days)]
        
        if len(ovulation) > 0:
            ov_attack = ovulation['attack'].mean()
            print(f"\n   Days 12-16 (Ovulation):")
            print(f"   - Expected: Protective (OR 0.44)")
            print(f"   - Attack rate: {ov_attack:.2%}")
            
            results['ovulation'] = {'attack_rate': ov_attack}
        
        self.validation_results['menstrual'] = results
    
    def _validate_gender_differences(self):
        """Validate gender differences (Vetvik & MacGregor 2017)."""
        print("\n" + "=" * 70)
        print("7. GENDER DIFFERENCES")
        print("   Source: Vetvik & MacGregor 2017")
        print("=" * 70)
        
        results = {}
        
        # Gender distribution
        gender_dist = self.df.groupby('patient_id')['gender'].first().value_counts(normalize=True)
        female_pct = gender_dist.get('F', 0)
        male_pct = gender_dist.get('M', 0)
        
        expected_female = self.config['demographics']['female_proportion']
        gender_pass = abs(female_pct - expected_female) < 0.05
        
        print(f"\n   Gender Distribution:")
        print(f"   - Expected: F={expected_female:.0%}, M={1-expected_female:.0%}")
        print(f"   - Actual:   F={female_pct:.1%}, M={male_pct:.1%}")
        print(f"   - Status:   {'✓ PASS' if gender_pass else '⚠ CHECK'}")
        
        results['distribution'] = {
            'female': female_pct,
            'male': male_pct,
            'pass': gender_pass
        }
        
        # Attack duration by gender
        female_attacks = self.df[(self.df['gender'] == 'F') & (self.df['attack'] == 1)]
        male_attacks = self.df[(self.df['gender'] == 'M') & (self.df['attack'] == 1)]
        
        if len(female_attacks) > 0 and len(male_attacks) > 0:
            female_duration = female_attacks['duration_hours'].mean()
            male_duration = male_attacks['duration_hours'].mean()
            ratio = female_duration / male_duration if male_duration > 0 else 0
            
            # Vetvik 2017: women 38.8h vs men 12.8h (ratio ~3.0)
            # Our model uses 1.5x multiplier
            expected_ratio = self.config['features']['attack_duration']['female_multiplier']
            duration_pass = female_duration > male_duration
            
            print(f"\n   Attack Duration:")
            print(f"   - Expected: F > M (multiplier {expected_ratio}x)")
            print(f"   - Actual:   F={female_duration:.1f}h, M={male_duration:.1f}h")
            print(f"   - Ratio:    {ratio:.2f}x")
            print(f"   - Status:   {'✓ PASS' if duration_pass else '⚠ CHECK'}")
            
            results['duration'] = {
                'female': female_duration,
                'male': male_duration,
                'ratio': ratio,
                'pass': duration_pass
            }
        
        self.validation_results['gender'] = results
    
    def _validate_temporal_patterns(self):
        """Check for realistic temporal patterns."""
        print("\n" + "=" * 70)
        print("8. TEMPORAL PATTERNS")
        print("=" * 70)
        
        results = {}
        
        # Attack clustering
        expected_clustering = self.config['temporal_constraints']['clustering_probability']
        
        # Check patients marked as clustering
        patient_info = self.df.groupby('patient_id').agg({
            'shows_clustering': 'first',
            'attack': 'sum'
        })
        
        actual_clustering = patient_info['shows_clustering'].mean()
        
        print(f"\n   Attack Clustering:")
        print(f"   - Expected: {expected_clustering:.0%} of patients")
        print(f"   - Actual:   {actual_clustering:.1%} of patients")
        
        results['clustering'] = {
            'expected': expected_clustering,
            'actual': actual_clustering
        }
        
        # Day-of-week pattern
        dow_attacks = self.df.groupby('day_of_week')['attack'].mean()
        
        print(f"\n   Day-of-Week Attack Rates:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(days):
            if i in dow_attacks.index:
                print(f"   - {day}: {dow_attacks[i]:.3f}")
        
        results['day_of_week'] = dow_attacks.to_dict()
        
        # Missing data rate
        missing_rate = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))
        expected_missing = self.config['missing_data']['random_missing_rate']
        
        print(f"\n   Missing Data:")
        print(f"   - Expected: ~{expected_missing:.1%}")
        print(f"   - Actual:   {missing_rate:.2%}")
        
        results['missing_data'] = {
            'expected': expected_missing,
            'actual': missing_rate
        }
        
        self.validation_results['temporal'] = results
    
    def _validate_prodrome_patterns(self):
        """Validate prodromal symptom patterns."""
        print("\n" + "=" * 70)
        print("9. PRODROMAL SYMPTOM PATTERNS")
        print("   Sources: Gago-Veiga 2018, Schwedt 2024")
        print("=" * 70)
        
        # Check prodrome prevalence
        expected_prodrome = self.config['prodrome']['prevalence']
        actual_prodrome = self.df.groupby('patient_id')['has_prodrome'].first().mean()
        
        print(f"\n   Prodrome Prevalence:")
        print(f"   - Expected: {expected_prodrome:.0%}")
        print(f"   - Actual:   {actual_prodrome:.1%}")
        
        # Check prodrome symptoms on day before attack
        prodrome_cols = [col for col in self.df.columns if col.startswith('prodrome_')]
        
        if prodrome_cols:
            print(f"\n   Prodrome Symptom Rates (day before attack):")
            
            # Find days before attacks
            attack_indices = self.df[self.df['attack'] == 1].index
            pre_attack_indices = attack_indices - 1
            pre_attack_indices = pre_attack_indices[pre_attack_indices >= 0]
            
            pre_attack_df = self.df.loc[pre_attack_indices]
            
            for col in prodrome_cols:
                rate = pre_attack_df[col].mean()
                symptom_name = col.replace('prodrome_', '').replace('_', ' ').title()
                expected = self.config['prodrome']['symptom_probabilities'].get(
                    col.replace('prodrome_', ''), 0
                )
                print(f"   - {symptom_name}: {rate:.1%} (expected: {expected:.0%})")
        
        self.validation_results['prodrome'] = {
            'prevalence': actual_prodrome,
            'expected': expected_prodrome
        }
    
    def _validate_trigger_sensitivity_prevalence(self):
        """Validate trigger sensitivity prevalence matches literature."""
        print("\n" + "=" * 70)
        print("10. TRIGGER SENSITIVITY PREVALENCE")
        print("=" * 70)
        
        results = {}
        
        sensitivity_cols = [col for col in self.df.columns if col.startswith('sensitive_')]
        
        print(f"\n   {'Trigger':<20} {'Expected':<12} {'Actual':<12} {'Status'}")
        print("   " + "-" * 50)
        
        for col in sensitivity_cols:
            trigger_name = col.replace('sensitive_', '')
            
            # For menstrual, calculate prevalence among females only
            if trigger_name == 'menstrual':
                female_patients = self.df[self.df['gender'] == 'F']
                if len(female_patients) > 0:
                    actual = female_patients.groupby('patient_id')[col].first().mean()
                else:
                    actual = 0
            else:
                actual = self.df.groupby('patient_id')[col].first().mean()
            
            if trigger_name in self.config['triggers']:
                expected = self.config['triggers'][trigger_name]['prevalence']
                diff = abs(actual - expected)
                status = "✓" if diff < 0.05 else "⚠"
                
                print(f"   {trigger_name:<20} {expected:<12.1%} {actual:<12.1%} {status}")
                
                results[trigger_name] = {
                    'expected': expected,
                    'actual': actual,
                    'pass': diff < 0.05
                }
        
        self.validation_results['sensitivity_prevalence'] = results
    
    def _generate_summary_report(self, save_dir: str):
        """Generate summary of all validation results."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        # Count passes
        checks = []
        for key, value in self.validation_results.items():
            if isinstance(value, dict) and 'pass' in value:
                checks.append(value['pass'])
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and 'pass' in subvalue:
                        checks.append(subvalue['pass'])
        
        passed = sum(checks)
        total = len(checks)
        
        print(f"\n   Checks passed: {passed}/{total}")
        
        # Critical check
        refractory_pass = self.validation_results.get('refractory', {}).get('pass', False)
        
        if refractory_pass and passed >= total * 0.8:
            print("\n   ✓ DATASET VALIDATION PASSED")
            print("   Dataset is publication-ready and clinically grounded.")
        elif not refractory_pass:
            print("\n   ✗ CRITICAL FAILURE: Refractory period violations detected")
            print("   Dataset cannot be used until this is fixed.")
        else:
            print("\n   ⚠ VALIDATION WARNINGS")
            print("   Review failed checks before proceeding.")
        
        print("\n" + "=" * 70)
    
    def plot_validation_figures(self, save_dir: str = 'validation_output'):
        """Generate comprehensive validation plots."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Figure 1: Main validation metrics (2x3 grid)
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
        fig1.suptitle('Synthetic Migraine Data - Validation Metrics', 
                      fontsize=14, fontweight='bold')
        
        # 1.1 Phenotype distribution
        ax = axes1[0, 0]
        patient_phenotypes = self.df.groupby('patient_id')['phenotype'].first()
        actual = patient_phenotypes.value_counts(normalize=True)
        expected = pd.Series({k: v['prevalence'] for k, v in self.config['phenotypes'].items()})
        
        x = np.arange(len(actual))
        width = 0.35
        ax.bar(x - width/2, [expected.get(p, 0) for p in actual.index], width, label='Expected', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, actual.values, width, label='Actual', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', '\n') for p in actual.index], fontsize=9)
        ax.set_ylabel('Proportion')
        ax.set_title('Phenotype Distribution')
        ax.legend()
        
        # 1.2 Sleep distribution
        ax = axes1[0, 1]
        self.df['sleep_hours'].dropna().hist(bins=30, ax=ax, color='lightblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=self.config['features']['sleep']['mean_hours'], color='green', 
                   linestyle='--', linewidth=2, label='Expected mean')
        ax.axvline(x=self.config['triggers']['sleep']['threshold_hours'], color='red', 
                   linestyle='--', linewidth=2, label='Trigger threshold')
        ax.set_xlabel('Hours')
        ax.set_ylabel('Frequency')
        ax.set_title('Sleep Duration Distribution')
        ax.legend()
        
        # 1.3 Attack rate by sleep
        ax = axes1[0, 2]
        sleep_bins = pd.cut(self.df['sleep_hours'].dropna(), bins=[0, 4, 6, 8, 10, 12])
        attack_by_sleep = self.df.groupby(sleep_bins, observed=True)['attack'].mean()
        attack_by_sleep.plot(kind='bar', ax=ax, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sleep Hours')
        ax.set_ylabel('Attack Rate')
        ax.set_title('Attack Rate by Sleep Duration')
        ax.tick_params(axis='x', rotation=45)
        
        # 1.4 Menstrual cycle pattern
        ax = axes1[1, 0]
        female_df = self.df[self.df['gender'] == 'F']
        if len(female_df) > 0:
            menstrual_attacks = female_df.groupby('menstrual_cycle_day')['attack'].mean()
            menstrual_attacks.plot(ax=ax, marker='o', color='purple', linewidth=2, markersize=4)
            ax.axvspan(0, 1.5, alpha=0.2, color='red', label='High-risk (days 0-1)')
            ax.axvspan(12, 16, alpha=0.2, color='green', label='Protective (ovulation)')
            ax.set_xlabel('Cycle Day')
            ax.set_ylabel('Attack Rate')
            ax.set_title('Attack Rate by Menstrual Cycle Day')
            ax.legend(fontsize=8)
        
        # 1.5 Inter-attack intervals (refractory check)
        ax = axes1[1, 1]
        gaps = []
        for patient_id in self.df['patient_id'].unique()[:200]:  # Sample
            patient_df = self.df[self.df['patient_id'] == patient_id]
            attack_days = patient_df[patient_df['attack'] == 1]['day'].values
            if len(attack_days) > 1:
                gaps.extend(np.diff(attack_days))
        
        if gaps:
            ax.hist(gaps, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
            ax.axvline(x=2, color='red', linestyle='--', linewidth=2, label='48h threshold')
            ax.set_xlabel('Days Between Attacks')
            ax.set_ylabel('Frequency')
            ax.set_title('Inter-Attack Intervals')
            ax.legend()
        
        # 1.6 Attack rate over time
        ax = axes1[1, 2]
        daily_attacks = self.df.groupby('day')['attack'].mean()
        ax.plot(daily_attacks.index, daily_attacks.values, alpha=0.5, color='steelblue', linewidth=0.5)
        ax.axhline(y=daily_attacks.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        
        # Add rolling average
        rolling = daily_attacks.rolling(7).mean()
        ax.plot(rolling.index, rolling.values, color='darkblue', linewidth=2, label='7-day rolling avg')
        
        ax.set_xlabel('Day')
        ax.set_ylabel('Attack Rate')
        ax.set_title('Daily Attack Rate Over Time')
        ax.legend()
        
        plt.tight_layout()
        fig1.savefig(f'{save_dir}/validation_metrics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/validation_metrics.png")
        
        # Figure 2: Trigger analysis
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle('Trigger Effect Analysis', fontsize=14, fontweight='bold')
        
        # 2.1 Sleep trigger OR
        ax = axes2[0, 0]
        sleep_sensitive = self.df[self.df['sensitive_sleep'] == 1]
        if len(sleep_sensitive) > 0:
            threshold = self.config['triggers']['sleep']['threshold_hours']
            categories = ['<' + str(threshold) + 'h', '≥' + str(threshold) + 'h']
            rates = [
                sleep_sensitive[sleep_sensitive['sleep_hours'] < threshold]['attack'].mean(),
                sleep_sensitive[sleep_sensitive['sleep_hours'] >= threshold]['attack'].mean()
            ]
            bars = ax.bar(categories, rates, color=['coral', 'steelblue'], alpha=0.7, edgecolor='black')
            ax.set_ylabel('Attack Rate')
            ax.set_title(f'Sleep Deprivation Effect\n(Expected OR: {self.config["triggers"]["sleep"]["odds_ratio"]})')
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{rate:.3f}', ha='center', fontsize=10)
        
        # 2.2 Weather trigger
        ax = axes2[0, 1]
        weather_sensitive = self.df[self.df['sensitive_weather'] == 1]
        if len(weather_sensitive) > 0:
            threshold = self.config['triggers']['weather']['threshold_mb']
            categories = [f'Drop >{threshold}mb', 'Normal']
            rates = [
                weather_sensitive[weather_sensitive['pressure_change'] < -threshold]['attack'].mean(),
                weather_sensitive[weather_sensitive['pressure_change'] >= -threshold]['attack'].mean()
            ]
            bars = ax.bar(categories, rates, color=['coral', 'steelblue'], alpha=0.7, edgecolor='black')
            ax.set_ylabel('Attack Rate')
            ax.set_title(f'Pressure Drop Effect\n(Expected OR: {self.config["triggers"]["weather"]["odds_ratio"]})')
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{rate:.3f}', ha='center', fontsize=10)
        
        # 2.3 Trigger sensitivity prevalence
        ax = axes2[1, 0]
        sensitivity_cols = [col for col in self.df.columns if col.startswith('sensitive_')]
        if sensitivity_cols:
            trigger_names = [col.replace('sensitive_', '') for col in sensitivity_cols]
            actual_prev = [self.df.groupby('patient_id')[col].first().mean() for col in sensitivity_cols]
            expected_prev = [self.config['triggers'].get(name, {}).get('prevalence', 0) for name in trigger_names]
            
            x = np.arange(len(trigger_names))
            width = 0.35
            ax.bar(x - width/2, expected_prev, width, label='Expected', color='steelblue', alpha=0.7)
            ax.bar(x + width/2, actual_prev, width, label='Actual', color='coral', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(trigger_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Prevalence')
            ax.set_title('Trigger Sensitivity Prevalence')
            ax.legend()
        
        # 2.4 Gender comparison
        ax = axes2[1, 1]
        gender_attacks = self.df.groupby('gender')['attack'].mean()
        gender_duration = self.df[self.df['attack'] == 1].groupby('gender')['duration_hours'].mean()
        
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [gender_attacks.get('F', 0) * 100, gender_attacks.get('M', 0) * 100], 
               width, label='Attack Rate (%)', color='steelblue', alpha=0.7)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, [gender_duration.get('F', 0), gender_duration.get('M', 0)], 
                width, label='Duration (h)', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Female', 'Male'])
        ax.set_ylabel('Attack Rate (%)', color='steelblue')
        ax2.set_ylabel('Duration (hours)', color='coral')
        ax.set_title('Gender Differences')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        fig2.savefig(f'{save_dir}/trigger_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/trigger_analysis.png")
        
        plt.close('all')
        
        print(f"\n✓ All validation plots saved to {save_dir}/")


def main():
    """Main entry point for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate synthetic migraine data')
    parser.add_argument('--data', '-d', default='synthetic_data/migraine_synthetic_data.csv',
                        help='Path to synthetic data CSV')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--output', '-o', default='validation_output',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Run validation
    validator = DataValidator(args.data, args.config)
    results = validator.validate_all(args.output)
    
    # Generate plots
    if not args.no_plots:
        validator.plot_validation_figures(args.output)


if __name__ == '__main__':
    main()