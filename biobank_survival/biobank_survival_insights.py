import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from lifelines import CoxPHFitter
from pathlib import Path


def generate_survival_report(
    model_dir: str,
    output_file: str = "survival_insights_report.md",
    top_features: int = 10,
):
    """
    Generate a markdown report with clinical insights from survival analysis.
    
    Args:
        model_dir: Directory containing model results
        output_file: Output markdown file
        top_features: Number of top features to include
    """
    # Locate Cox model summary files
    cox_summary_files = list(Path(model_dir).glob("**/M*_summary.csv"))
    if not cox_summary_files:
        print(f"No Cox model summary files found in {model_dir}")
        return
    
    # Locate hazard ratio files
    hr_files = list(Path(model_dir).glob("**/*hazard_ratios.csv"))
    
    # Create a markdown report
    with open(output_file, "w") as f:
        f.write("# Survival Analysis Clinical Insights Report\n\n")
        
        # Write introduction
        f.write("## Overview\n\n")
        f.write("This report summarizes key findings from survival analysis models ")
        f.write("to provide clinical insights about risk factors and protective factors.\n\n")
        
        # Summarize the models
        f.write("## Models\n\n")
        f.write("The following models were evaluated:\n\n")
        
        for i, model_name in enumerate([
            "PaPaGei Only (M0)", 
            "Traditional Factors Only (M1)", 
            "PaPaGei + Traditional Factors (M2)",
            "pyPPG Only (M3)",
            "pyPPG + Traditional Factors (M4)"
        ]):
            f.write(f"{i+1}. **{model_name}**\n")
        
        f.write("\n")
        
        # Summarize performance metrics
        performance_files = list(Path(model_dir).glob("**/survival_analysis_summary.csv"))
        if performance_files:
            f.write("## Model Performance\n\n")
            
            perf_df = pd.read_csv(performance_files[0])
            
            # Extract C-index for each model
            f.write("| Model | C-index |\n")
            f.write("|-------|--------|\n")
            
            for i, row in perf_df.iterrows():
                model = row['Model'].split(':')[0]
                c_index = row['C-index']
                f.write(f"| {model} | {c_index} |\n")
            
            f.write("\n")
            f.write("C-index measures the model's ability to rank patients by risk. ")
            f.write("Values range from 0.5 (random) to 1.0 (perfect).\n\n")
        
        # Key risk factors section
        f.write("## Key Risk Factors\n\n")
        f.write("The following factors were associated with increased risk:\n\n")
        
        risk_factors = []
        
        # Process each hazard ratio file
        for hr_file in hr_files:
            model_name = os.path.basename(hr_file).split('_')[0]
            
            try:
                hr_df = pd.read_csv(hr_file)
                
                # Filter for significant risk factors (HR > 1, p < 0.05)
                risk_df = hr_df[(hr_df['Hazard_Ratio'] > 1) & (hr_df['P_Value'] < 0.05)]
                
                # Sort by hazard ratio (descending)
                risk_df = risk_df.sort_values('Hazard_Ratio', ascending=False)
                
                # Take top factors
                for _, row in risk_df.head(top_features).iterrows():
                    feature = row['Feature']
                    hr = row['Hazard_Ratio']
                    ci_lower = row['HR_Lower_CI']
                    ci_upper = row['HR_Upper_CI']
                    p_value = row['P_Value']
                    
                    risk_factors.append({
                        'Model': model_name,
                        'Feature': feature,
                        'HR': hr,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'P_Value': p_value
                    })
            except Exception as e:
                print(f"Error processing {hr_file}: {e}")
        
        # Create a table of top risk factors
        if risk_factors:
            # Sort by HR (highest first)
            risk_factors_sorted = sorted(risk_factors, key=lambda x: x['HR'], reverse=True)
            
            f.write("| Factor | Hazard Ratio | 95% CI | P-value | Model |\n")
            f.write("|--------|--------------|--------|---------|-------|\n")
            
            # Show top 10 overall
            for factor in risk_factors_sorted[:10]:
                feature = factor['Feature']
                hr = factor['HR']
                ci = f"{factor['CI_Lower']:.2f}-{factor['CI_Upper']:.2f}"
                p = factor['P_Value']
                model = factor['Model']
                
                f.write(f"| {feature} | {hr:.2f} | {ci} | {p:.4f} | {model} |\n")
            
            f.write("\n")
            f.write("Hazard Ratio > 1 indicates increased risk. ")
            f.write("Higher values mean stronger association with adverse outcomes.\n\n")
        else:
            f.write("No significant risk factors identified.\n\n")
        
        # Protective factors section
        f.write("## Protective Factors\n\n")
        f.write("The following factors were associated with decreased risk (protective):\n\n")
        
        protective_factors = []
        
        # Process each hazard ratio file again for protective factors
        for hr_file in hr_files:
            model_name = os.path.basename(hr_file).split('_')[0]
            
            try:
                hr_df = pd.read_csv(hr_file)
                
                # Filter for significant protective factors (HR < 1, p < 0.05)
                protect_df = hr_df[(hr_df['Hazard_Ratio'] < 1) & (hr_df['P_Value'] < 0.05)]
                
                # Sort by hazard ratio (ascending)
                protect_df = protect_df.sort_values('Hazard_Ratio', ascending=True)
                
                # Take top protective factors
                for _, row in protect_df.head(top_features).iterrows():
                    feature = row['Feature']
                    hr = row['Hazard_Ratio']
                    ci_lower = row['HR_Lower_CI']
                    ci_upper = row['HR_Upper_CI']
                    p_value = row['P_Value']
                    
                    protective_factors.append({
                        'Model': model_name,
                        'Feature': feature,
                        'HR': hr,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'P_Value': p_value
                    })
            except Exception as e:
                print(f"Error processing {hr_file}: {e}")
        
        # Create a table of top protective factors
        if protective_factors:
            # Sort by HR (lowest first)
            protective_factors_sorted = sorted(protective_factors, key=lambda x: x['HR'])
            
            f.write("| Factor | Hazard Ratio | 95% CI | P-value | Model |\n")
            f.write("|--------|--------------|--------|---------|-------|\n")
            
            # Show top 10 overall
            for factor in protective_factors_sorted[:10]:
                feature = factor['Feature']
                hr = factor['HR']
                ci = f"{factor['CI_Lower']:.2f}-{factor['CI_Upper']:.2f}"
                p = factor['P_Value']
                model = factor['Model']
                
                f.write(f"| {feature} | {hr:.2f} | {ci} | {p:.4f} | {model} |\n")
            
            f.write("\n")
            f.write("Hazard Ratio < 1 indicates decreased risk (protective effect). ")
            f.write("Lower values mean stronger protective association.\n\n")
        else:
            f.write("No significant protective factors identified.\n\n")
        
        # Survival probability estimations
        f.write("## Survival Probability Estimates\n\n")
        f.write("The following table shows estimated survival probabilities at different time points ")
        f.write("for patients with different risk profiles:\n\n")
        
        # Find a Cox model to use for predictions
        cox_model_files = list(Path(model_dir).glob("**/M*_classifier.joblib"))
        
        if cox_model_files:
            import joblib
            
            try:
                # Load a Cox model (preferably M2 or M4 which include all features)
                for model_file in cox_model_files:
                    if "M2" in str(model_file) or "M4" in str(model_file):
                        cox_model = joblib.load(model_file)
                        break
                else:
                    # If no M2 or M4, use the first available model
                    cox_model = joblib.load(cox_model_files[0])
                
                # Create hypothetical patients
                if isinstance(cox_model, CoxPHFitter):
                    # Define time points
                    time_points = [30, 90, 180, 365, 730]  # days
                    
                    # Create example risk profiles
                    f.write("| Risk Profile | 30-day | 90-day | 180-day | 1-year | 2-year |\n")
                    f.write("|--------------|--------|--------|---------|--------|--------|\n")
                    
                    # Get the median values for all features
                    X_median = cox_model.params_.to_frame().T
                    X_median.iloc[0, :] = 0  # Set to baseline
                    
                    # Create low, medium, and high risk profiles
                    risk_profiles = []
                    
                    # Try to use age and sex if available
                    if 'age' in X_median.columns:
                        profile = X_median.copy()
                        profile['age'] = 1  # Standardized value for older age
                        risk_profiles.append(("Older Age", profile))
                    
                    if 'sex' in X_median.columns:
                        profile = X_median.copy()
                        profile['sex'] = 1  # Assuming 1 is male
                        risk_profiles.append(("Male", profile))
                    
                    # Add feature-based profiles using significant risk factors
                    if risk_factors:
                        # Get the top risk factor
                        top_risk_feature = risk_factors_sorted[0]['Feature']
                        if top_risk_feature in X_median.columns:
                            profile = X_median.copy()
                            profile[top_risk_feature] = 1  # Standardized value for high risk
                            risk_profiles.append((f"High {top_risk_feature}", profile))
                    
                    # Add a generic baseline profile
                    risk_profiles.append(("Baseline", X_median))
                    
                    # Add feature-based profiles using significant protective factors
                    if protective_factors:
                        # Get the top protective factor
                        top_protective_feature = protective_factors_sorted[0]['Feature']
                        if top_protective_feature in X_median.columns:
                            profile = X_median.copy()
                            profile[top_protective_feature] = 1  # Standardized value
                            risk_profiles.append((f"High {top_protective_feature}", profile))
                    
                    # Calculate survival probabilities for each profile
                    for profile_name, profile in risk_profiles:
                        # Get survival function
                        surv_func = cox_model.predict_survival_function(profile)
                        
                        # Interpolate survival probabilities at time points
                        surv_probs = []
                        for t in time_points:
                            # Find closest time point
                            closest_time = min(surv_func.index, key=lambda x: abs(x - t))
                            surv_probs.append(surv_func.loc[closest_time, 0])
                        
                        # Format as percentages
                        surv_prob_strs = [f"{p*100:.1f}%" for p in surv_probs]
                        
                        # Add to table
                        f.write(f"| {profile_name} | {' | '.join(surv_prob_strs)} |\n")
                    
                    f.write("\n")
                    f.write("Values represent the estimated survival probability at each time point.\n\n")
            except Exception as e:
                print(f"Error creating survival estimates: {e}")
                f.write("Survival probability estimates could not be generated.\n\n")
        else:
            f.write("No Cox model was found for generating survival probability estimates.\n\n")
        
        # Final notes and interpretation
        f.write("## Interpretation Notes\n\n")
        f.write("- The hazard ratio (HR) quantifies the effect of a variable on survival outcome.\n")
        f.write("- HR > 1: Factor associated with increased risk (worse survival).\n")
        f.write("- HR < 1: Factor associated with decreased risk (better survival).\n")
        f.write("- The 95% confidence interval (CI) indicates the precision of the hazard ratio estimate.\n")
        f.write("- P-value < 0.05 is considered statistically significant.\n")
        f.write("- These findings are associations and do not necessarily imply causation.\n\n")
        
        # Model details
        f.write("## Model Details\n\n")
        f.write("- Cox Proportional Hazards regression was used for this analysis.\n")
        f.write("- Advanced models include AFT (Accelerated Failure Time) and Random Survival Forest.\n")
        f.write("- Feature importance was assessed using hazard ratios and permutation importance.\n")
        f.write("- The C-index was used to evaluate model discrimination.\n\n")
        
        # Created date
        from datetime import datetime
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
    
    print(f"Survival insights report generated: {output_file}")
    return output_file


def create_individual_risk_calculator(
    cox_model_path: str,
    scaler_path: str,
    output_file: str = "risk_calculator.py",
    feature_subset: Optional[List[str]] = None,
):
    """
    Create a standalone risk calculator from a trained Cox model.
    
    Args:
        cox_model_path: Path to the saved Cox model
        scaler_path: Path to the saved scaler
        output_file: Output Python script
        feature_subset: Optional subset of features to include in the calculator
                       (if None, uses all features in the model)
    """
    import joblib
    
    # Load the model and scaler
    try:
        cox_model = joblib.load(cox_model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return
    
    # Determine features to include
    if feature_subset is None:
        if isinstance(cox_model, CoxPHFitter):
            feature_subset = cox_model.params_.index.tolist()
        else:
            print("Could not determine features from model")
            return
    
    # Create a Python script
    with open(output_file, "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("\"\"\"Survival Risk Calculator\n\n")
        f.write("This script provides a risk calculator based on a trained Cox Proportional Hazards model.\n")
        f.write("\"\"\"\n\n")
        
        f.write("import numpy as np\n")
        f.write("import pandas as pd\n")
        f.write("import joblib\n")
        f.write("from pathlib import Path\n\n")
        
        # Write the feature list
        f.write("# Features required for the calculator\n")
        f.write("FEATURES = [\n")
        for feature in feature_subset:
            f.write(f"    '{feature}',\n")
        f.write("]\n\n")
        
        # Write the calculator function
        f.write("def calculate_risk_score(patient_data):\n")
        f.write("    \"\"\"\n")
        f.write("    Calculate risk score and survival probabilities for a patient.\n")
        f.write("    \n")
        f.write("    Args:\n")
        f.write("        patient_data: Dictionary of patient features\n")
        f.write("    \n")
        f.write("    Returns:\n")
        f.write("        Dictionary of risk score and survival probabilities\n")
        f.write("    \"\"\"\n")
        f.write("    # Load the model and scaler\n")
        f.write("    model_dir = Path(__file__).parent\n")
        f.write(f"    cox_model = joblib.load(model_dir / '{os.path.basename(cox_model_path)}')\n")
        f.write(f"    scaler = joblib.load(model_dir / '{os.path.basename(scaler_path)}')\n\n")
        
        f.write("    # Check for missing features\n")
        f.write("    missing_features = [f for f in FEATURES if f not in patient_data]\n")
        f.write("    if missing_features:\n")
        f.write("        raise ValueError(f'Missing required features: {missing_features}')\n\n")
        
        f.write("    # Create a DataFrame with patient data\n")
        f.write("    patient_df = pd.DataFrame([patient_data])\n\n")
        
        f.write("    # Select and scale features\n")
        f.write("    X = patient_df[FEATURES]\n")
        f.write("    X_scaled = scaler.transform(X)\n")
        f.write("    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)\n\n")
        
        f.write("    # Calculate risk score (hazard ratio)\n")
        f.write("    risk_score = cox_model.predict_partial_hazard(X_scaled_df).iloc[0]\n\n")
        
        f.write("    # Calculate survival probabilities at different time points\n")
        f.write("    time_points = [30, 90, 180, 365, 730]  # days\n")
        f.write("    surv_func = cox_model.predict_survival_function(X_scaled_df)\n")
        f.write("    \n")
        f.write("    # Get survival probabilities at each time point\n")
        f.write("    survival_probs = {}\n")
        f.write("    for t in time_points:\n")
        f.write("        # Find closest time point\n")
        f.write("        closest_time = min(surv_func.index, key=lambda x: abs(x - t))\n")
        f.write("        survival_probs[f'{t}_days'] = surv_func.loc[closest_time, 0]\n\n")
        
        f.write("    return {\n")
        f.write("        'risk_score': risk_score,\n")
        f.write("        'survival_probabilities': survival_probs\n")
        f.write("    }\n\n")
        
        # Write a simple command-line interface
        f.write("def main():\n")
        f.write("    \"\"\"Command-line interface for the risk calculator.\"\"\"\n")
        f.write("    import argparse\n")
        f.write("    import json\n\n")
        
        f.write("    parser = argparse.ArgumentParser(description='Survival Risk Calculator')\n")
        
        # Add arguments for each feature
        for feature in feature_subset:
            f.write(f"    parser.add_argument('--{feature}', type=float, required=True, help='{feature} value')\n")
        
        f.write("    parser.add_argument('--json', action='store_true', help='Output as JSON')\n")
        f.write("    args = parser.parse_args()\n\n")
        
        f.write("    # Collect patient data from arguments\n")
        f.write("    patient_data = {}\n")
        for feature in feature_subset:
            f.write(f"    patient_data['{feature}']
