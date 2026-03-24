"""
Model Training Script

Complete pipeline to download data, preprocess, train models, and save.
Run this script to train the credit risk models.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.analytics.portfolio_metrics import PortfolioAnalytics


def main(n_samples: int = 100000):
    """
    Main training pipeline.
    
    Args:
        n_samples: Number of samples to use (None for all)
    """
    print("="*60)
    print("ENTERPRISE CREDIT RISK MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Download and load data
    print("\n[1/6] Downloading and loading Lending Club dataset...")
    loader = DataLoader(data_dir="data/raw")
    df = loader.load_accepted_loans(nrows=n_samples)
    
    # Step 2: Preprocess data
    print("\n[2/6] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    
    # Get full processed df for feature engineering
    processed_df = preprocessor.get_processed_dataframe()
    
    # Step 3: Feature engineering
    print("\n[3/6] Engineering credit risk features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(processed_df)
    
    # Get all feature columns (original + engineered)
    all_feature_cols = list(X.columns) + engineer.get_engineered_feature_names()
    available_cols = [c for c in all_feature_cols if c in df_features.columns]
    
    X_enhanced = df_features[available_cols].fillna(0)
    y = df_features['target']
    
    # Step 4: Train models
    print("\n[4/6] Training models...")
    trainer = ModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.prepare_data(X_enhanced, y)
    
    trainer.train_models(X_train, y_train, use_smote=True)
    
    # Cross-validation
    print("\n[5/6] Cross-validating and evaluating models...")
    cv_results = trainer.cross_validate(X_enhanced, y, cv=5)
    best_model = trainer.select_best_model(cv_results)
    
    # Evaluate
    evaluator = ModelEvaluator()
    for model_name in trainer.models.keys():
        y_prob = trainer.predict_proba(X_test, model_name)
        results = evaluator.evaluate_model(y_test.values, y_prob, model_name)
        evaluator.print_evaluation_summary(results)
    
    # Step 5: Save models
    print("\n[6/6] Saving models...")
    trainer.save_models()
    
    # Portfolio metrics summary
    print("\n" + "="*60)
    print("PORTFOLIO SUMMARY")
    print("="*60)
    
    # Add predictions to full dataset
    df_features['pd'] = trainer.predict_proba(X_enhanced)
    
    analytics = PortfolioAnalytics(lgd=0.40)
    metrics = analytics.calculate_portfolio_metrics(
        df_features, 
        pd_column='pd',
        loan_amount_column='loan_amnt' if 'loan_amnt' in df_features.columns else None
    )
    
    print(analytics.generate_portfolio_summary(metrics))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest Model: {best_model}")
    print(f"Models saved to: models/saved/")
    print("\nRun the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train credit risk models')
    parser.add_argument('--samples', type=int, default=100000, 
                        help='Number of samples to use (default: 100000)')
    args = parser.parse_args()
    
    main(n_samples=args.samples)
