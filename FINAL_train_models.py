"""
GPA Prediction System Model Training
Trains and evaluates multiple ML models
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# OUTPUT PATH
OUTPUT_PATH = r"C:\python\files"

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):

        print("="*70)
        print("LOADING PROCESSED DATA")
        print("="*70)
        
        df = pd.read_csv(os.path.join(OUTPUT_PATH, 'processed_features.csv'))
        
        metadata_cols = ['student_id', 'target_semester', 'major', 'target_gpa']
        X = df.drop(metadata_cols, axis=1)
        y = df['target_gpa']
        metadata = df[['student_id', 'target_semester', 'major']]
        
        print(f"✓ Loaded {len(df)} samples with {X.shape[1]} features")
        return X, y, metadata
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        print("\n" + "="*70)
        print("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        print("✓ Model trained")
        return model
    
    def train_ridge_regression(self, X_train, y_train):
        """Train Ridge Regression with hyperparameter tuning"""
        print("\n" + "="*70)
        print("Training Ridge Regression...")
        
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge()
        
        grid_search = GridSearchCV(
            ridge, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best alpha: {grid_search.best_params_['alpha']}")
        print(f"✓ Best CV MAE: {-grid_search.best_score_:.4f}")
        
        self.models['Ridge Regression'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        print("\n" + "="*70)
        print("Training Random Forest (this may take a few minutes)...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV MAE: {-grid_search.best_score_:.4f}")
        
        self.models['Random Forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost"""
        print("\n" + "="*70)
        print("Training XGBoost (this may take a few minutes)...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV MAE: {-grid_search.best_score_:.4f}")
        
        self.models['XGBoost'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, model_name, X_train, y_train, X_test, y_test):
  
        # Training predictions
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.results[model_name] = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'predictions': y_test_pred
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Training   - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  Test       - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        return y_test_pred
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
    
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        # Train each model
        self.train_linear_regression(X_train, y_train)
        self.train_ridge_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        # Evaluate all models
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        
        # Find best model
        best_test_mae = float('inf')
        for model_name, results in self.results.items():
            if results['test_mae'] < best_test_mae:
                best_test_mae = results['test_mae']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test MAE: {best_test_mae:.4f}")
        print(f"{'='*70}")
    
    def create_visualizations(self, y_test, X_train):
       
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            predictions = self.results[model_name]['predictions']
            
            ax.scatter(y_test, predictions, alpha=0.5, s=30)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            
            mae = self.results[model_name]['test_mae']
            r2 = self.results[model_name]['test_r2']
            
            ax.set_xlabel('Actual GPA', fontsize=12)
            ax.set_ylabel('Predicted GPA', fontsize=12)
            ax.set_title(f'{model_name}\nMAE: {mae:.4f}, R²: {r2:.4f}', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved model_comparison.png")
        plt.close()
        
        # Metrics comparison
        metrics_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Model Metrics Comparison', fontsize=16, fontweight='bold')
        
        axes[0].bar(metrics_df.index, metrics_df['test_mae'], color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[0].set_title('Test MAE (Lower is Better)', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(metrics_df.index, metrics_df['test_rmse'], color='lightcoral', edgecolor='black')
        axes[1].set_ylabel('Root Mean Squared Error', fontsize=12)
        axes[1].set_title('Test RMSE (Lower is Better)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(metrics_df.index, metrics_df['test_r2'], color='lightgreen', edgecolor='black')
        axes[2].set_ylabel('R² Score', fontsize=12)
        axes[2].set_title('Test R² (Higher is Better)', fontsize=12)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved metrics_comparison.png")
        plt.close()
        
        # Feature importance (Random Forest)
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 20 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved feature_importance.png")
        plt.close()
    
    def save_models(self):
        
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        with open(os.path.join(OUTPUT_PATH, 'trained_models.pkl'), 'wb') as f:
            pickle.dump({
                'models': self.models,
                'best_model_name': self.best_model_name,
                'results': self.results
            }, f)
        print("✓ Saved trained_models.pkl")
        
        # Save results summary
        summary = []
        summary.append("="*80)
        summary.append("GPA PREDICTION SYSTEM - MODEL TRAINING RESULTS")
        summary.append("="*80)
        summary.append("")
        summary.append("MODEL PERFORMANCE COMPARISON:")
        summary.append("-"*80)
        summary.append(f"{'Model':<25} {'Train MAE':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10}")
        summary.append("-"*80)
        
        for model_name, results in self.results.items():
            summary.append(
                f"{model_name:<25} "
                f"{results['train_mae']:<12.4f} "
                f"{results['test_mae']:<12.4f} "
                f"{results['test_rmse']:<12.4f} "
                f"{results['test_r2']:<10.4f}"
            )
        
        summary.append("-"*80)
        summary.append(f"\nBEST MODEL: {self.best_model_name}")
        summary.append(f"Best Test MAE: {self.results[self.best_model_name]['test_mae']:.4f}")
        summary.append("")
        summary.append("="*80)
        
        summary_text = "\n".join(summary)
        
        with open(os.path.join(OUTPUT_PATH, 'training_results.txt'), 'w') as f:
            f.write(summary_text)
        print("✓ Saved training_results.txt")
        
        print(f"\nAll files saved to: {OUTPUT_PATH}")


def main():
    """Main training execution"""
    trainer = ModelTrainer()
    
    # Load data
    X, y, metadata = trainer.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train all models
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Create visualizations
    trainer.create_visualizations(y_test, X_train)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*70)
    print("✓ MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Test MAE: {trainer.results[trainer.best_model_name]['test_mae']:.4f}")
    print("\nNext step: Run FINAL_predictions.py (optional)")


if __name__ == "__main__":
    main()
