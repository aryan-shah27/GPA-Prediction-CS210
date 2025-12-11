"""
GPA Prediction System - FINAL VERSION - Sample Predictions
Demonstrates predictions for sample students
"""

import pandas as pd
import pickle
import os

# OUTPUT PATH
OUTPUT_PATH = r"C:\python\files"

def load_models():
    """Load trained models"""
    with open(os.path.join(OUTPUT_PATH, 'trained_models.pkl'), 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def load_processed_data():
    """Load processed features"""
    df = pd.read_csv(os.path.join(OUTPUT_PATH, 'processed_features.csv'))
    return df

def generate_sample_predictions(df, model_data, num_samples=10):
    """Generate predictions for sample students"""
    print("="*70)
    print("SAMPLE GPA PREDICTIONS")
    print("="*70)
    
    best_model_name = model_data['best_model_name']
    best_model = model_data['models'][best_model_name]
    
    # Get random samples
    samples = df.sample(min(num_samples, len(df)))
    
    # Prepare features (drop metadata columns)
    metadata_cols = ['student_id', 'target_semester', 'major', 'target_gpa']
    X_sample = samples.drop(metadata_cols, axis=1)
    
    # Make predictions
    predictions = best_model.predict(X_sample)
    
    # Display results
    print(f"\nUsing Best Model: {best_model_name}\n")
    print(f"{'Student ID':<12} {'Semester':<15} {'Major':<20} {'Actual GPA':<12} {'Predicted':<12} {'Error':<10}")
    print("-"*95)
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        student_id = int(row['student_id'])
        semester = row['target_semester']
        major = row['major']
        actual = row['target_gpa']
        predicted = predictions[idx]
        error = abs(actual - predicted)
        
        print(f"{student_id:<12} {semester:<15} {major:<20} {actual:<12.3f} {predicted:<12.3f} {error:<10.3f}")
    
    # Summary statistics
    actual_values = samples['target_gpa'].values
    mae = abs(actual_values - predictions).mean()
    
    print("-"*95)
    print(f"\nSample MAE: {mae:.4f}")
    print("\nPrediction Ranges:")
    print(f"  3.5-4.0 (High):      {sum((predictions >= 3.5) & (predictions <= 4.0))} students")
    print(f"  3.0-3.5 (Good):      {sum((predictions >= 3.0) & (predictions < 3.5))} students")
    print(f"  2.5-3.0 (Average):   {sum((predictions >= 2.5) & (predictions < 3.0))} students")
    print(f"  <2.5 (At Risk):      {sum(predictions < 2.5)} students")

def save_all_predictions(df, model_data):
    """Save predictions for all students"""
    print("\n" + "="*70)
    print("GENERATING ALL PREDICTIONS")
    print("="*70)
    
    best_model_name = model_data['best_model_name']
    best_model = model_data['models'][best_model_name]
    
    # Prepare features
    metadata_cols = ['student_id', 'target_semester', 'major', 'target_gpa']
    X = df.drop(metadata_cols, axis=1)
    
    # Make predictions
    predictions = best_model.predict(X)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'student_id': df['student_id'],
        'semester': df['target_semester'],
        'major': df['major'],
        'actual_gpa': df['target_gpa'],
        'predicted_gpa': predictions,
        'prediction_error': abs(df['target_gpa'] - predictions),
        'model_used': best_model_name
    })
    
    # Add probability ranges
    def get_range(gpa):
        if gpa >= 3.5:
            return "3.5-4.0 (High)"
        elif gpa >= 3.0:
            return "3.0-3.5 (Good)"
        elif gpa >= 2.5:
            return "2.5-3.0 (Average)"
        else:
            return "<2.5 (At Risk)"
    
    predictions_df['probability_range'] = predictions_df['predicted_gpa'].apply(get_range)
    
    # Save to file
    predictions_df.to_csv(os.path.join(OUTPUT_PATH, 'all_predictions.csv'), index=False)
    print(f"✓ Saved all_predictions.csv ({len(predictions_df)} predictions)")
    
    # Summary statistics
    print(f"\nOverall Prediction Statistics:")
    print(f"  Mean Absolute Error: {predictions_df['prediction_error'].mean():.4f}")
    print(f"  Median Error: {predictions_df['prediction_error'].median():.4f}")
    print(f"  Max Error: {predictions_df['prediction_error'].max():.4f}")


def main():
    """Main execution"""
    # Load data and models
    print("Loading models and data...")
    model_data = load_models()
    df = load_processed_data()
    
    # Generate sample predictions
    generate_sample_predictions(df, model_data, num_samples=15)
    
    # Save all predictions
    save_all_predictions(df, model_data)
    
    print("\n" + "="*70)
    print("✓ PREDICTIONS COMPLETE!")
    print("="*70)
    print(f"\nFiles saved to: {OUTPUT_PATH}")
    print("  - all_predictions.csv")


if __name__ == "__main__":
    main()
