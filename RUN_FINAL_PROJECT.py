"""
GPA Prediction System MASTER RUNNER
Runs the complete pipeline in the correct order

Run this file to execute the entire system!
"""

import subprocess
import sys
import os

OUTPUT_PATH = r"C:\python\files"

def print_header(text):

    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def run_script(script_name, description):
    print_header(description)
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              check=True, 
                              capture_output=False)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Could not find {script_name}")
        print("Make sure all scripts are in the same directory!")
        return False

def verify_output_directory():
    
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        print(f"✓ Output directory ready: {OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"✗ Could not create output directory: {e}")
        return False

def main():
    """Main execution pipeline"""
    print_header("GPA PREDICTION SYSTEM - FINAL SUBMISSION")
    print("This script will run the complete end-to-end pipeline:")
    print("1. Generate synthetic student data (1,000 students)")
    print("2. Engineer features (25+ features)")
    print("3. Train ML models (4 models)")
    print("4. Generate predictions")
    print(f"\nAll output files will be saved to:")
    print(f"  {OUTPUT_PATH}")
    print("\nEstimated time: 5-10 minutes")
    
    input("\nPress Enter to start...")
    
    # Verify output directory
    if not verify_output_directory():
        print("\nPipeline stopped due to error.")
        return
    
    # Step 1: Generate data
    if not run_script('FINAL_generate_data.py', 'STEP 1: DATA GENERATION (1-2 min)'):
        print("\nPipeline stopped due to error.")
        return
    
    # Step 2: Feature engineering
    if not run_script('FINAL_feature_engineering.py', 'STEP 2: FEATURE ENGINEERING (1 min)'):
        print("\nPipeline stopped due to error.")
        return
    
    # Step 3: Train models
    if not run_script('FINAL_train_models.py', 'STEP 3: MODEL TRAINING (3-5 min)'):
        print("\nPipeline stopped due to error.")
        return
    
    # Step 4: Generate predictions
    if not run_script('FINAL_predictions.py', 'STEP 4: GENERATE PREDICTIONS (30 sec)'):
        print("\nPipeline stopped due to error.")
        return
    
    # Optional Step 5: Database setup
    print("\n" + "="*80)
    print("OPTIONAL: PostgreSQL Database Setup")
    print("="*80)
    db_choice = input("\nDo you want to set up PostgreSQL database? (y/n): ").lower()
    if db_choice == 'y':
        if not run_script('FINAL_database_setup.py', 'STEP 5: DATABASE SETUP (Optional)'):
            print("\n⚠️  Database setup failed, but continuing...")
    else:
        print("Skipping database setup. You can run FINAL_database_setup.py later.")
    
    # Final summary
    print_header("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    
    print("✓ Generated Files:")
    print(f"\n  Location: {OUTPUT_PATH}\n")
    print("  Data Files:")
    print("    - students.csv (1,000 students)")
    print("    - courses.csv (200-400 courses)")
    print("    - enrollments.csv (32,000+ records)")
    print("    - semester_gpa.csv (8,000+ records)")
    print("    - processed_features.csv (ML-ready features)")
    print("\n  Model Files:")
    print("    - trained_models.pkl (All 4 trained models)")
    print("    - feature_metadata.pkl (Scalers and encoders)")
    print("\n  Results:")
    print("    - training_results.txt (Model performance summary)")
    print("    - all_predictions.csv (Predictions for all students)")
    print("\n  Visualizations:")
    print("    - model_comparison.png")
    print("    - metrics_comparison.png")
    print("    - feature_importance.png")
    
    print("\n" + "="*80)
    print("ALL FILES READY FOR FINAL SUBMISSION!")
    print("="*80)
    
    print("\n📊 Quick Stats:")
    try:
        import pandas as pd
        results_path = os.path.join(OUTPUT_PATH, 'training_results.txt')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                content = f.read()
                if 'BEST MODEL:' in content:
                    best_model_line = [line for line in content.split('\n') if 'BEST MODEL:' in line][0]
                    print(f"  {best_model_line}")
    except:
        pass
    
    print(f"\n✓ Check your files at: {OUTPUT_PATH}")
    print("\n🎉 Ready for final submission!")
    
    # Optional GUI launch
    print("\n" + "="*80)
    print("BONUS: Interactive GUI Application")
    print("="*80)
    gui_choice = input("\nDo you want to launch the GUI application? (y/n): ").lower()
    if gui_choice == 'y':
        print("\nLaunching GUI... (Close the window when done)")
        run_script('FINAL_gui_application.py', 'GUI APPLICATION')
    else:
        print("You can launch the GUI later by running: python FINAL_gui_application.py")

if __name__ == "__main__":
    main()
