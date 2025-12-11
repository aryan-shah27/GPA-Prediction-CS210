"""
GPA Prediction System GUI Application
Interactive GUI showing predictions with detailed explanations
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle
import os
import random

# OUTPUT PATH
OUTPUT_PATH = r"C:\python\files"

class GPAPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPA Prediction System - Student Dashboard")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load data
        self.load_data()
        
        # Create GUI
        self.create_widgets()
        
    def load_data(self):
        """Load processed data and models"""
        try:
            self.predictions_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'all_predictions.csv'))
            
            self.features_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'processed_features.csv'))
            
            with open(os.path.join(OUTPUT_PATH, 'trained_models.pkl'), 'rb') as f:
                model_data = pickle.load(f)
            self.best_model_name = model_data['best_model_name']
            
            print("✓ Data loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load data: {e}\n\nMake sure to run RUN_FINAL_PROJECT.py first!")
            self.root.destroy()
    
    def create_widgets(self):
        """Create GUI widgets"""

        header = tk.Frame(self.root, bg='#2c3e50', height=80)
        header.pack(fill='x')
        
        title = tk.Label(header, text="🎓 GPA Prediction & Analysis System", 
                        font=('Arial', 24, 'bold'), bg='#2c3e50', fg='white')
        title.pack(pady=20)
        
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Student selection
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 10))
        
        tk.Label(left_panel, text="Select Student", font=('Arial', 14, 'bold'), 
                bg='white').pack(pady=10)

        self.student_listbox = tk.Listbox(left_panel, width=25, height=20, 
                                          font=('Arial', 10))
        self.student_listbox.pack(padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(left_panel)
        scrollbar.pack(side='right', fill='y')
        self.student_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.student_listbox.yview)

        students = self.predictions_df['student_id'].unique()[:50]  # Show first 50
        for student_id in students:
            major = self.predictions_df[self.predictions_df['student_id'] == student_id]['major'].iloc[0]
            self.student_listbox.insert('end', f"ID: {student_id} - {major[:15]}")
        
        tk.Button(left_panel, text="Show Prediction", command=self.show_prediction,
                 bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=10).pack(pady=20)
        
        # Right panel - Results
        self.right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        self.right_panel.pack(side='right', fill='both', expand=True)
        
        self.create_welcome_message()
    
    def create_welcome_message(self):
        """Create welcome message"""
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        welcome_frame = tk.Frame(self.right_panel, bg='white')
        welcome_frame.pack(expand=True)
        
        tk.Label(welcome_frame, text="Welcome to GPA Prediction System", 
                font=('Arial', 18, 'bold'), bg='white').pack(pady=20)
        
        tk.Label(welcome_frame, 
                text="This system predicts future GPA based on:\n\n"
                     "• Historical academic performance\n"
                     "• Course difficulty patterns\n"
                     "• Major-specific trends\n"
                     "• Credit load analysis\n"
                     "• Performance consistency\n\n"
                     "Select a student from the left to see predictions!",
                font=('Arial', 12), bg='white', justify='left').pack(pady=10)
    
    def show_prediction(self):
        """Show prediction for selected student"""
        selection = self.student_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a student first!")
            return
        
        # Get student ID
        selected_text = self.student_listbox.get(selection[0])
        student_id = int(selected_text.split(':')[1].split('-')[0].strip())
        
        student_predictions = self.predictions_df[self.predictions_df['student_id'] == student_id]
        
        if len(student_predictions) == 0:
            messagebox.showinfo("Info", "No predictions available for this student")
            return
        
        # Get latest prediction
        latest_pred = student_predictions.iloc[-1]
        
        # Display results
        self.display_results(student_id, latest_pred, student_predictions)
    
    def display_results(self, student_id, latest_pred, all_predictions):
        
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        canvas = tk.Canvas(self.right_panel, bg='white')
        scrollbar = tk.Scrollbar(self.right_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        header_frame = tk.Frame(scrollable_frame, bg='#3498db', height=60)
        header_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(header_frame, text=f"Student ID: {student_id}", 
                font=('Arial', 16, 'bold'), bg='#3498db', fg='white').pack(pady=5)
        tk.Label(header_frame, text=f"Major: {latest_pred['major']}", 
                font=('Arial', 12), bg='#3498db', fg='white').pack()

        pred_frame = tk.Frame(scrollable_frame, bg='#ecf0f1', relief='raised', bd=2)
        pred_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(pred_frame, text="📊 Next Semester GPA Prediction", 
                font=('Arial', 14, 'bold'), bg='#ecf0f1').pack(pady=10)
        
        predicted_gpa = latest_pred['predicted_gpa']
        color = self.get_gpa_color(predicted_gpa)
        
        tk.Label(pred_frame, text=f"{predicted_gpa:.3f}", 
                font=('Arial', 48, 'bold'), bg='#ecf0f1', fg=color).pack(pady=10)
        
        tk.Label(pred_frame, text=f"Probability Range: {latest_pred['probability_range']}", 
                font=('Arial', 12), bg='#ecf0f1').pack(pady=5)
        
        tk.Label(pred_frame, text=f"Model Used: {latest_pred['model_used']}", 
                font=('Arial', 10, 'italic'), bg='#ecf0f1', fg='gray').pack(pady=5)
        
        explain_frame = tk.Frame(scrollable_frame, bg='white')
        explain_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(explain_frame, text="📖 What This Prediction Means", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w', pady=10)

        explanation = self.generate_explanation(predicted_gpa, latest_pred['probability_range'])
        
        explain_text = tk.Text(explain_frame, height=8, width=70, wrap='word', 
                              font=('Arial', 11), bg='#f9f9f9', relief='flat', padx=10, pady=10)
        explain_text.insert('1.0', explanation)
        explain_text.config(state='disabled')
        explain_text.pack(fill='x', pady=5)
        
        factors_frame = tk.Frame(scrollable_frame, bg='white')
        factors_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(factors_frame, text="🎯 Key Performance Factors", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w', pady=10)
        
        factors = self.get_performance_factors(student_id, latest_pred)
        
        for factor_name, factor_value, factor_desc in factors:
            factor_row = tk.Frame(factors_frame, bg='#f9f9f9', relief='raised', bd=1)
            factor_row.pack(fill='x', pady=5)
            
            tk.Label(factor_row, text=f"• {factor_name}: {factor_value}", 
                    font=('Arial', 11, 'bold'), bg='#f9f9f9').pack(anchor='w', padx=10, pady=5)
            tk.Label(factor_row, text=f"  {factor_desc}", 
                    font=('Arial', 10), bg='#f9f9f9', fg='gray').pack(anchor='w', padx=25, pady=(0, 5))
        
        # Historical performance
        history_frame = tk.Frame(scrollable_frame, bg='white')
        history_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(history_frame, text="📈 Historical Performance", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w', pady=10)
        
        # Show last few predictions
        recent = all_predictions.tail(5)
        for _, row in recent.iterrows():
            hist_row = tk.Frame(history_frame, bg='#f9f9f9')
            hist_row.pack(fill='x', pady=2)
            
            tk.Label(hist_row, text=f"{row['semester']}: ", 
                    font=('Arial', 10, 'bold'), bg='#f9f9f9', width=15, anchor='w').pack(side='left', padx=5)
            tk.Label(hist_row, text=f"Actual: {row['actual_gpa']:.3f}", 
                    font=('Arial', 10), bg='#f9f9f9', width=15, anchor='w').pack(side='left')
            tk.Label(hist_row, text=f"Predicted: {row['predicted_gpa']:.3f}", 
                    font=('Arial', 10), bg='#f9f9f9', width=15, anchor='w').pack(side='left')
            error_color = '#27ae60' if row['prediction_error'] < 0.15 else '#e74c3c'
            tk.Label(hist_row, text=f"Error: {row['prediction_error']:.3f}", 
                    font=('Arial', 10), bg='#f9f9f9', fg=error_color, width=15, anchor='w').pack(side='left')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def get_gpa_color(self, gpa):
        """Get color based on GPA"""
        if gpa >= 3.5:
            return '#27ae60'  # Green
        elif gpa >= 3.0:
            return '#3498db'  # Blue
        elif gpa >= 2.5:
            return '#f39c12'  # Orange
        else:
            return '#e74c3c'  # Red
    
    def generate_explanation(self, predicted_gpa, prob_range):
        """Generate human-readable explanation"""
        
        explanations = {
            "3.5-4.0 (High)": (
                f"Excellent prediction! Your predicted GPA of {predicted_gpa:.3f} indicates strong academic performance. "
                f"This means you're likely to achieve a GPA between 3.5 and 4.0 in the next semester.\n\n"
                f"What this means for you:\n"
                f"• You're on track for Dean's List or honors\n"
                f"• Your current study strategies are working well\n"
                f"• You can consider taking more challenging courses\n"
                f"• Keep maintaining your current performance level"
            ),
            "3.0-3.5 (Good)": (
                f"Good prediction! Your predicted GPA of {predicted_gpa:.3f} shows solid academic performance. "
                f"This means you're likely to achieve a GPA between 3.0 and 3.5 in the next semester.\n\n"
                f"What this means for you:\n"
                f"• You're maintaining a strong academic standing\n"
                f"• You have room to improve to reach honors level\n"
                f"• Consider focusing on subjects where you perform best\n"
                f"• You're doing well overall - keep it up!"
            ),
            "2.5-3.0 (Average)": (
                f"Your predicted GPA of {predicted_gpa:.3f} indicates average academic performance. "
                f"This means you're likely to achieve a GPA between 2.5 and 3.0 in the next semester.\n\n"
                f"What this means for you:\n"
                f"• You're meeting basic academic standards\n"
                f"• There's significant opportunity for improvement\n"
                f"• Consider adjusting your study strategies\n"
                f"• Seek help in courses where you struggle\n"
                f"• Focus on improving consistency"
            ),
            "<2.5 (At Risk)": (
                f"Alert: Your predicted GPA of {predicted_gpa:.3f} indicates you may be at academic risk. "
                f"This means your next semester GPA might fall below 2.5.\n\n"
                f"What this means for you:\n"
                f"• Immediate action needed to improve performance\n"
                f"• Consider meeting with an academic advisor\n"
                f"• Evaluate your course load and difficulty\n"
                f"• Seek tutoring or study groups\n"
                f"• Focus on fundamental courses first"
            )
        }
        
        return explanations.get(prob_range, "Prediction analysis not available.")
    
    def get_performance_factors(self, student_id, latest_pred):
        """Get key performance factors for student"""
        
        # Get student features
        student_features = self.features_df[self.features_df['student_id'] == student_id]
        
        if len(student_features) == 0:
            return [
                ("No detailed data", "N/A", "Performance factors not available for this student")
            ]
        
        latest_features = student_features.iloc[-1]
        
        factors = []
        
        # Current GPA
        if 'current_gpa' in latest_features:
            current_gpa = latest_features['current_gpa']
            factors.append((
                "Current GPA",
                f"{current_gpa:.3f}",
                "Your most recent semester GPA - strong indicator of next semester performance"
            ))
        
        # GPA Trend
        if 'gpa_trend' in latest_features:
            trend = latest_features['gpa_trend']
            trend_text = "Improving" if trend > 0 else "Declining" if trend < 0 else "Stable"
            factors.append((
                "GPA Trend",
                trend_text,
                f"Your GPA has been {trend_text.lower()} over recent semesters"
            ))
        
        # Course Difficulty
        if 'avg_difficulty' in latest_features:
            difficulty = latest_features['avg_difficulty']
            factors.append((
                "Course Difficulty",
                f"{difficulty:.1f}/10",
                "Average difficulty of courses you've taken - higher means more challenging"
            ))
        
        # Major Performance
        if 'major_dept_gpa' in latest_features:
            major_gpa = latest_features['major_dept_gpa']
            factors.append((
                "Major Course Performance",
                f"{major_gpa:.3f}",
                "Your GPA specifically in major-related courses"
            ))
        
        if 'grade_consistency' in latest_features:
            consistency = latest_features['grade_consistency']
            consist_text = "High" if consistency > 5 else "Moderate" if consistency > 3 else "Low"
            factors.append((
                "Performance Consistency",
                consist_text,
                "How consistent your grades are across different courses"
            ))
        
        return factors


def main():
    """Main execution"""
    
    # Check if data files exist
    required_files = [
        'all_predictions.csv',
        'processed_features.csv',
        'trained_models.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(OUTPUT_PATH, file)):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n Please run RUN_FINAL_PROJECT.py first to generate all data!")
        input("\nPress Enter to exit...")
        return
    
    # Create GUI
    root = tk.Tk()
    app = GPAPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
