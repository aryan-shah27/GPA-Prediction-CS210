"""
GPA Prediction System Feature Engineering
Creates ML-ready features from raw data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# OUTPUT PATH
OUTPUT_PATH = r"C:\python\files"

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load data from CSV files"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        students_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'students.csv'))
        courses_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'courses.csv'))
        enrollments_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'enrollments.csv'))
        semester_gpa_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'semester_gpa.csv'))
        
        print(f"✓ Loaded {len(students_df)} students")
        print(f"✓ Loaded {len(courses_df)} courses")
        print(f"✓ Loaded {len(enrollments_df)} enrollments")
        print(f"✓ Loaded {len(semester_gpa_df)} semester GPA records")
        
        return students_df, courses_df, enrollments_df, semester_gpa_df
    
    def create_features(self, students_df, courses_df, enrollments_df, semester_gpa_df):
        print("\n" + "="*70)
        print("ENGINEERING FEATURES")
        print("="*70)
        
        merged = enrollments_df.merge(courses_df, on='course_id')
        merged = merged.merge(students_df, on='student_id')
        
        student_features = []
        
        for student_id in students_df['student_id'].unique():
            student_data = merged[merged['student_id'] == student_id]
            student_info = students_df[students_df['student_id'] == student_id].iloc[0]
            student_gpa = semester_gpa_df[semester_gpa_df['student_id'] == student_id]
            
            if len(student_gpa) < 2:
                continue
            
            for i in range(1, len(student_gpa)):
                features = self._create_student_semester_features(
                    student_id, i, student_data, student_info, student_gpa
                )
                student_features.append(features)
        
        features_df = pd.DataFrame(student_features)
        print(f"✓ Created {len(features_df)} feature records")
        print(f"✓ Total features: {len(features_df.columns) - 4}")  # Exclude metadata
        
        return features_df
    
    def _create_student_semester_features(self, student_id, semester_idx, 
                                         student_data, student_info, student_gpa):
        
        current_semester = student_gpa.iloc[semester_idx]['semester']
        historical_gpa = student_gpa.iloc[:semester_idx]
        
        historical_enrollments = student_data[
            student_data['semester_num'] <= semester_idx
        ]
        
        features = {
            'student_id': student_id,
            'target_semester': current_semester,
            'target_gpa': student_gpa.iloc[semester_idx]['gpa'],
            'major': student_info['major'],
            
            # Basic progression
            'semesters_completed': semester_idx,
            
            # Historical GPA features
            'current_gpa': historical_gpa.iloc[-1]['gpa'],
            'avg_gpa': historical_gpa['gpa'].mean(),
            'min_gpa': historical_gpa['gpa'].min(),
            'max_gpa': historical_gpa['gpa'].max(),
            'gpa_std': historical_gpa['gpa'].std() if len(historical_gpa) > 1 else 0,
            'gpa_trend': self._calculate_trend(historical_gpa['gpa'].values),
            
            # Recent performance
            'recent_gpa_avg': historical_gpa.iloc[-2:]['gpa'].mean() if len(historical_gpa) >= 2 else historical_gpa['gpa'].mean(),
            
            # Course difficulty features
            'avg_difficulty': historical_enrollments['difficulty_level'].mean(),
            'max_difficulty': historical_enrollments['difficulty_level'].max(),
            'difficulty_std': historical_enrollments['difficulty_level'].std(),
            
            # Credit load features
            'avg_credits_per_sem': historical_gpa['credits'].mean(),
            'max_credits': historical_gpa['credits'].max(),
            'total_credits': historical_gpa['credits'].sum(),
            
            # Performance by difficulty
            'performance_easy': self._performance_by_difficulty(historical_enrollments, 0, 3),
            'performance_medium': self._performance_by_difficulty(historical_enrollments, 3, 6),
            'performance_hard': self._performance_by_difficulty(historical_enrollments, 6, 10),
            
            # Department performance
            'major_dept_gpa': self._performance_by_dept(historical_enrollments, self._get_major_dept(student_info['major'])),
            'non_major_gpa': self._performance_by_dept(historical_enrollments, self._get_major_dept(student_info['major']), inverse=True),
            
            # Grade distribution
            'pct_A': self._grade_percentage(historical_enrollments, ['A', 'A-']),
            'pct_B': self._grade_percentage(historical_enrollments, ['B+', 'B', 'B-']),
            'pct_C_or_below': self._grade_percentage(historical_enrollments, ['C+', 'C', 'C-', 'D', 'F']),
            
            # Course level distribution
            'pct_upper_level': self._course_level_percentage(historical_enrollments, 300),
            
            # Consistency metrics
            'grade_consistency': 1 / (historical_enrollments['grade_point'].std() + 0.1),
        }
        
        return features
    
    def _calculate_trend(self, values):
        """Calculate linear trend of GPA over time"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        return z[0]
    
    def _performance_by_difficulty(self, enrollments, min_diff, max_diff):
        """Calculate average GPA for courses in difficulty range"""
        filtered = enrollments[
            (enrollments['difficulty_level'] >= min_diff) & 
            (enrollments['difficulty_level'] < max_diff)
        ]
        return filtered['grade_point'].mean() if len(filtered) > 0 else enrollments['grade_point'].mean()
    
    def _performance_by_dept(self, enrollments, dept, inverse=False):
        """Calculate average GPA for major/non-major courses"""
        if inverse:
            filtered = enrollments[enrollments['dept'] != dept]
        else:
            filtered = enrollments[enrollments['dept'] == dept]
        return filtered['grade_point'].mean() if len(filtered) > 0 else enrollments['grade_point'].mean()
    
    def _get_major_dept(self, major):
        """Map major to department code"""
        mapping = {
            'Computer Science': 'CS',
            'Engineering': 'ENG',
            'Mathematics': 'MATH',
            'Business': 'BUS',
            'Biology': 'BIO',
            'Physics': 'PHY',
            'Economics': 'ECON'
        }
        return mapping.get(major, 'CS')
    
    def _grade_percentage(self, enrollments, grades):
        """Calculate percentage of grades in list"""
        total = len(enrollments)
        count = len(enrollments[enrollments['grade_letter'].isin(grades)])
        return count / total if total > 0 else 0
    
    def _course_level_percentage(self, enrollments, min_level):
        """Calculate percentage of courses at or above level"""
        total = len(enrollments)
        count = len(enrollments[enrollments['course_num'] >= min_level])
        return count / total if total > 0 else 0
    
    def prepare_ml_data(self, features_df):
        """Prepare data for machine learning"""
        print("\n" + "="*70)
        print("PREPARING ML DATA")
        print("="*70)
        
        # Separate features and target
        X = features_df.drop(['student_id', 'target_semester', 'target_gpa', 'major'], axis=1)
        y = features_df['target_gpa']
        
        # Encode major
        major_encoded = self.label_encoder.fit_transform(features_df['major'])
        X['major_encoded'] = major_encoded
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print(f"✓ Feature matrix shape: {X_scaled_df.shape}")
        print(f"✓ Target variable shape: {y.shape}")
        print(f"✓ Target GPA - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
        
        metadata = {
            'feature_columns': X.columns.tolist(),
            'major_encoder': self.label_encoder,
            'scaler': self.scaler
        }
        
        return X_scaled_df, y, features_df[['student_id', 'target_semester', 'major']], metadata
    
    def save_processed_data(self, X, y, metadata_df, metadata):
        """Save processed data and metadata"""
        print("\n" + "="*70)
        print("SAVING PROCESSED DATA")
        print("="*70)
        
        # Combine for saving
        processed_df = X.copy()
        processed_df['target_gpa'] = y.values
        processed_df['student_id'] = metadata_df['student_id'].values
        processed_df['target_semester'] = metadata_df['target_semester'].values
        processed_df['major'] = metadata_df['major'].values
        
        # Save to CSV
        processed_df.to_csv(os.path.join(OUTPUT_PATH, 'processed_features.csv'), index=False)
        print(f"✓ Saved processed_features.csv")
        
        # Save metadata
        with open(os.path.join(OUTPUT_PATH, 'feature_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Saved feature_metadata.pkl")
        
        print(f"\nFiles saved to: {OUTPUT_PATH}")


def main():
    """Main execution"""
    engineer = FeatureEngineer()
    
    # Load data
    students_df, courses_df, enrollments_df, semester_gpa_df = engineer.load_data()
    
    # Create features
    features_df = engineer.create_features(students_df, courses_df, enrollments_df, semester_gpa_df)
    
    # Prepare ML data
    X, y, metadata_df, metadata = engineer.prepare_ml_data(features_df)
    
    # Save processed data
    engineer.save_processed_data(X, y, metadata_df, metadata)
    
    print("\n" + "="*70)
    print("✓ FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\nTotal Features: {X.shape[1]}")
    print(f"Total Samples: {X.shape[0]}")
    print("\nNext step: Run FINAL_train_models.py")


if __name__ == "__main__":
    main()
