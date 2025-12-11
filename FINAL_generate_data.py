"""
GPA Prediction System Data Generation
Generates complete synthetic student data
"""

import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# OUTPUT PATH
OUTPUT_PATH = r"C:\python\files"

class FinalDataGenerator:
    def __init__(self, num_students=1000, num_semesters=8):
        self.num_students = num_students
        self.num_semesters = num_semesters
        self.majors = ['Computer Science', 'Engineering', 'Mathematics', 
                       'Business', 'Biology', 'Physics', 'Economics']
        self.departments = ['CS', 'MATH', 'ENG', 'BUS', 'BIO', 'PHY', 'ECON', 'HUM']
        self.grade_points = {'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 
                            'B-': 2.7, 'C+': 2.3, 'C': 2.0, 'C-': 1.7,
                            'D': 1.0, 'F': 0.0}
        
    def generate_students(self):
        """Generate student demographic data"""
        students = []
        for i in range(1, self.num_students + 1):
            student = {
                'student_id': i,
                'major': random.choice(self.majors),
                'start_year': random.choice([2020, 2021, 2022]),
                'grad_year': None
            }
            student['grad_year'] = student['start_year'] + 4
            students.append(student)
        
        return pd.DataFrame(students)
    
    def generate_courses(self):
        """Generate course catalog with difficulty levels"""
        courses = []
        course_id = 1
        
        for dept in self.departments:
            for level in [100, 200, 300, 400]:
                num_courses = random.randint(5, 10)
                for i in range(num_courses):
                    difficulty = self._calculate_difficulty(level)
                    credits = random.choice([3, 3, 3, 4, 4])
                    
                    courses.append({
                        'course_id': course_id,
                        'dept': dept,
                        'course_num': level + i,
                        'difficulty_level': difficulty,
                        'credits': credits
                    })
                    course_id += 1
        
        return pd.DataFrame(courses)
    
    def _calculate_difficulty(self, level):
        """Calculate difficulty based on course level"""
        base_difficulty = {
            100: random.uniform(1.0, 3.0),
            200: random.uniform(2.0, 4.5),
            300: random.uniform(3.5, 6.5),
            400: random.uniform(5.0, 8.0)
        }
        return round(base_difficulty[level], 2)
    
    def generate_enrollments(self, students_df, courses_df):
        """Generate course enrollments with realistic grade patterns"""
        enrollments = []
        
        for _, student in students_df.iterrows():
            student_id = student['student_id']
            major = student['major']
            
            base_ability = np.random.normal(3.0, 0.5)
            major_dept = self._get_major_dept(major)
            
            semester_num = 0
            
            for year in range(student['start_year'], student['grad_year']):
                for sem in ['Fall', 'Spring']:
                    semester_num += 1
                    semester = f"{sem} {year}"
                    
                    num_courses = random.randint(4, 6)
                    
                    available_courses = self._filter_courses_by_level(
                        courses_df, semester_num, major_dept
                    )
                    
                    selected_courses = available_courses.sample(
                        min(num_courses, len(available_courses))
                    )
                    
                    for _, course in selected_courses.iterrows():
                        grade_letter, grade_point = self._generate_grade(
                            student_id, course, base_ability, major_dept, semester_num
                        )
                        
                        enrollments.append({
                            'student_id': student_id,
                            'course_id': course['course_id'],
                            'semester': semester,
                            'semester_num': semester_num,
                            'grade_letter': grade_letter,
                            'grade_point': grade_point
                        })
                    
                    if semester_num >= self.num_semesters:
                        break
                
                if semester_num >= self.num_semesters:
                    break
        
        return pd.DataFrame(enrollments)
    
    def _get_major_dept(self, major):
        """Map major to primary department"""
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
    
    def _filter_courses_by_level(self, courses_df, semester_num, major_dept):
        if semester_num <= 2:
            level_filter = courses_df['course_num'] <= 299
        elif semester_num <= 4:
            level_filter = (courses_df['course_num'] >= 200) & (courses_df['course_num'] <= 399)
        elif semester_num <= 6:
            level_filter = courses_df['course_num'] >= 300
        else:
            level_filter = courses_df['course_num'] >= 300
        
        filtered = courses_df[level_filter].copy()
        
        if random.random() < 0.7:
            major_courses = filtered[filtered['dept'] == major_dept]
            if len(major_courses) > 0:
                return major_courses
        
        return filtered
    
    def _generate_grade(self, student_id, course, base_ability, major_dept, semester_num):
        """Generate realistic grade based on multiple factors"""
        difficulty = course['difficulty_level']
        is_major_course = (course['dept'] == major_dept)
        
        performance = base_ability
        
        if is_major_course:
            performance += random.uniform(0.1, 0.5)
        
        performance -= (difficulty / 10) * random.uniform(0.3, 0.7)
        performance += (semester_num / 20) * random.uniform(0, 0.3)
        performance += np.random.normal(0, 0.3)
        
        performance = np.clip(performance, 0.0, 4.0)
        
        if performance >= 3.85:
            return 'A', 4.0
        elif performance >= 3.5:
            return 'A-', 3.7
        elif performance >= 3.15:
            return 'B+', 3.3
        elif performance >= 2.85:
            return 'B', 3.0
        elif performance >= 2.5:
            return 'B-', 2.7
        elif performance >= 2.15:
            return 'C+', 2.3
        elif performance >= 1.85:
            return 'C', 2.0
        elif performance >= 1.5:
            return 'C-', 1.7
        elif performance >= 1.0:
            return 'D', 1.0
        else:
            return 'F', 0.0
    
    def calculate_semester_gpa(self, enrollments_df, courses_df):
        """Calculate GPA for each semester"""
        merged = enrollments_df.merge(courses_df[['course_id', 'credits']], on='course_id')
        
        semester_gpa = []
        
        for (student_id, semester), group in merged.groupby(['student_id', 'semester']):
            total_points = (group['grade_point'] * group['credits']).sum()
            total_credits = group['credits'].sum()
            gpa = total_points / total_credits if total_credits > 0 else 0.0
            
            semester_gpa.append({
                'student_id': student_id,
                'semester': semester,
                'gpa': round(gpa, 3),
                'credits': total_credits
            })
        
        return pd.DataFrame(semester_gpa)
    
    def generate_all_data(self):
        """Generate complete dataset"""
        print("="*70)
        print("GPA PREDICTION SYSTEM - FINAL DATA GENERATION")
        print("="*70)
        
        print("\nGenerating students...")
        students = self.generate_students()
        print(f"✓ Generated {len(students)} students")
        
        print("\nGenerating courses...")
        courses = self.generate_courses()
        print(f"✓ Generated {len(courses)} courses")
        
        print("\nGenerating enrollments (this may take a minute)...")
        enrollments = self.generate_enrollments(students, courses)
        print(f"✓ Generated {len(enrollments)} enrollments")
        
        print("\nCalculating semester GPAs...")
        semester_gpa = self.calculate_semester_gpa(enrollments, courses)
        print(f"✓ Generated {len(semester_gpa)} semester records")
        
        return students, courses, enrollments, semester_gpa


def save_data(students, courses, enrollments, semester_gpa):
    """Save generated data to CSV files"""
    print("\n" + "="*70)
    print("SAVING DATA TO FILES")
    print("="*70)
    
    # Create directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    students.to_csv(os.path.join(OUTPUT_PATH, 'students.csv'), index=False)
    print(f"✓ Saved students.csv")
    
    courses.to_csv(os.path.join(OUTPUT_PATH, 'courses.csv'), index=False)
    print(f"✓ Saved courses.csv")
    
    enrollments.to_csv(os.path.join(OUTPUT_PATH, 'enrollments.csv'), index=False)
    print(f"✓ Saved enrollments.csv")
    
    semester_gpa.to_csv(os.path.join(OUTPUT_PATH, 'semester_gpa.csv'), index=False)
    print(f"✓ Saved semester_gpa.csv")
    
    print(f"\nAll files saved to: {OUTPUT_PATH}")
    
    print("\n" + "="*70)
    print("DATA GENERATION STATISTICS")
    print("="*70)
    print(f"Total Students: {len(students)}")
    print(f"Total Courses: {len(courses)}")
    print(f"Total Enrollments: {len(enrollments)}")
    print(f"Total Semester Records: {len(semester_gpa)}")
    print(f"\nMajor Distribution:")
    print(students['major'].value_counts())
    print(f"\nAverage GPA: {semester_gpa['gpa'].mean():.3f}")
    print(f"GPA Std Dev: {semester_gpa['gpa'].std():.3f}")


if __name__ == "__main__":
    generator = FinalDataGenerator(num_students=1000, num_semesters=8)
    students, courses, enrollments, semester_gpa = generator.generate_all_data()
    save_data(students, courses, enrollments, semester_gpa)
    
    print("\n" + "="*70)
    print("✓ DATA GENERATION COMPLETE!")
    print("="*70)
    print("\nNext step: Run FINAL_feature_engineering.py")
