"""
GPA Prediction System PostgreSQL Database Setup
Creates database, tables, and loads data from CSV files
"""

import psycopg2
from psycopg2 import sql
import pandas as pd
import os

# OUTPUT PATH
OUTPUT_PATH = r"C:\python\files"

# DATABASE CONFIGURATION
DB_CONFIG = {
    'dbname': 'gpa_prediction',
    'user': 'postgres',
    'password': 'aryan',  # Change this to your PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}

class DatabaseSetup:
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to PostgreSQL server"""
        try:
            # Connect to default postgres database 
            conn = psycopg2.connect(
                dbname='postgres',
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port']
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Drop database if exists and create new one
            print("Creating database...")
            cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG['dbname']}")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            print(f"✓ Database '{DB_CONFIG['dbname']}' created")
            
            cursor.close()
            conn.close()
            
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            print("✓ Connected to database")
            
        except Exception as e:
            print(f"✗ Database connection error: {e}")
            print("\nMake sure PostgreSQL is running and credentials are correct!")
            print("Update DB_CONFIG in this file with your PostgreSQL password.")
            raise
    
    def create_tables(self):
        """Create database tables"""
        print("\n" + "="*70)
        print("CREATING TABLES")
        print("="*70)
        
        # Students table
        self.cursor.execute("""
            CREATE TABLE students (
                student_id INTEGER PRIMARY KEY,
                major VARCHAR(100) NOT NULL,
                start_year INTEGER NOT NULL,
                grad_year INTEGER NOT NULL
            )
        """)
        print("✓ Created table: students")
        
        # Courses table
        self.cursor.execute("""
            CREATE TABLE courses (
                course_id INTEGER PRIMARY KEY,
                dept VARCHAR(10) NOT NULL,
                course_num INTEGER NOT NULL,
                difficulty_level DECIMAL(4,2) NOT NULL,
                credits INTEGER NOT NULL
            )
        """)
        print("✓ Created table: courses")
        
        # Enrollments table
        self.cursor.execute("""
            CREATE TABLE enrollments (
                enrollment_id SERIAL PRIMARY KEY,
                student_id INTEGER REFERENCES students(student_id),
                course_id INTEGER REFERENCES courses(course_id),
                semester VARCHAR(20) NOT NULL,
                semester_num INTEGER NOT NULL,
                grade_letter VARCHAR(2) NOT NULL,
                grade_point DECIMAL(3,2) NOT NULL
            )
        """)
        print("✓ Created table: enrollments")
        
        # Semester GPA table
        self.cursor.execute("""
            CREATE TABLE semester_gpa (
                gpa_id SERIAL PRIMARY KEY,
                student_id INTEGER REFERENCES students(student_id),
                semester VARCHAR(20) NOT NULL,
                gpa DECIMAL(4,3) NOT NULL,
                credits INTEGER NOT NULL
            )
        """)
        print("✓ Created table: semester_gpa")
        
        # Predictions table
        self.cursor.execute("""
            CREATE TABLE predictions (
                prediction_id SERIAL PRIMARY KEY,
                student_id INTEGER REFERENCES students(student_id),
                semester VARCHAR(20) NOT NULL,
                predicted_gpa DECIMAL(4,3) NOT NULL,
                actual_gpa DECIMAL(4,3),
                prediction_error DECIMAL(4,3),
                probability_range VARCHAR(30),
                model_used VARCHAR(50),
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ Created table: predictions")
        
        # Create indexes
        print("\nCreating indexes...")
        self.cursor.execute("CREATE INDEX idx_enrollments_student ON enrollments(student_id)")
        self.cursor.execute("CREATE INDEX idx_enrollments_course ON enrollments(course_id)")
        self.cursor.execute("CREATE INDEX idx_semester_gpa_student ON semester_gpa(student_id)")
        self.cursor.execute("CREATE INDEX idx_predictions_student ON predictions(student_id)")
        print("✓ Created indexes")
        
        self.conn.commit()
    
    def load_data(self):
        """Load data from CSV files into database"""
        print("\n" + "="*70)
        print("LOADING DATA FROM CSV FILES")
        print("="*70)
        
        # Load students
        students_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'students.csv'))
        for _, row in students_df.iterrows():
            self.cursor.execute("""
                INSERT INTO students (student_id, major, start_year, grad_year)
                VALUES (%s, %s, %s, %s)
            """, (int(row['student_id']), row['major'], int(row['start_year']), int(row['grad_year'])))
        print(f"✓ Loaded {len(students_df)} students")
        
        # Load courses
        courses_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'courses.csv'))
        for _, row in courses_df.iterrows():
            self.cursor.execute("""
                INSERT INTO courses (course_id, dept, course_num, difficulty_level, credits)
                VALUES (%s, %s, %s, %s, %s)
            """, (int(row['course_id']), row['dept'], int(row['course_num']), 
                  float(row['difficulty_level']), int(row['credits'])))
        print(f"✓ Loaded {len(courses_df)} courses")
        
        # Load enrollments
        enrollments_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'enrollments.csv'))
        for _, row in enrollments_df.iterrows():
            self.cursor.execute("""
                INSERT INTO enrollments (student_id, course_id, semester, semester_num, grade_letter, grade_point)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (int(row['student_id']), int(row['course_id']), row['semester'], 
                  int(row['semester_num']), row['grade_letter'], float(row['grade_point'])))
        print(f"✓ Loaded {len(enrollments_df)} enrollments")
        
        # Load semester GPA
        semester_gpa_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'semester_gpa.csv'))
        for _, row in semester_gpa_df.iterrows():
            self.cursor.execute("""
                INSERT INTO semester_gpa (student_id, semester, gpa, credits)
                VALUES (%s, %s, %s, %s)
            """, (int(row['student_id']), row['semester'], float(row['gpa']), int(row['credits'])))
        print(f"✓ Loaded {len(semester_gpa_df)} semester GPA records")
        
        # Load predictions if available
        predictions_file = os.path.join(OUTPUT_PATH, 'all_predictions.csv')
        if os.path.exists(predictions_file):
            predictions_df = pd.read_csv(predictions_file)
            for _, row in predictions_df.iterrows():
                self.cursor.execute("""
                    INSERT INTO predictions (student_id, semester, predicted_gpa, actual_gpa, 
                                           prediction_error, probability_range, model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (int(row['student_id']), row['semester'], float(row['predicted_gpa']), 
                      float(row['actual_gpa']), float(row['prediction_error']), 
                      row['probability_range'], row['model_used']))
            print(f"✓ Loaded {len(predictions_df)} predictions")
        
        self.conn.commit()
    
    def run_sample_queries(self):
        """Run sample SQL queries to demonstrate database"""
        print("\n" + "="*70)
        print("SAMPLE SQL QUERIES")
        print("="*70)
        
        # Query 1: Average GPA by major
        print("\n1. Average GPA by Major:")
        print("-" * 50)
        self.cursor.execute("""
            SELECT s.major, ROUND(AVG(sg.gpa), 3) as avg_gpa, COUNT(DISTINCT s.student_id) as num_students
            FROM students s
            JOIN semester_gpa sg ON s.student_id = sg.student_id
            GROUP BY s.major
            ORDER BY avg_gpa DESC
        """)
        results = self.cursor.fetchall()
        print(f"{'Major':<20} {'Avg GPA':<10} {'Students':<10}")
        print("-" * 50)
        for row in results:
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10}")
        
        # Query 2: Top 10 students by GPA
        print("\n2. Top 10 Students by Average GPA:")
        print("-" * 50)
        self.cursor.execute("""
            SELECT s.student_id, s.major, ROUND(AVG(sg.gpa), 3) as avg_gpa
            FROM students s
            JOIN semester_gpa sg ON s.student_id = sg.student_id
            GROUP BY s.student_id, s.major
            ORDER BY avg_gpa DESC
            LIMIT 10
        """)
        results = self.cursor.fetchall()
        print(f"{'Student ID':<12} {'Major':<20} {'Avg GPA':<10}")
        print("-" * 50)
        for row in results:
            print(f"{row[0]:<12} {row[1]:<20} {row[2]:<10}")
        
        # Query 3: Course difficulty vs average grade
        print("\n3. Course Difficulty vs Average Grade (Top 10 hardest):")
        print("-" * 70)
        self.cursor.execute("""
            SELECT c.dept, c.course_num, ROUND(c.difficulty_level, 2) as difficulty,
                   ROUND(AVG(e.grade_point), 3) as avg_grade, COUNT(*) as enrollments
            FROM courses c
            JOIN enrollments e ON c.course_id = e.course_id
            GROUP BY c.course_id, c.dept, c.course_num, c.difficulty_level
            HAVING COUNT(*) >= 10
            ORDER BY c.difficulty_level DESC
            LIMIT 10
        """)
        results = self.cursor.fetchall()
        print(f"{'Dept':<8} {'Course':<8} {'Difficulty':<12} {'Avg Grade':<12} {'Enrollments':<12}")
        print("-" * 70)
        for row in results:
            print(f"{row[0]:<8} {row[1]:<8} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
        
        # Query 4: Prediction accuracy
        if os.path.exists(os.path.join(OUTPUT_PATH, 'all_predictions.csv')):
            print("\n4. Prediction Accuracy Summary:")
            print("-" * 50)
            self.cursor.execute("""
                SELECT model_used, 
                       ROUND(AVG(prediction_error), 4) as avg_error,
                       ROUND(MIN(prediction_error), 4) as min_error,
                       ROUND(MAX(prediction_error), 4) as max_error,
                       COUNT(*) as total_predictions
                FROM predictions
                GROUP BY model_used
            """)
            results = self.cursor.fetchall()
            print(f"{'Model':<20} {'Avg Error':<12} {'Min Error':<12} {'Max Error':<12} {'Count':<10}")
            print("-" * 70)
            for row in results:
                print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<10}")
    
    def save_schema_sql(self):
        """Save database schema to SQL file"""
        schema_sql = """
-- GPA Prediction System - PostgreSQL Database Schema

-- Drop existing tables
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS semester_gpa CASCADE;
DROP TABLE IF EXISTS enrollments CASCADE;
DROP TABLE IF EXISTS courses CASCADE;
DROP TABLE IF EXISTS students CASCADE;

-- Create students table
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    major VARCHAR(100) NOT NULL,
    start_year INTEGER NOT NULL,
    grad_year INTEGER NOT NULL
);

-- Create courses table
CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    dept VARCHAR(10) NOT NULL,
    course_num INTEGER NOT NULL,
    difficulty_level DECIMAL(4,2) NOT NULL,
    credits INTEGER NOT NULL
);

-- Create enrollments table
CREATE TABLE enrollments (
    enrollment_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    course_id INTEGER REFERENCES courses(course_id),
    semester VARCHAR(20) NOT NULL,
    semester_num INTEGER NOT NULL,
    grade_letter VARCHAR(2) NOT NULL,
    grade_point DECIMAL(3,2) NOT NULL
);

-- Create semester_gpa table
CREATE TABLE semester_gpa (
    gpa_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    semester VARCHAR(20) NOT NULL,
    gpa DECIMAL(4,3) NOT NULL,
    credits INTEGER NOT NULL
);

-- Create predictions table
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    semester VARCHAR(20) NOT NULL,
    predicted_gpa DECIMAL(4,3) NOT NULL,
    actual_gpa DECIMAL(4,3),
    prediction_error DECIMAL(4,3),
    probability_range VARCHAR(30),
    model_used VARCHAR(50),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_enrollments_student ON enrollments(student_id);
CREATE INDEX idx_enrollments_course ON enrollments(course_id);
CREATE INDEX idx_semester_gpa_student ON semester_gpa(student_id);
CREATE INDEX idx_predictions_student ON predictions(student_id);

-- Sample queries for analysis

-- 1. Average GPA by major
SELECT s.major, ROUND(AVG(sg.gpa), 3) as avg_gpa, COUNT(DISTINCT s.student_id) as num_students
FROM students s
JOIN semester_gpa sg ON s.student_id = sg.student_id
GROUP BY s.major
ORDER BY avg_gpa DESC;

-- 2. Top performing students
SELECT s.student_id, s.major, ROUND(AVG(sg.gpa), 3) as avg_gpa
FROM students s
JOIN semester_gpa sg ON s.student_id = sg.student_id
GROUP BY s.student_id, s.major
ORDER BY avg_gpa DESC
LIMIT 10;

-- 3. Course difficulty analysis
SELECT c.dept, c.course_num, ROUND(c.difficulty_level, 2) as difficulty,
       ROUND(AVG(e.grade_point), 3) as avg_grade, COUNT(*) as enrollments
FROM courses c
JOIN enrollments e ON c.course_id = e.course_id
GROUP BY c.course_id, c.dept, c.course_num, c.difficulty_level
ORDER BY c.difficulty_level DESC;

-- 4. Student performance trend
SELECT s.student_id, s.major, sg.semester, sg.gpa
FROM students s
JOIN semester_gpa sg ON s.student_id = sg.student_id
WHERE s.student_id = 1
ORDER BY sg.semester;

-- 5. Prediction accuracy
SELECT model_used, 
       ROUND(AVG(prediction_error), 4) as avg_error,
       COUNT(*) as total_predictions
FROM predictions
GROUP BY model_used;
"""
        
        with open(os.path.join(OUTPUT_PATH, 'database_schema.sql'), 'w') as f:
            f.write(schema_sql)
        print(f"\n✓ Saved database schema to: database_schema.sql")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("\n✓ Database connection closed")


def main():
    """Main execution"""
    print("="*70)
    print("GPA PREDICTION SYSTEM - POSTGRESQL DATABASE SETUP")
    print("="*70)
    
    print("\n⚠️  IMPORTANT: Make sure PostgreSQL is running!")
    print(f"⚠️  Update DB_CONFIG in this file with your PostgreSQL password")
    print(f"\nDatabase: {DB_CONFIG['dbname']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    
    input("\nPress Enter to continue...")
    
    db = DatabaseSetup()
    
    try:
        # Connect to database
        db.connect()
        
        # Create tables
        db.create_tables()
        
        # Load data
        db.load_data()
        
        # Run sample queries
        db.run_sample_queries()
        
        # Save schema
        db.save_schema_sql()
        
        print("\n" + "="*70)
        print("✓ DATABASE SETUP COMPLETE!")
        print("="*70)
        print(f"\nDatabase: {DB_CONFIG['dbname']}")
        print("Tables created: students, courses, enrollments, semester_gpa, predictions")
        print(f"Total records loaded: ~33,000+")
        print(f"\nYou can now connect to the database using:")
        print(f"  psql -U {DB_CONFIG['user']} -d {DB_CONFIG['dbname']}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is installed and running")
        print("2. Update DB_CONFIG password in this file")
        print("3. Make sure CSV files exist in:", OUTPUT_PATH)
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
