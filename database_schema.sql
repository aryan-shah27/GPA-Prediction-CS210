
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
