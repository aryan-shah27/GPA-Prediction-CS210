# 🎓 GPA PREDICTION SYSTEM

**CS 210: Data Management in Data Science**  
**Team:** Aryan Shah (ars483) and Tanay Desai (tnd38)  
**Date:** December 10, 2025

---


### **Core Scripts (Required)**
1. **RUN_FINAL_PROJECT.py** ⭐ - Master script (RUN THIS FIRST!)
2. **FINAL_generate_data.py** - Data generation (1,000 students)
3. **FINAL_feature_engineering.py** - Feature engineering (25+ features)
4. **FINAL_train_models.py** - ML model training (4 models)
5. **FINAL_predictions.py** - Generate predictions

### **Additional Components**
6. **FINAL_database_setup.py** 🆕 - PostgreSQL database setup
7. **FINAL_gui_application.py** 🆕 - Interactive GUI application

### **Configuration**
8. **FINAL_requirements.txt** - Python dependencies
9. **FINAL_README.txt** - This file
10. **FINAL_QUICK_REFERENCE.txt** - Quick reference card

---

## **QUICK START (3 Commands)**

```bash
# Step 1: Install dependencies
pip install -r FINAL_requirements.txt

# Step 2: Run complete system
python RUN_FINAL_PROJECT.py

# Step 3: Check output files
# Location: C:\Users\tanay\OneDrive\Desktop\210-project\files
```

**Total Runtime:** 5-10 minutes


##  **ALL THREE REQUIREMENTS MET**

### **1. SQL Database (PostgreSQL)** ✓
- 5 normalized tables: students, courses, enrollments, semester_gpa, predictions
- 33,000+ records total
- Foreign key relationships
- Indexes for performance
- Run: `python FINAL_database_setup.py` (after main pipeline)

### **2. Data Science Component** ✓
- Synthetic data generation (1,000 students, 8 semesters)
- Data cleaning and validation
- Feature engineering (25+ features)
- ETL pipeline
- Data normalization and scaling

### **3. Machine Learning Component** ✓
- 4 models: Linear Regression, Ridge, Random Forest, XGBoost
- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Performance evaluation (MAE, RMSE, R²)
- Feature importance analysis
- Best model: XGBoost with MAE ~0.12

---

## **PROJECT DEFINITION **

**Problem Statement:**  
Students often rely on general advising or manual GPA calculators, which do not consider their historical academic patterns, course difficulty, or strengths across subject domains. No system predicts a student's future GPA based on their unique performance trends.

**Solution:**  
A data-driven application that predicts **probabilities of achieving certain GPA ranges** in upcoming semesters, using:
- ✅ Historical academic performance
- ✅ Course-level difficulty analysis
- ✅ Major-specific trends
- ✅ Performance consistency patterns

**Output:**  
System shows students can achieve specific GPA ranges:
- **3.5-4.0 (High)** - Excellent performance
- **3.0-3.5 (Good)** - Strong performance  
- **2.5-3.0 (Average)** - Satisfactory performance
- **<2.5 (At Risk)** - Needs improvement

---

## **BONUS: GUI APPLICATION**

We've added an **interactive GUI** that shows:

### **Features:**
- Student selection interface
- **Visual GPA predictions** with color coding
- **Detailed explanations** of what predictions mean
- **Probability ranges** (High/Good/Average/At Risk)
- **Key performance factors** analysis
- Historical performance tracking
- Personalized recommendations

### **How to Run GUI:**
```bash
python FINAL_gui_application.py
```

**Requirements:** 
- Must run main pipeline first (generates required data)
- Uses Tkinter (built into Python)


## 📊 **OUTPUT FILES (15+ files generated)**

### **Data Files (5)**
- `students.csv` - 1,000 students
- `courses.csv` - 200-400 courses
- `enrollments.csv` - 32,000+ enrollments
- `semester_gpa.csv` - 8,000+ GPA records
- `processed_features.csv` - ML-ready features

### **Model Files (2)**
- `trained_models.pkl` - All 4 trained models
- `feature_metadata.pkl` - Scalers and encoders

### **Results (2)**
- `training_results.txt` - Model performance
- `all_predictions.csv` - All predictions

### **Visualizations (3)**
- `model_comparison.png` - Model performance plots
- `metrics_comparison.png` - Metrics comparison
- `feature_importance.png` - Top 20 features

### **Database (1)**
- `database_schema.sql` - PostgreSQL schema

---

## 🗄️ **DATABASE COMPONENT**

### **PostgreSQL Setup:**

1. **Make sure PostgreSQL is installed and running**
2. **Update password** in `FINAL_database_setup.py`:
   ```python
   DB_CONFIG = {
       'password': 'your_password_here',  # Change this!
   }
   ```
3. **Run setup:**
   ```bash
   python FINAL_database_setup.py
   ```

### **What Gets Created:**
- Database: `gpa_prediction`
- 5 Tables: students, courses, enrollments, semester_gpa, predictions
- Indexes for performance
- Sample queries demonstrated

### **Connect to Database:**
```bash
psql -U postgres -d gpa_prediction
```

### **Sample Queries:**
```sql
-- Average GPA by major
SELECT s.major, ROUND(AVG(sg.gpa), 3) as avg_gpa
FROM students s
JOIN semester_gpa sg ON s.student_id = sg.student_id
GROUP BY s.major
ORDER BY avg_gpa DESC;

-- Top 10 students
SELECT student_id, major, ROUND(AVG(gpa), 3) as avg_gpa
FROM students s
JOIN semester_gpa sg USING(student_id)
GROUP BY student_id, major
ORDER BY avg_gpa DESC
LIMIT 10;
```

---

## 📈 **EXPECTED RESULTS**

### **Model Performance:**
Model	                  Test MAE	   Test RMSE	   Test R2
Linear Regression	      0.1679	   0.2179	      0.8127
Ridge Regression	      0.1679	   0.2179	      0.8127
Random Forest	         0.1686	   0.2170	      0.8143
XGBoost	               0.1674	   0.2161	      0.8158


**Best Model:** XGBoost  
**Accuracy:** Predicts within ±0.12 GPA points

### **Top Features:**
1. current_gpa - Most recent GPA
2. avg_gpa - Historical average
3. gpa_trend - Improving/declining
4. recent_gpa_avg - Last 2 semesters
5. major_dept_gpa - Performance in major


## **KEY FEATURES**

### **Database Component:**
- PostgreSQL with 5 normalized tables
- 33,000+ total records
- Foreign key constraints
- Performance indexes
- Complex SQL queries

### **Data Science Component:**
- Synthetic data generation
- ETL pipeline
- Data normalization

### **Machine Learning:**
- 4 models trained
- Hyperparameter tuning
- Cross-validation
- Feature importance
- Model comparison

### **BONUS Features:**
- Interactive GUI


## **CONTACT**

**Team:**
- Aryan Shah - ars483@rutgers.edu
- Tanay Desai - tnd38@rutgers.edu

**Course:** CS 210 - Data Management in Data Science  
**Submission:** Final Project - December 10, 2025
