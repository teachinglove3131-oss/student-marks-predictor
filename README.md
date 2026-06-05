# 🎓 Student Marks Prediction System

A Machine Learning web app built with **Streamlit** that predicts student final exam marks based on 6 key academic factors.

---

## 🚀 Deploy on Streamlit Cloud (Free — Share with anyone!)

### Step 1: Upload to GitHub
1. Create a free account at [github.com](https://github.com)
2. Create a **New Repository** → Name it `student-marks-predictor`
3. Upload all files from this folder:
   - `app.py`
   - `student.csv`
   - `student_model.pkl`
   - `train_model.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy on Streamlit Cloud
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your GitHub repo → Branch: `main` → Main file: `app.py`
5. Click **"Deploy!"**
6. In ~2 minutes your app is LIVE with a shareable link like:
   `https://your-name-student-marks-predictor.streamlit.app`

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 File Structure

```
student-marks-predictor/
├── app.py               ← Main Streamlit app
├── train_model.py       ← Retrain model if needed
├── student.csv          ← Dataset (300 records)
├── student_model.pkl    ← Trained ML model
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🧠 Model Details

| Property       | Value                  |
|----------------|------------------------|
| Algorithm      | Linear Regression      |
| R² Score       | 93.9% accuracy         |
| MAE            | ±2.48 marks            |
| Training data  | 300 student records    |
| Features used  | 6                      |

## 📊 Features

| Input Feature       | Effect on Marks |
|---------------------|-----------------|
| Study Hours/day     | +3.53 per hour  |
| Attendance %        | +0.25 per %     |
| Previous Marks      | +0.30 per mark  |
| Practice Test Score | +0.14 per point |
| Sleep Hours/night   | +1.06 per hour  |
| Social Media Hours  | -1.13 per hour  |

---

## 📌 Category Guide

| Marks | Category      |
|-------|---------------|
| 80+   | 🏆 Excellent  |
| 60–79 | 👍 Good       |
| 40–59 | ⚠️ Below Avg  |
| <40   | 🚨 Poor       |
