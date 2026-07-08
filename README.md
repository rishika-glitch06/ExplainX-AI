# ExplainX AI

ExplainX AI is a Flask-based Explainable Artificial Intelligence dashboard that allows users to upload any classification dataset, train a Random Forest model, evaluate its performance, visualize feature importance, generate SHAP explanations, and download a professional PDF report.

---

## Features

- Upload CSV datasets
- Dataset preview
- Target column selection
- Random Forest Classification
- Accuracy, Precision, Recall & F1 Score
- Feature Importance Visualization
- Confusion Matrix
- Class Distribution
- SHAP Summary Plot
- SHAP Feature Importance
- SHAP Waterfall Plot
- PDF Report Generation
- Interactive Plotly Visualizations
- Responsive Dashboard

---

## Tech Stack

### Backend

- Python
- Flask
- Scikit-learn
- SHAP
- Pandas
- NumPy

### Frontend

- HTML5
- CSS3
- Jinja2

### Visualization

- Plotly
- Matplotlib

### Report Generation

- ReportLab

---

## Project Structure

```
ExplainXAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── uploads/
├── reports/
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── graph.html
│
└── utils/
    ├── model.py
    ├── plots.py
    ├── report.py
    └── shap_utils.py
```

---

## Installation

```bash
git clone https://github.com/rishika-glitch06/SHAP-Explainability-Dashboard.git

cd SHAP-Explainability-Dashboard

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

---

## Screenshots

Create a folder named **screenshots** and add:

- Home Page
- Dataset Preview
- Dashboard
- SHAP Summary Plot

---

## Future Improvements

- Multiple ML Algorithms
- Regression Support
- Model Comparison
- Hyperparameter Tuning
- Model Export
- Dark Mode

---

## Author

**Rishika Kumari**

B.Tech CSE (Data Science)

Explainable Artificial Intelligence Project