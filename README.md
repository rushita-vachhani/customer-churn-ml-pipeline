# customer-churn-ml-pipeline
Customer Churn Prediction – End-to-End Machine Learning Pipeline

Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps companies take proactive retention actions.

This project builds an end-to-end machine learning pipeline that predicts whether a telecom customer will churn using behavioral and billing data.

The system includes:
	•	Exploratory Data Analysis (EDA)
	•	Data preprocessing pipeline
	•	Machine learning model training
	•	REST API for predictions
	•	Interactive dashboard for analysis

The goal is to demonstrate production-style ML system design, combining data science, backend APIs, and analytics dashboards.

⸻

Dataset

The dataset used in this project is the Telco Customer Churn dataset.

Features include:
	•	Customer demographics
	•	Subscription services
	•	Contract information
	•	Billing details
	•	Payment methods

Target variable:

Churn
	•	Yes → customer left the service
	•	No → customer retained

Dataset size: ~7,000 customer records

⸻

Project Architecture

customer-churn-ml-pipeline
│
├── api/                # FastAPI service for predictions
│   └── main.py
│
├── dashboard/          # Streamlit dashboard
│   └── app.py
│
├── data/
│   ├── raw/            # original dataset
│   └── processed/      # cleaned dataset
│
├── models/             # trained ML models
│   └── churn_model.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   └── train_model.py
│
├── README.md
└── requirements.txt

⸻

Exploratory Data Analysis

EDA was performed to understand patterns related to churn.

Key insights discovered:
	•	Customers with month-to-month contracts churn more frequently
	•	Higher monthly charges correlate with churn
	•	Customers using electronic check payments show higher churn probability
	•	Longer tenure significantly reduces churn likelihood

Visualizations include:
	•	churn distribution
	•	tenure vs churn
	•	monthly charges analysis
	•	correlation heatmap

EDA notebook:

notebooks/01_eda.ipynb

⸻

Machine Learning Pipeline

The modeling pipeline includes:

Data Preprocessing
	•	missing value handling
	•	categorical encoding
	•	feature scaling
	•	feature selection

Model Training

Multiple models were evaluated:
	•	Logistic Regression
	•	Random Forest
	•	Gradient Boosting

Model Evaluation

Metrics used:
	•	Accuracy
	•	Precision
	•	Recall
	•	ROC-AUC

Best performing model:

Random Forest Classifier

Model saved as:
```
models/churn_model.joblib
```
Model training notebook:

notebooks/02_modeling.ipynb

⸻

Model Performance

Metric	                Score
Score           	    ~0.80
Precision       	    ~0.78
Recall          	    ~0.73
ROC-AUC         	    ~0.84

The model demonstrates strong ability to identify high-risk churn customers.

⸻

API Service

A FastAPI service exposes the trained model for real-time predictions.

Start API:
```
uvicorn api.main:app --reload
```

API Endpoint
```
POST /predict
```

Example Request
```
{
  "tenure": 12,
  "monthly_charges": 70.0,
  "contract": "Month-to-month",
  "payment_method": "Electronic check"
}
```

Example Response
```
{
  "churn_probability": 0.78,
  "prediction": "Churn"
}
```

⸻

Interactive Dashboard

A Streamlit dashboard allows users to explore churn predictions interactively.

Run dashboard:
```
streamlit run dashboard/app.py
```
Dashboard features:
	•	customer churn probability prediction
	•	feature input simulation
	•	visualization of churn patterns
	•	model output interpretation

⸻

Business Impact

This system helps organizations:
	•	identify high-risk churn customers early
	•	prioritize customer retention campaigns
	•	reduce revenue loss from subscription cancellations
	•	translate ML predictions into actionable business insights

⸻

Technologies Used

Languages
	•	Python

Machine Learning
	•	Scikit-learn
	•	Pandas
	•	NumPy

Visualization
	•	Matplotlib
	•	Seaborn

Backend
	•	FastAPI

Dashboard
	•	Streamlit

Model Serialization
	•	Joblib

⸻

How to Run the Project

Clone repository
```
git clone <repo-url>
cd customer-churn-ml-pipeline
```

Install dependencies
```
pip install -r requirements.txt
```

Run API
```
uvicorn api.main:app --reload
```

Run dashboard
```
streamlit run dashboard/app.py
```
⸻

Future Improvements

Potential enhancements include:
	•	automated retraining pipeline
	•	feature store integration
	•	real-time data ingestion
	•	model monitoring and drift detection
	•	deployment using Docker and cloud infrastructure

⸻

Author

Rushitaben Vachhani
MS Software Engineering – Northeastern University
Gold Medalist | Data Science & AI Enthusiast