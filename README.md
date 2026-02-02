# customer-churn-ml-pipeline
End-to-End Customer Churn Prediction (ML Product)

📌 Customer Churn Prediction – End-to-End ML Pipeline
🔍 Problem Statement

Customer churn is a critical challenge for subscription-based businesses. Identifying customers at risk of churn enables companies to take proactive retention actions and reduce revenue loss.

This project builds an end-to-end machine learning pipeline to predict customer churn and identify key drivers influencing churn behavior.

📊 Dataset
- Source: Telco Customer Churn dataset
- Size: 7,043 customers, 33 features
- Target Variable:
    - Churn (1 = churned, 0 = retained)

Key Feature Categories
- Customer demographics
- Service subscriptions
- Contract type
- Billing and payment information

⚙️ Project Workflow
1. Data Cleaning & EDA
    - Handled missing and inconsistent values
    - Identified key churn drivers (contract type, tenure, monthly charges)

2. Feature Engineering
    - Removed data leakage features (e.g., churn reason, churn score)
    - Encoded categorical variables using One-Hot Encoding

3. Modeling
    - Logistic Regression (baseline)
    - Random Forest (non-linear model)

4. Evaluation
    - ROC-AUC
    - Precision / Recall
    - Confusion Matrix

5. Interpretability
    - Feature importance analysis to explain churn drivers

🤖 Models & Performance
Model	                ROC-AUC	    Key Notes
Logistic Regression	    ~0.85	    Strong baseline, interpretable
Random Forest	        ~0.83	    Captures non-linear relationships
> Evaluation focused on ROC-AUC and churn recall due to class imbalance.

🔑 Key Insights
-> Month-to-month contracts have significantly higher churn
-> Short tenure customers are at highest churn risk
-> Higher monthly charges correlate with increased churn
-> Model insights align strongly with exploratory data analysis

📈 Feature Importance (Random Forest)

Top predictors of churn include:
    - Contract type
    - Tenure months
    - Monthly charges
    - Internet service features
These insights can directly support targeted retention strategies.

🧰 Tech Stack
- Language: Python
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- ML Techniques:
    - One-Hot Encoding
    - Logistic Regression
    - Random Forest
    - ROC-AUC evaluation
- Tools: Git, VS Code, Jupyter Notebook

📁 Repository Structure
    customer-churn-ml-pipeline/
    │── data/
    │   ├── raw/
    │   └── processed/
    │── notebooks/
    │   ├── 01_eda.ipynb
    │   └── 02_modeling.ipynb
    │── src/
    │   ├── data_preprocessing.py
    │   └── train_model.py
    │── api/
    │   └── main.py
    │── dashboard/
    │   └── app.py
    │── requirements.txt
    │── README.md

▶️ How to Run
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook

🚀 Future Improvements
-> Threshold tuning to optimize churn recall
-> Cross-validation and hyperparameter tuning
-> Model deployment using FastAPI / Streamlit

👤 Author
~ Rushitaben Vachhani