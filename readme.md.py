# <div align="center">Customer Churn Prediction System</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

</div>

## What is Customer Churn?

Customer churn occurs when customers stop doing business with a company or service. In the telecommunications industry, customers can easily switch between service providers, resulting in an annual churn rate of 15-25% in this highly competitive market.

Retaining existing customers is significantly less expensive than acquiring new ones. By predicting which customers are likely to churn, companies can focus retention efforts on high-risk customers, improving profitability and market position.

This project builds a machine learning system to identify customers at risk of churning, enabling proactive retention strategies that could save millions in lost revenue.

## Objectives:

- Identify the percentage of churned customers and analyze churn patterns
- Determine key features responsible for customer churn through exploratory analysis
- Build and evaluate multiple machine learning models to accurately predict churn
- Deploy an interactive web application for real-time churn risk assessment

## Dataset:

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset includes 7,043 customer records with information about:
- **Customer demographics** – gender, senior citizen status, partners, and dependents
- **Account information** – tenure, contract type, payment method, paperless billing, monthly charges, and total charges
- **Services subscribed** – phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV and movies
- **Churn status** – whether the customer left within the last month

## Implementation:

**Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib

**Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting

## Key Findings from EDA:

### 1. Overall Churn Distribution:
![Churn Distribution](visualizations/churn_distribution.png)

> 26.5% of customers churned, indicating a significant retention challenge.

### 2. Contract Type Analysis:
![Churn by Contract](visualizations/churn_analysis_by_features.png)

> Month-to-month contracts show 42% churn rate compared to 14% for annual contracts – a 3x higher risk. This suggests contract flexibility increases churn likelihood.

### 3. Payment Method Impact:
> Electronic check users exhibit 45% churn rate, significantly higher than other payment methods. Customers using automatic payment methods (credit card, bank transfer) show lower churn rates.

### 4. Tech Support Services:
> Customers without tech support have 42% churn rate. This indicates that quality support services are critical for retention.

### 5. Customer Tenure:
> New customers (0-6 months) are most vulnerable to churn. Churn rate decreases significantly after the first year, suggesting the importance of early engagement.

### 6. Internet Service Type:
> Fiber optic customers churn at higher rates than DSL customers, potentially indicating service quality or pricing issues with fiber optic offerings.

### 7. Feature Correlation:
![Correlation Matrix](visualizations/correlation_matrix.png)

> Tenure shows strong negative correlation with churn (-0.35), while monthly charges show positive correlation (0.19). Total charges are negatively correlated, reflecting the relationship between tenure and cumulative spending.

## Feature Engineering:

To improve model performance, I engineered 7 new features:

1. **Customer Lifetime Value (CLV)** – Monthly charges multiplied by tenure
2. **Charges Ratio** – Detects price changes over customer lifetime
3. **Service Count** – Total number of services subscribed
4. **Tenure Groups** – Categorical bucketing of customer tenure (0-6 months, 6-12 months, 1-2 years, 2-4 years, 4+ years)
5. **Tech Support Flag** – Binary indicator for tech support subscription
6. **Security Services Flag** – Combined indicator for security-related services
7. **Charges Category** – Low/Medium/High monthly charges buckets

These engineered features increased the total feature set from 21 to 41, improving model predictive power.

## Machine Learning Model Evaluation:

### Model Comparison:
![Model Comparison](visualizations/model_comparison.png)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **80.1%** | **65.8%** | **51.9%** | **58.0%** | **84.5%** |
| Gradient Boosting | 79.6% | 64.1% | 52.9% | 58.0% | 83.8% |
| Random Forest | 79.3% | 63.9% | 51.1% | 56.8% | 83.0% |
| Decision Tree | 77.1% | 57.6% | 52.4% | 54.9% | 74.5% |

### Final Model: Logistic Regression

Logistic Regression was selected as the best model based on ROC-AUC score of 84.5%.

![Best Model Evaluation](visualizations/best_model_evaluation.png)

#### Confusion Matrix Results:
```
                Predicted No Churn    Predicted Churn
Actual No Churn        934                  101
Actual Churn           180                  194
```

> The model correctly identified 934 customers who stayed and 194 customers who churned. It missed 180 actual churners (false negatives) and incorrectly flagged 101 loyal customers (false positives).

#### Performance Metrics:
- **Accuracy**: 80.1% – Overall prediction correctness
- **Precision**: 65.8% – When predicting churn, the model is correct 66% of the time
- **Recall**: 51.9% – Model catches 52% of actual churners
- **ROC-AUC**: 84.5% – Excellent discrimination ability between churn and non-churn

## Business Impact:

### Test Set Results (1,409 customers):
- **Customers correctly identified as high-risk**: 194 out of 374 churners
- **Revenue saved**: $368,600 (194 customers × $1,900 net value per customer)
- **Retention costs**: $10,100 (101 false positives × $100 retention offer)
- **Net benefit**: $358,500

### Projected Annual Impact (7,043 customers):
- **Estimated annual savings**: $1,791,991
- **ROI**: 652% return on retention investment

By identifying high-risk customers early, the company can implement targeted retention strategies worth $100 per customer, saving an average customer lifetime value of $2,000.

## Streamlit Web Application:

I deployed an interactive Streamlit dashboard with four main pages:

1. **Home** – Project overview and key metrics
2. **Predict Churn** – Real-time prediction interface with customer input form
3. **Model Performance** – Visualizations and evaluation metrics
4. **About** – Technical details and methodology

The application provides:
- Instant churn probability calculations
- Risk level assessment (High/Medium/Low)
- Personalized retention recommendations
- Key risk factor identification

### Running the Application:
```bash
streamlit run streamlit_app.py
```

## Project Structure:

```
customer-churn-prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── data_preprocessed_full.csv
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── visualizations/
│   ├── churn_distribution.png
│   ├── churn_analysis_by_features.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── best_model_evaluation.png
│
├── phase1_data_loading.py
├── phase2_deep_eda.py
├── phase3_feature_engineering.py
├── phase4_model_training.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Installation:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis scripts in order (phase1 through phase4)
4. Launch the app: `streamlit run streamlit_app.py`

## Key Learnings:

- Month-to-month contracts are the strongest churn predictor with 3x higher risk
- Electronic check payment method correlates with significantly higher churn
- Tech support and security services reduce churn substantially
- Customer tenure is inversely related to churn – retention efforts should focus on new customers
- Feature engineering improved model performance more than algorithm complexity
- Logistic Regression performed best despite being simpler than ensemble methods

## Future Improvements:

- Implement SMOTE or other oversampling techniques to improve recall
- Add time-series forecasting for temporal churn patterns
- Incorporate customer service ticket data as additional features
- Build automated retraining pipeline for model updates
- Deploy to cloud platform (AWS/GCP) for production use
- Develop A/B testing framework to measure retention campaign effectiveness

## Contact:

**Nikhil Vanama**

Email: vanamanikhil0@gmail.com