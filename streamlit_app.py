# Phase 5: Streamlit App - Customer Churn Predictor
# Customer Churn Prediction Project

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Title
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "🔮 Predict Churn", "📈 Model Performance", "ℹ️ About"])

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown("## Welcome to the Customer Churn Prediction System!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Overview")
        st.write("""
        This machine learning system predicts which customers are likely to churn (cancel their service).
        By identifying at-risk customers early, businesses can:
        - Implement targeted retention strategies
        - Save millions in lost revenue
        - Improve customer satisfaction
        """)

        st.markdown("### 🔑 Key Features")
        st.write("""
        - **Real-time Predictions**: Get instant churn risk assessments
        - **High Accuracy**: 80.1% prediction accuracy
        - **Explainable Results**: Understand why customers might churn
        - **Business Value**: Projected $1.8M annual savings
        """)

    with col2:
        st.markdown("### 📊 Model Performance")
        st.metric("Model", "Logistic Regression")
        st.metric("Accuracy", "80.1%")
        st.metric("ROC-AUC Score", "0.845")
        st.metric("Recall (Churn Detection)", "51.9%")

        st.markdown("### 💰 Business Impact")
        st.metric("Annual Revenue Protection", "$1.79M")
        st.metric("Customers Saved", "194 (Test Set)")
        st.metric("Net Benefit", "$358,500 (Test Set)")

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to **Predict Churn** to try the model!")

# ============================================================================
# PREDICT CHURN PAGE
# ============================================================================
elif page == "🔮 Predict Churn":
    st.markdown("## 🔮 Predict Customer Churn")
    st.write("Enter customer information below to predict churn probability.")

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        st.markdown("### 📅 Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)", "Credit card (automatic)"])

    with col2:
        st.markdown("### 📞 Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("### 💵 Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, 5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure), 100.0)

    # Predict button
    if st.button("🔮 Predict Churn Risk", key="predict_button"):
        # Create input dataframe
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_data])

        # Feature engineering (same as training)
        input_df['CLV'] = input_df['MonthlyCharges'] * input_df['tenure']
        input_df['ChargesRatio'] = input_df['TotalCharges'] / (input_df['MonthlyCharges'] * input_df['tenure'])
        input_df['ChargesRatio'].fillna(0, inplace=True)
        input_df['ChargesRatio'] = input_df['ChargesRatio'].replace([np.inf, -np.inf], 0)

        # Service count
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        input_df['ServiceCount'] = 0
        for col in service_cols:
            input_df['ServiceCount'] += (input_df[col] == 'Yes').astype(int) | \
                                        (input_df[col] == 'DSL').astype(int) | \
                                        (input_df[col] == 'Fiber optic').astype(int)

        input_df['HasTechSupport'] = (input_df['TechSupport'] == 'Yes').astype(int)
        input_df['HasSecurityServices'] = ((input_df['OnlineSecurity'] == 'Yes') |
                                           (input_df['OnlineBackup'] == 'Yes') |
                                           (input_df['DeviceProtection'] == 'Yes')).astype(int)


        # Tenure group
        def tenure_group(t):
            if t <= 6:
                return '0-6 months'
            elif t <= 12:
                return '6-12 months'
            elif t <= 24:
                return '1-2 years'
            elif t <= 48:
                return '2-4 years'
            else:
                return '4+ years'


        input_df['TenureGroup'] = input_df['tenure'].apply(tenure_group)


        # Charges category
        def charges_category(c):
            if c < 30:
                return 'Low'
            elif c < 70:
                return 'Medium'
            else:
                return 'High'


        input_df['ChargesCategory'] = input_df['MonthlyCharges'].apply(charges_category)

        # Encode categorical variables
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

        # One-hot encoding
        categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                            'StreamingMovies', 'Contract', 'PaymentMethod', 'TenureGroup', 'ChargesCategory']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Load feature names from training
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]

        # Align columns with training data
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]

        # Scale numeric features
        numeric_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'ChargesRatio', 'ServiceCount']
        cols_to_scale = [col for col in numeric_cols_to_scale if col in input_encoded.columns]
        input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])

        # Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]

        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        churn_prob = prediction_proba[1] * 100

        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1f}%")

        with col2:
            risk_level = "🔴 HIGH RISK" if churn_prob > 70 else "🟡 MEDIUM RISK" if churn_prob > 40 else "🟢 LOW RISK"
            st.metric("Risk Level", risk_level)

        with col3:
            prediction_text = "⚠️ LIKELY TO CHURN" if prediction == 1 else "✅ LIKELY TO STAY"
            st.metric("Prediction", prediction_text)

        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        if churn_prob > 70:
            st.error("""
            **High Risk Customer** - Immediate action required:
            - Offer personalized retention discount (10-20%)
            - Assign dedicated account manager
            - Reach out within 48 hours
            - Consider upgrading to annual contract with incentive
            """)
        elif churn_prob > 40:
            st.warning("""
            **Medium Risk Customer** - Proactive engagement needed:
            - Send satisfaction survey
            - Highlight unused service benefits
            - Offer loyalty rewards or service upgrade
            - Schedule check-in call
            """)
        else:
            st.success("""
            **Low Risk Customer** - Maintain relationship:
            - Continue standard engagement
            - Consider upsell opportunities
            - Reward loyalty with perks
            - Monitor satisfaction regularly
            """)

        # Risk factors
        st.markdown("### ⚠️ Key Risk Factors")
        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append("❌ Month-to-month contract (high risk)")
        if payment_method == "Electronic check":
            risk_factors.append("❌ Electronic check payment (high risk)")
        if tech_support == "No":
            risk_factors.append("❌ No tech support")
        if tenure < 12:
            risk_factors.append("❌ Short tenure (< 12 months)")
        if internet_service == "Fiber optic":
            risk_factors.append("⚠️ Fiber optic service (higher churn rate)")

        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("✅ No major risk factors identified")

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Analysis")

    # Load and display images
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Comparison")
        try:
            img1 = Image.open('model_comparison.png')
            st.image(img1, use_container_width=True)
        except:
            st.warning("Model comparison image not found")

    with col2:
        st.markdown("### Best Model Evaluation")
        try:
            img2 = Image.open('best_model_evaluation.png')
            st.image(img2, use_container_width=True)
        except:
            st.warning("Best model evaluation image not found")

    st.markdown("---")

    # Performance metrics
    st.markdown("### 🎯 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "80.1%", help="Overall correctness of predictions")
    col2.metric("Precision", "65.8%", help="When we predict churn, we're right 66% of the time")
    col3.metric("Recall", "51.9%", help="We catch 52% of actual churners")
    col4.metric("ROC-AUC", "0.845", help="Excellent discrimination ability")

    st.markdown("### 💰 Business Impact")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Set Net Benefit", "$358,500")
    col2.metric("Projected Annual Savings", "$1,791,991")
    col3.metric("Customers Saved", "194 (of 374 churners)")

    st.markdown("---")

    # Load model comparison data
    try:
        comparison_df = pd.read_csv('model_comparison.csv')
        st.markdown("### 📊 Detailed Model Comparison")
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score',
                                                                       'ROC-AUC']))
    except:
        st.warning("Model comparison data not found")

# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")

    st.markdown("### 📖 Project Description")
    st.write("""
    This Customer Churn Prediction System was built as an end-to-end machine learning project
    to demonstrate the complete ML workflow from data analysis to deployment.
    """)

    st.markdown("### 🛠️ Technical Stack")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Analysis & ML:**")
        st.write("""
        - Python 3.x
        - Pandas & NumPy
        - Scikit-learn
        - Matplotlib & Seaborn
        """)

    with col2:
        st.markdown("**Deployment:**")
        st.write("""
        - Streamlit
        - Joblib (model serialization)
        - Git & GitHub
        """)

    st.markdown("### 📊 Dataset")
    st.write("""
    - **Source:** Telco Customer Churn Dataset
    - **Size:** 7,043 customers
    - **Features:** 41 engineered features from 21 original variables
    - **Target:** Binary classification (Churn: Yes/No)
    """)

    st.markdown("### 🔬 Methodology")
    st.write("""
    1. **Exploratory Data Analysis:** Identified key churn factors
    2. **Feature Engineering:** Created 7 new predictive features
    3. **Model Training:** Tested 4 algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
    4. **Evaluation:** Selected best model based on ROC-AUC score
    5. **Deployment:** Built interactive Streamlit application
    """)

    st.markdown("### 🎓 Key Learnings")
    st.write("""
    - Month-to-month contracts have 3x higher churn than annual contracts
    - Electronic check payment correlates with higher churn
    - Tech support significantly reduces churn risk
    - New customers (< 6 months) are most vulnerable
    - Customer lifetime value is inversely correlated with churn
    """)

    st.markdown("### 📧 Contact")
    st.write("""
    **Created by:** [Your Name]

    **GitHub:** [Your GitHub Profile]

    **LinkedIn:** [Your LinkedIn Profile]
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Customer Churn Prediction System | Built with Streamlit and Scikit-learn</p>
    </div>
""", unsafe_allow_html=True)