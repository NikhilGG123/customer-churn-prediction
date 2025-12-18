# Phase 2: Deep EDA & Visualizations
# Customer Churn Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')

print("="*80)
print("CUSTOMER CHURN PREDICTION - PHASE 2: DEEP EDA & VISUALIZATIONS")
print("="*80)

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ============================================================================
# 1. FIX TOTALCHARGES DATA TYPE ISSUE
# ============================================================================
print("\n[1] Fixing TotalCharges Data Type")
print("-" * 80)

# Convert TotalCharges to numeric (errors become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for NaN values
nan_count = df['TotalCharges'].isna().sum()
print(f"Found {nan_count} rows with invalid TotalCharges")

# These are likely new customers with 0 tenure - let's verify
print("\nRows with missing TotalCharges:")
print(df[df['TotalCharges'].isna()][['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']])

# Fill NaN with 0 (these are new customers)
df['TotalCharges'].fillna(0, inplace=True)
print(f"\n✓ Filled {nan_count} missing values with 0")

# ============================================================================
# 2. CHURN RATE BY CATEGORICAL FEATURES
# ============================================================================
print("\n[2] Analyzing Churn Rates Across Categorical Features")
print("-" * 80)

# Select key categorical features
cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod']

# Calculate churn rates for each feature
churn_rates = {}
for feature in cat_features:
    churn_rate = df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    churn_rates[feature] = churn_rate
    print(f"\n{feature}:")
    print(churn_rate.sort_values(ascending=False))

# ============================================================================
# 3. VISUALIZE TOP CHURN FACTORS
# ============================================================================
print("\n[3] Creating Visualizations for Key Churn Factors")
print("-" * 80)

# Create figure with subplots for key features
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Churn Rate Analysis by Key Features', fontsize=16, fontweight='bold', y=0.995)

# Feature 1: Contract Type
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
contract_churn.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'], width=0.7)
axes[0, 0].set_title('Churn by Contract Type', fontweight='bold')
axes[0, 0].set_xlabel('Contract Type')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].legend(['No Churn', 'Churn'], loc='upper right')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# Feature 2: Internet Service
internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
internet_churn.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'], width=0.7)
axes[0, 1].set_title('Churn by Internet Service', fontweight='bold')
axes[0, 1].set_xlabel('Internet Service')
axes[0, 1].set_ylabel('Proportion')
axes[0, 1].legend(['No Churn', 'Churn'], loc='upper right')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# Feature 3: Payment Method
payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
payment_churn.plot(kind='bar', ax=axes[0, 2], color=['#2ecc71', '#e74c3c'], width=0.7)
axes[0, 2].set_title('Churn by Payment Method', fontweight='bold')
axes[0, 2].set_xlabel('Payment Method')
axes[0, 2].set_ylabel('Proportion')
axes[0, 2].legend(['No Churn', 'Churn'], loc='upper right')
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45, ha='right')

# Feature 4: Tech Support
tech_churn = df.groupby('TechSupport')['Churn'].value_counts(normalize=True).unstack()
tech_churn.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'], width=0.7)
axes[1, 0].set_title('Churn by Tech Support', fontweight='bold')
axes[1, 0].set_xlabel('Tech Support')
axes[1, 0].set_ylabel('Proportion')
axes[1, 0].legend(['No Churn', 'Churn'], loc='upper right')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# Feature 5: Senior Citizen
df['SeniorCitizen_Label'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
senior_churn = df.groupby('SeniorCitizen_Label')['Churn'].value_counts(normalize=True).unstack()
senior_churn.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'], width=0.7)
axes[1, 1].set_title('Churn by Senior Citizen Status', fontweight='bold')
axes[1, 1].set_xlabel('Senior Citizen')
axes[1, 1].set_ylabel('Proportion')
axes[1, 1].legend(['No Churn', 'Churn'], loc='upper right')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

# Feature 6: Partner
partner_churn = df.groupby('Partner')['Churn'].value_counts(normalize=True).unstack()
partner_churn.plot(kind='bar', ax=axes[1, 2], color=['#2ecc71', '#e74c3c'], width=0.7)
axes[1, 2].set_title('Churn by Partner Status', fontweight='bold')
axes[1, 2].set_xlabel('Has Partner')
axes[1, 2].set_ylabel('Proportion')
axes[1, 2].legend(['No Churn', 'Churn'], loc='upper right')
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)

# Feature 7: Tenure Distribution
axes[2, 0].hist([df[df['Churn']=='No']['tenure'], df[df['Churn']=='Yes']['tenure']],
                bins=20, label=['No Churn', 'Churn'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[2, 0].set_title('Churn by Tenure', fontweight='bold')
axes[2, 0].set_xlabel('Tenure (months)')
axes[2, 0].set_ylabel('Count')
axes[2, 0].legend()

# Feature 8: Monthly Charges Distribution
axes[2, 1].hist([df[df['Churn']=='No']['MonthlyCharges'], df[df['Churn']=='Yes']['MonthlyCharges']],
                bins=20, label=['No Churn', 'Churn'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[2, 1].set_title('Churn by Monthly Charges', fontweight='bold')
axes[2, 1].set_xlabel('Monthly Charges ($)')
axes[2, 1].set_ylabel('Count')
axes[2, 1].legend()

# Feature 9: Total Charges Distribution
axes[2, 2].hist([df[df['Churn']=='No']['TotalCharges'], df[df['Churn']=='Yes']['TotalCharges']],
                bins=20, label=['No Churn', 'Churn'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[2, 2].set_title('Churn by Total Charges', fontweight='bold')
axes[2, 2].set_xlabel('Total Charges ($)')
axes[2, 2].set_ylabel('Count')
axes[2, 2].legend()

plt.tight_layout()
plt.savefig('churn_analysis_by_features.png', dpi=300, bbox_inches='tight')
print("✓ Saved: churn_analysis_by_features.png")
plt.show()

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("\n[4] Correlation Analysis")
print("-" * 80)

# Create numeric version of dataframe for correlation
df_numeric = df.copy()

# Convert Yes/No to 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

# Convert Churn to numeric
df_numeric['Churn_Binary'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})

# Select numeric columns for correlation
numeric_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                   'PhoneService', 'PaperlessBilling', 'MonthlyCharges',
                   'TotalCharges', 'Churn_Binary']

# Calculate correlation
correlation = df_numeric[numeric_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numeric Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_matrix.png")
plt.show()

# Print top correlations with churn
churn_correlations = correlation['Churn_Binary'].sort_values(ascending=False)
print("\nTop Correlations with Churn:")
print(churn_correlations)

# ============================================================================
# 5. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 COMPLETE - KEY INSIGHTS:")
print("="*80)

# Calculate specific insights
month_to_month_churn = df[df['Contract']=='Month-to-month']['Churn'].value_counts(normalize=True)['Yes'] * 100
fiber_churn = df[df['InternetService']=='Fiber optic']['Churn'].value_counts(normalize=True)['Yes'] * 100
electronic_check_churn = df[df['PaymentMethod']=='Electronic check']['Churn'].value_counts(normalize=True)['Yes'] * 100
senior_churn = df[df['SeniorCitizen']==1]['Churn'].value_counts(normalize=True)['Yes'] * 100
no_tech_support_churn = df[df['TechSupport']=='No']['Churn'].value_counts(normalize=True)['Yes'] * 100

print(f"\n📊 HIGH-RISK FACTORS:")
print(f"  • Month-to-month contracts: {month_to_month_churn:.1f}% churn rate")
print(f"  • Fiber optic internet: {fiber_churn:.1f}% churn rate")
print(f"  • Electronic check payment: {electronic_check_churn:.1f}% churn rate")
print(f"  • Senior citizens: {senior_churn:.1f}% churn rate")
print(f"  • No tech support: {no_tech_support_churn:.1f}% churn rate")

print(f"\n🔢 NUMERIC INSIGHTS:")
avg_tenure_churned = df[df['Churn']=='Yes']['tenure'].mean()
avg_tenure_stayed = df[df['Churn']=='No']['tenure'].mean()
print(f"  • Average tenure (churned): {avg_tenure_churned:.1f} months")
print(f"  • Average tenure (stayed): {avg_tenure_stayed:.1f} months")

avg_monthly_churned = df[df['Churn']=='Yes']['MonthlyCharges'].mean()
avg_monthly_stayed = df[df['Churn']=='No']['MonthlyCharges'].mean()
print(f"  • Average monthly charges (churned): ${avg_monthly_churned:.2f}")
print(f"  • Average monthly charges (stayed): ${avg_monthly_stayed:.2f}")

print("\n✓ TotalCharges issue fixed (11 missing values filled)")
print("✓ Created 2 comprehensive visualizations")
print("✓ Identified key risk factors for churn")
print("\nNext Step: Phase 3 - Feature Engineering & Preprocessing")
print("="*80)

# Save cleaned data
df.to_csv('data_checkpoint_phase2.csv', index=False)
print("\n✓ Cleaned data saved as 'data_checkpoint_phase2.csv'")