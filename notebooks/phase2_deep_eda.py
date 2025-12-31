"""
Phase 2: Deep Exploratory Data Analysis
Analyze churn patterns across features and create comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print("PHASE 2: DEEP EDA & VISUALIZATIONS\n")

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges data type
print("Fixing TotalCharges:")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
nan_count = df['TotalCharges'].isna().sum()
print(f"Found {nan_count} invalid values (new customers with 0 tenure)")
df['TotalCharges'].fillna(0, inplace=True)
print("Filled with 0\n")

# Calculate churn rates for key features
print("Churn Rates by Feature:\n")
key_features = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'SeniorCitizen', 'Partner']

for feature in key_features:
    if feature == 'SeniorCitizen':
        df['SeniorCitizen_Label'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        churn_rate = df.groupby('SeniorCitizen_Label')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    else:
        churn_rate = df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)

    print(f"{feature}:")
    print(churn_rate.sort_values(ascending=False))
    print()

# Create comprehensive visualizations
print("Creating visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Churn Analysis by Key Features', fontsize=16, fontweight='bold', y=0.995)


# Helper function for categorical churn plots
def plot_churn(feature, ax, title):
    """Create stacked bar chart showing churn distribution"""
    churn_data = df.groupby(feature)['Churn'].value_counts(normalize=True).unstack()
    churn_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], width=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Proportion')
    ax.legend(['No Churn', 'Churn'], loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


# Generate plots
plot_churn('Contract', axes[0, 0], 'Churn by Contract Type')
plot_churn('InternetService', axes[0, 1], 'Churn by Internet Service')
plot_churn('PaymentMethod', axes[0, 2], 'Churn by Payment Method')
plot_churn('TechSupport', axes[1, 0], 'Churn by Tech Support')
plot_churn('SeniorCitizen_Label', axes[1, 1], 'Churn by Senior Citizen')
plot_churn('Partner', axes[1, 2], 'Churn by Partner Status')

# Numeric distributions
axes[2, 0].hist([df[df['Churn'] == 'No']['tenure'], df[df['Churn'] == 'Yes']['tenure']],
                bins=20, label=['No Churn', 'Churn'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[2, 0].set_title('Churn by Tenure', fontweight='bold')
axes[2, 0].set_xlabel('Tenure (months)')
axes[2, 0].legend()

axes[2, 1].hist([df[df['Churn'] == 'No']['MonthlyCharges'], df[df['Churn'] == 'Yes']['MonthlyCharges']],
                bins=20, label=['No Churn', 'Churn'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[2, 1].set_title('Churn by Monthly Charges', fontweight='bold')
axes[2, 1].set_xlabel('Monthly Charges ($)')
axes[2, 1].legend()

axes[2, 2].hist([df[df['Churn'] == 'No']['TotalCharges'], df[df['Churn'] == 'Yes']['TotalCharges']],
                bins=20, label=['No Churn', 'Churn'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[2, 2].set_title('Churn by Total Charges', fontweight='bold')
axes[2, 2].set_xlabel('Total Charges ($)')
axes[2, 2].legend()

plt.tight_layout()
plt.savefig('visualizations/churn_analysis_by_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/churn_analysis_by_features.png")

# Correlation analysis
print("\nCorrelation Analysis:")

# Convert categorical to numeric
df_numeric = df.copy()
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

df_numeric['Churn_Binary'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})

corr_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'PaperlessBilling', 'MonthlyCharges',
                 'TotalCharges', 'Churn_Binary']

correlation = df_numeric[corr_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/correlation_matrix.png")

print("\nTop correlations with churn:")
print(correlation['Churn_Binary'].sort_values(ascending=False))

# Calculate key insights
print("\nKey Insights:")
month_to_month = df[df['Contract'] == 'Month-to-month']['Churn'].value_counts(normalize=True)['Yes'] * 100
fiber = df[df['InternetService'] == 'Fiber optic']['Churn'].value_counts(normalize=True)['Yes'] * 100
echeck = df[df['PaymentMethod'] == 'Electronic check']['Churn'].value_counts(normalize=True)['Yes'] * 100

print(f"  Month-to-month contracts: {month_to_month:.1f}% churn")
print(f"  Fiber optic internet: {fiber:.1f}% churn")
print(f"  Electronic check payment: {echeck:.1f}% churn")

print(f"\n  Avg tenure (churned): {df[df['Churn'] == 'Yes']['tenure'].mean():.1f} months")
print(f"  Avg tenure (stayed): {df[df['Churn'] == 'No']['tenure'].mean():.1f} months")

print("\nPhase 2 Complete")
print("Next: Phase 3 - Feature Engineering")

df.to_csv('data/data_checkpoint_phase2.csv', index=False)
print("\nCheckpoint saved: data/data_checkpoint_phase2.csv")