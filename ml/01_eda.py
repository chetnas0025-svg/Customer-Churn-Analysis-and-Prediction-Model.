import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("Starting Exploratory Data Analysis (EDA)...")
    
    # 1. Load data from SQLite
    db_path = 'bank_churn.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} not found. Please run Phase 1 first.")
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Ensure charts directory exists
    charts_dir = os.path.join('ml', 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Define primary color palette (Modern Fintech Dark Blue & Accent)
    colors = ['#1f77b4', '#ff7f0e'] # Retained (0), Churned (1)
    
    # ----------------------------------------------------
    # Chart 1: Churn Distribution (Pie Chart)
    # ----------------------------------------------------
    print("Generating Churn Distribution pie chart...")
    plt.figure(figsize=(6, 6))
    churn_counts = df['exited'].value_counts()
    labels = ['Retained (0)', 'Churned (1)']
    plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['#3498db', '#e74c3c'], explode=(0, 0.1), shadow=True)
    plt.title('Bank Customer Churn Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'churn_distribution.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Chart 2: Correlation Heatmap
    # ----------------------------------------------------
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    # Select numerical columns for correlation, encoding gender/geography simply
    corr_df = df.copy()
    corr_df['gender_encoded'] = corr_df['gender'].map({'Male': 0, 'Female': 1})
    # Remove non-numeric cols and customer_id which is arbitrary
    numeric_cols = ['credit_score', 'age', 'tenure', 'balance', 'num_products', 
                    'has_credit_card', 'is_active_member', 'estimated_salary', 
                    'gender_encoded', 'exited']
    corr_matrix = corr_df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Customer Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Chart 3: Age Distribution by Churn
    # ----------------------------------------------------
    print("Generating Age Distribution by Churn...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='age', hue='exited', fill=True, common_norm=False, 
                palette={0: '#3498db', 1: '#e74c3c'}, alpha=0.5, linewidth=2)
    plt.title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Age (Years)')
    plt.ylabel('Density')
    plt.legend(title='Customer Status', labels=['Churned (1)', 'Retained (0)'])
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'age_distribution.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Chart 4: Balance Distribution by Churn
    # ----------------------------------------------------
    print("Generating Balance Distribution by Churn...")
    plt.figure(figsize=(10, 6))
    # Plotting balance, separating out zero balance if desired, but KDE works fine
    sns.kdeplot(data=df, x='balance', hue='exited', fill=True, common_norm=False, 
                palette={0: '#3498db', 1: '#e74c3c'}, alpha=0.5, linewidth=2)
    plt.title('Balance Distribution by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Account Balance ($)')
    plt.ylabel('Density')
    plt.legend(title='Customer Status', labels=['Churned (1)', 'Retained (0)'])
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'balance_distribution.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Chart 5: Geography Churn Bar Chart
    # ----------------------------------------------------
    print("Generating Geography Churn Bar Chart...")
    plt.figure(figsize=(10, 6))
    # Calculate churn rate by geography
    geo_stats = df.groupby('geography')['exited'].agg(['count', 'sum']).reset_index()
    geo_stats['churn_rate'] = (geo_stats['sum'] / geo_stats['count']) * 100
    
    ax = sns.barplot(data=geo_stats, x='geography', y='churn_rate', palette='viridis')
    plt.title('Churn Rate (%) by Geography', fontsize=14, fontweight='bold')
    plt.xlabel('Geography')
    plt.ylabel('Churn Rate (%)')
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() - 3),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', 
                    fontweight='bold', color='white' if p.get_height() > 15 else 'black')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'geography_churn.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Chart 6: Credit Score vs Churn Boxplot
    # ----------------------------------------------------
    print("Generating Credit Score vs Churn Boxplot...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='exited', y='credit_score', palette={0: '#3498db', 1: '#e74c3c'})
    plt.title('Credit Score Distribution by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status (0 = Retained, 1 = Churned)')
    plt.ylabel('Credit Score')
    plt.xticks([0, 1], ['Retained', 'Churned'])
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'credit_score_churn.png'), dpi=300)
    plt.close()
    
    print("EDA Visualizations saved successfully to /ml/charts/")

if __name__ == '__main__':
    main()
