import json
import os

def create_lab4_ipynb(filename, dataset_setup, target_col, options):
    cells = []
    
    # Standard header cells
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Prepare Dataset (Mock Data)"]
    })
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in dataset_setup.split('\n')[:-1]] + [dataset_setup.split('\n')[-1]]
    })
        
    if 'convert_categorical' in options:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### Convert Categorical Variables"]
        })
        if 'label_cols' in options:
            cols = options['label_cols']
            lines = [
                "from sklearn.preprocessing import LabelEncoder\n",
                "le = LabelEncoder()\n"
            ]
            for col in cols:
                lines.append(f"df['{col}'] = le.fit_transform(df['{col}'])\n")
            lines.append("print(df.head())")
            
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": lines
            })
        
    if 'convert_spam' in options:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### Convert text labels (spam/ham) to numeric values"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"df['{target_col}'] = df['{target_col}'].map({{'ham': 0, 'spam': 1}})\n",
                "print(df.head())"
            ]
        })
        
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            f"x = df.drop('{target_col}', axis=1)\n",
            f"y = df['{target_col}']\n",
            "print('--- Original Class Distribution ---')\n",
            "print(y.value_counts())"
        ]
    })
    
    if 'random_over' in options:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### a. Random Oversampling"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from imblearn.over_sampling import RandomOverSampler\n",
                "ros = RandomOverSampler(random_state=42)\n",
                "x_ros, y_ros = ros.fit_resample(x, y)\n",
                "print('--- After Random Oversampling ---')\n",
                "print(y_ros.value_counts())"
            ]
        })
        
    if 'random_under' in options:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### b. Random Undersampling"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from imblearn.under_sampling import RandomUnderSampler\n",
                "rus = RandomUnderSampler(random_state=42)\n",
                "x_rus, y_rus = rus.fit_resample(x, y)\n",
                "print('--- After Random Undersampling ---')\n",
                "print(y_rus.value_counts())"
            ]
        })
        
    if 'smote' in options:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### c. SMOTE"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from imblearn.over_sampling import SMOTE\n",
                "# NOTE: SMOTE requires k_neighbors <= n_samples in minority class. We set k_neighbors=1 for mock data.\n",
                "smote = SMOTE(random_state=42, k_neighbors=1)\n",
                "x_smote, y_smote = smote.fit_resample(x, y)\n",
                "print('--- After SMOTE ---')\n",
                "print(y_smote.value_counts())"
            ]
        })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

# 4.1: Bank Fraud
ds_4_1 = """data = {
    'Transaction_Type': ['Online', 'In-Store', 'Online', 'In-Store', 'Online', 'Online'],
    'Location': ['NY', 'CA', 'NY', 'CA', 'NY', 'TX'],
    'Amount': [100, 50, 120, 40, 200, 300],
    'Class': [0, 0, 0, 0, 1, 1]  # 4 Majority, 2 Minority
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab4_ipynb('N:/dmml_lab/sa SEE/4/4.1_Bank_Fraud_Sampling.ipynb', ds_4_1, 'Class', {
    'convert_categorical': True, 'label_cols': ['Transaction_Type', 'Location'], 'random_over': True, 'random_under': True, 'smote': True
})

# 4.2: Telecom Churn
ds_4_2 = """data = {
    'Geography': ['France', 'Spain', 'France', 'Germany', 'France', 'Germany'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
    'CreditScore': [600, 650, 700, 500, 720, 450],
    'Exited': [0, 0, 0, 0, 1, 1]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab4_ipynb('N:/dmml_lab/sa SEE/4/4.2_Telecom_Churn_Sampling.ipynb', ds_4_2, 'Exited', {
    'convert_categorical': True, 'label_cols': ['Geography', 'Gender'], 'random_over': True, 'smote': True
})

# 4.3: Spam/Ham Emails
ds_4_3 = """data = {
    'Message_Length': [20, 30, 25, 40, 150, 200],
    'Link_Count': [0, 0, 0, 0, 2, 3],
    'label': ['ham', 'ham', 'ham', 'ham', 'spam', 'spam']
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab4_ipynb('N:/dmml_lab/sa SEE/4/4.3_Email_Spam_Sampling.ipynb', ds_4_3, 'label', {
    'convert_spam': True, 'random_under': True, 'smote': True
})

# 4.4: Loan Prediction
ds_4_4 = """data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male'],
    'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Education': ['Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'No', 'Yes', 'No', 'No', 'Yes'],
    'Property_Area': ['Urban', 'Rural', 'Urban', 'Semiurban', 'Rural', 'Urban'],
    'Income': [5000, 4000, 6000, 3000, 7000, 2000],
    'Loan_Status': [1, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab4_ipynb('N:/dmml_lab/sa SEE/4/4.4_Loan_Default_Sampling.ipynb', ds_4_4, 'Loan_Status', {
    'convert_categorical': True, 'label_cols': ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], 'random_over': True, 'random_under': True, 'smote': True
})

# 4.5: E-Commerce Fraud
ds_4_5 = """data = {
    'Device_Type': ['Mobile', 'Desktop', 'Mobile', 'Mobile', 'Desktop', 'Mobile'],
    'Payment_Method': ['Card', 'Card', 'PayPal', 'Card', 'Crypto', 'Crypto'],
    'Time_Spent': [10, 15, 12, 8, 2, 1],
    'isFraud': [0, 0, 0, 0, 1, 1]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab4_ipynb('N:/dmml_lab/sa SEE/4/4.5_Ecommerce_Fraud_Sampling.ipynb', ds_4_5, 'isFraud', {
    'convert_categorical': True, 'label_cols': ['Device_Type', 'Payment_Method'], 'random_over': True, 'random_under': True, 'smote': True
})

print("Lab 4 Notebooks recreated successfully offline!")
