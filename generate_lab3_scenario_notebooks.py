import json
import os

def create_lab3_ipynb(filename, dataset_setup, encoding_setup):
    cells = []
    
    # 1. Imports
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "from sklearn.preprocessing import LabelEncoder\n"
        ]
    })
    
    # 2. Dataset Setup
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### a. Prepare Dataset (Mock Data)"]
    })
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in dataset_setup.split('\n')[:-1]] + [dataset_setup.split('\n')[-1]]
    })
    
    # 3. Encoding
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### b. Apply Encoding"]
    })
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in encoding_setup.split('\n')[:-1]] + [encoding_setup.split('\n')[-1]]
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


# 3.1 Customer Purchase
ds_3_1 = """data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Payment Method': ['Credit Card', 'PayPal', 'Cash', 'Credit Card', 'PayPal'],
    'Age': [25, 30, 22, 35, 28]
}
df = pd.DataFrame(data)
print("--- Before Encoding ---")
display(df)"""

enc_3_1 = """# Label Encoding for Gender (only 2 categories)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-Hot Encoding for Payment Method using pd.get_dummies()
df = pd.get_dummies(df, columns=['Payment Method'])

print("\\n--- After One-Hot Encoding ---")
display(df)"""
create_lab3_ipynb('N:/dmml_lab/sa SEE/3/3.1_Customer_Purchase_Encoding.ipynb', ds_3_1, enc_3_1)


# 3.2 Student Performance
ds_3_2 = """data = {
    'Grade Category': ['Excellent', 'Average', 'Poor', 'Excellent', 'Average'],
    'Study Level': ['High School', 'College', 'College', 'Masters', 'High School'],
    'Score': [95, 65, 40, 90, 70]
}
df = pd.DataFrame(data)
print("--- Before Encoding ---")
display(df)"""

enc_3_2 = """# Label Encoding for Grade Category (Ordered: Poor, Average, Excellent)
le = LabelEncoder()
df['Grade Category'] = le.fit_transform(df['Grade Category'])

# One-Hot Encoding for Study Level using pd.get_dummies()
df = pd.get_dummies(df, columns=['Study Level'])

print("\\n--- After One-Hot Encoding ---")
display(df)"""
create_lab3_ipynb('N:/dmml_lab/sa SEE/3/3.2_Student_Performance_Encoding.ipynb', ds_3_2, enc_3_2)


# 3.3 Employee Dataset
ds_3_3 = """data = {
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
    'Job Role': ['Manager', 'Developer', 'Analyst', 'Developer', 'Manager'],
    'Salary': [8000, 6000, 7000, 6500, 8500]
}
df = pd.DataFrame(data)
print("--- Before Encoding ---")
display(df)"""

enc_3_3 = """# Label Encoding for Department
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])

# One-Hot Encoding for Job Role using pd.get_dummies()
df = pd.get_dummies(df, columns=['Job Role'])

print("\\n--- After One-Hot Encoding ---")
display(df)"""
create_lab3_ipynb('N:/dmml_lab/sa SEE/3/3.3_Employee_Encoding.ipynb', ds_3_3, enc_3_3)


# 3.4 Car Evaluation
ds_3_4 = """data = {
    'Buying Price': ['High', 'Low', 'Medium', 'High', 'Low'],
    'Safety Rating': ['Low', 'High', 'Medium', 'Medium', 'High'],
    'Doors': [4, 2, 4, 4, 2]
}
df = pd.DataFrame(data)
print("--- Before Encoding ---")
display(df)"""

enc_3_4 = """# Label Encoding for Buying Price (Ordered)
le = LabelEncoder()
df['Buying Price'] = le.fit_transform(df['Buying Price'])

# One-Hot Encoding for Safety Rating using pd.get_dummies()
df = pd.get_dummies(df, columns=['Safety Rating'])

print("\\n--- After One-Hot Encoding ---")
display(df)"""
create_lab3_ipynb('N:/dmml_lab/sa SEE/3/3.4_Car_Evaluation_Encoding.ipynb', ds_3_4, enc_3_4)


# 3.5 Online Shopping
ds_3_5 = """data = {
    'Browser Type': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Firefox'],
    'Visitor Type': ['Returning', 'New', 'Returning', 'Returning', 'New'],
    'Time Spent': [12.5, 5.0, 18.2, 3.5, 9.1]
}
df = pd.DataFrame(data)
print("--- Before Encoding ---")
display(df)"""

enc_3_5 = """# Label Encoding for Visitor Type
le = LabelEncoder()
df['Visitor Type'] = le.fit_transform(df['Visitor Type'])

# One-Hot Encoding for Browser Type using pd.get_dummies()
df = pd.get_dummies(df, columns=['Browser Type'])

print("\\n--- After One-Hot Encoding ---")
display(df)"""
create_lab3_ipynb('N:/dmml_lab/sa SEE/3/3.5_Online_Shopping_Encoding.ipynb', ds_3_5, enc_3_5)

print("Lab 3 Notebooks rewritten successfully!")
