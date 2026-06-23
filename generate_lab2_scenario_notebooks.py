import json
import os

def create_lab2_ipynb(filename, dataset_setup, options):
    cells = []
    
    # 1. Imports
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
            "from sklearn.model_selection import train_test_split\n"
        ]
    })
    
    # 2. Mock Dataset
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
        
    if options['split']:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"### Split Dataset ({options['split_ratio']})"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"train_df, test_df = train_test_split(df, test_size={options['test_size']}, random_state=42)\n",
                "print('Train shape:', train_df.shape)\n",
                "print('Test shape:', test_df.shape)"
            ]
        })
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### a. Normalize the required columns"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "minmax_scaler = MinMaxScaler()\n",
                f"train_df[{options['norm_cols']}] = minmax_scaler.fit_transform(train_df[{options['norm_cols']}])\n",
                f"test_df[{options['norm_cols']}] = minmax_scaler.transform(test_df[{options['norm_cols']}])\n",
                "print('--- After Normalization ---')\n",
                "print(train_df.head())"
            ]
        })
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### b. Standardize the required columns"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "std_scaler = StandardScaler()\n",
                f"train_df[{options['std_cols']}] = std_scaler.fit_transform(train_df[{options['std_cols']}])\n",
                f"test_df[{options['std_cols']}] = std_scaler.transform(test_df[{options['std_cols']}])\n",
                "print('--- After Standardization ---')\n",
                "print(train_df.head())"
            ]
        })
        
    else:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### a. Normalize the required columns"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "minmax_scaler = MinMaxScaler()\n",
                f"df[{options['norm_cols']}] = minmax_scaler.fit_transform(df[{options['norm_cols']}])\n",
                "print('--- After Normalization ---')\n",
                "print(df.head())"
            ]
        })
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### b. Standardize the required columns"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "std_scaler = StandardScaler()\n",
                f"df[{options['std_cols']}] = std_scaler.fit_transform(df[{options['std_cols']}])\n",
                "print('--- After Standardization ---')\n",
                "print(df.head())"
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

# 2.1: Student Marks. No split.
ds_2_1 = """data = {
    'Mathematics': [85, 90, 78, 92, 88],
    'Physics': [80, 85, 75, 89, 82],
    'Chemistry': [88, 92, 80, 95, 90]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab2_ipynb('N:/dmml_lab/sa SEE/2/2.1_Student_Marks_Preprocessing.ipynb', ds_2_1, {
    'split': False,
    'norm_cols': "['Mathematics', 'Physics']",
    'std_cols': "['Chemistry']"
})

# 2.2: HR Dataset. Split 80:20
ds_2_2 = """data = {
    'salary': [50000, 60000, 45000, 70000, 55000, 80000, 52000, 65000, 48000, 72000],
    'years_of_experience': [2, 5, 1, 8, 3, 10, 2, 6, 1, 9]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab2_ipynb('N:/dmml_lab/sa SEE/2/2.2_HR_Preprocessing.ipynb', ds_2_2, {
    'split': True,
    'test_size': 0.20,
    'split_ratio': '80:20',
    'norm_cols': "['salary']",
    'std_cols': "['years_of_experience']"
})

# 2.3: Real Estate. Split 70:30
ds_2_3 = """data = {
    'area': [1200, 1500, 1000, 2000, 1300, 1800, 1100, 1600, 900, 2200],
    'bedrooms': [2, 3, 2, 4, 3, 4, 2, 3, 1, 5],
    'price': [250000, 300000, 200000, 450000, 280000, 400000, 230000, 320000, 180000, 500000]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab2_ipynb('N:/dmml_lab/sa SEE/2/2.3_Real_Estate_Preprocessing.ipynb', ds_2_3, {
    'split': True,
    'test_size': 0.30,
    'split_ratio': '70:30',
    'norm_cols': "['area']",
    'std_cols': "['bedrooms', 'price']"
})

# 2.4: Rainfall. No split.
ds_2_4 = """data = {
    'rainfall': [120, 150, 80, 200, 100],
    'temperature': [25, 28, 22, 30, 24],
    'Humidity': [70, 80, 60, 90, 65],
    'soil_moisture': [30, 40, 20, 50, 25]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab2_ipynb('N:/dmml_lab/sa SEE/2/2.4_Rainfall_Preprocessing.ipynb', ds_2_4, {
    'split': False,
    'norm_cols': "['rainfall', 'temperature']",
    'std_cols': "['Humidity', 'soil_moisture']"
})

# 2.5: Bank Dataset. Split 75:25
ds_2_5 = """data = {
    'income': [60000, 40000, 80000, 30000, 90000, 50000, 70000, 45000, 85000, 35000],
    'loan_amount': [15000, 5000, 25000, 2000, 30000, 10000, 20000, 8000, 28000, 3000],
    'credit_score': [700, 600, 750, 550, 800, 650, 720, 620, 780, 580]
}
df = pd.DataFrame(data)
print(df.head())"""
create_lab2_ipynb('N:/dmml_lab/sa SEE/2/2.5_Bank_Preprocessing.ipynb', ds_2_5, {
    'split': True,
    'test_size': 0.25,
    'split_ratio': '75:25',
    'norm_cols': "['income']",
    'std_cols': "['loan_amount', 'credit_score']"
})

print("Lab 2 Notebooks recreated successfully offline!")
