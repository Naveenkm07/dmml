import json
import os

def create_lab8_ipynb(filename, dataset_setup, model_setup, imports_extra=""):
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
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.naive_bayes import GaussianNB\n",
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        ] + ([imports_extra] if imports_extra else [])
    })
    
    # 2. Dataset Setup
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### a. Prepare Dataset & Preprocess"]
    })
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in dataset_setup.split('\n')[:-1]] + [dataset_setup.split('\n')[-1]]
    })
    
    # 3. Model Training
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### b. Train Naïve Bayes & Evaluate Model"]
    })
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in model_setup.split('\n')[:-1]] + [model_setup.split('\n')[-1]]
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


# 8.1 Spam Email
ds_8_1 = """# Mocking Email Dataset
data = {
    'Word Frequency (Free)': [0.8, 0.1, 0.9, 0.0, 0.7, 0.2],
    'Presence of Links': [1, 0, 1, 0, 1, 0],
    'Email Length': [150, 450, 200, 600, 100, 500],
    'Target': ['Spam', 'Not Spam', 'Spam', 'Not Spam', 'Spam', 'Not Spam']
}
df = pd.DataFrame(data)

# Encode target
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])

X = df.drop('Target', axis=1)
y = df['Target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_8_1 = """# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict & Evaluate
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred, target_names=le.classes_))"""
create_lab8_ipynb('N:/dmml_lab/sa SEE/8/8.1_SpamEmail_NaiveBayes.ipynb', ds_8_1, model_8_1)

# 8.2 Medical History
ds_8_2 = """# Mocking Medical History
data = {
    'Fever': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Cough': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
    'Fatigue': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Headache': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
    'Disease': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Encode categorical variables
le = LabelEncoder()
df['Fever'] = le.fit_transform(df['Fever'])
df['Cough'] = le.fit_transform(df['Cough'])
df['Fatigue'] = le.fit_transform(df['Fatigue'])
df['Headache'] = le.fit_transform(df['Headache'])
df['Disease'] = le.fit_transform(df['Disease'])

X = df.drop('Disease', axis=1)
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_8_2 = """gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc*100:.2f}%")
print("\\nConfusion Matrix:\\n", confusion_matrix(y_test, y_pred))"""
create_lab8_ipynb('N:/dmml_lab/sa SEE/8/8.2_Medical_NaiveBayes.ipynb', ds_8_2, model_8_2)

# 8.3 Admission
ds_8_3 = """# Mocking Admission Dataset
data = {
    'GRE Score': [320, 290, 310, 330, 280, 315],
    'GPA': [3.8, 2.9, 3.5, 3.9, 2.5, 3.6],
    'Interview Score': [9, 5, 7, 9, 4, 8],
    'Admitted': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['Admitted'] = le.fit_transform(df['Admitted'])

X = df.drop('Admitted', axis=1)
y = df['Admitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_8_3 = """gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc*100:.2f}%")"""
create_lab8_ipynb('N:/dmml_lab/sa SEE/8/8.3_Admission_NaiveBayes.ipynb', ds_8_3, model_8_3)

# 8.4 Loan
ds_8_4 = """# Mocking Loan Dataset
data = {
    'Income': [60000, 30000, 80000, 20000, 90000, 35000],
    'Credit Score': [720, 600, 750, 580, 780, 620],
    'Employment Status': ['Permanent', 'Temporary', 'Permanent', 'Temporary', 'Permanent', 'Temporary'],
    'Loan History': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Good'],
    'Loan Approved': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['Employment Status'] = le.fit_transform(df['Employment Status'])
df['Loan History'] = le.fit_transform(df['Loan History'])
df['Loan Approved'] = le.fit_transform(df['Loan Approved'])

X = df.drop('Loan Approved', axis=1)
y = df['Loan Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_8_4 = """gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print("\\nInterpretation: The Naive Bayes model accurately predicts the loan approval outcome based on Income, Credit Score, Employment, and Loan History.")"""
create_lab8_ipynb('N:/dmml_lab/sa SEE/8/8.4_Loan_NaiveBayes.ipynb', ds_8_4, model_8_4)

# 8.5 Weather Play
ds_8_5 = """# Mocking Weather/Play Dataset
data = {
    'Weather': ['Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal', 'Normal'],
    'Play': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['Weather'] = le.fit_transform(df['Weather'])
df['Temperature'] = le.fit_transform(df['Temperature'])
df['Humidity'] = le.fit_transform(df['Humidity'])
df['Play'] = le.fit_transform(df['Play'])

X = df.drop('Play', axis=1)
y = df['Play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_8_5 = """gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")"""
create_lab8_ipynb('N:/dmml_lab/sa SEE/8/8.5_Weather_NaiveBayes.ipynb', ds_8_5, model_8_5)

print("Lab 8 Notebooks created successfully in sa SEE/8/ !")
