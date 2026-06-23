import json
import os

def create_lab9_ipynb(filename, dataset_setup, model_setup):
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
            "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.svm import SVC\n",
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        ]
    })
    
    # 2. Dataset Setup
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### a. Prepare Dataset & Split"]
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
        "source": ["### b. Train SVM Classifier & Evaluate"]
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


# 9.1 Placement
ds_9_1 = """# Mocking Placement Data
data = {
    '10th_Marks': [85, 60, 95, 55, 88, 70],
    '12th_Marks': [80, 65, 90, 50, 85, 75],
    'College_CGPA': [8.5, 6.0, 9.2, 5.5, 8.8, 7.2],
    'Aptitude_Score': [90, 50, 95, 40, 85, 60],
    'Status': ['Placed', 'Not Placed', 'Placed', 'Not Placed', 'Placed', 'Not Placed']
}
df = pd.DataFrame(data)

# Encode Target
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])

X = df.drop('Status', axis=1)
y = df['Status']

# Standardize Features BEFORE split
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_9_1 = """# SVC Model
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict & Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc*100:.2f}%")"""
create_lab9_ipynb('N:/dmml_lab/sa SEE/9/9.1_Placement_SVM.ipynb', ds_9_1, model_9_1)


# 9.2 Iris
ds_9_2 = """# Mocking Iris Data
data = {
    'SepalLength': [5.1, 4.9, 7.0, 6.4, 6.3, 5.8],
    'SepalWidth': [3.5, 3.0, 3.2, 3.2, 3.3, 2.7],
    'PetalLength': [1.4, 1.4, 4.7, 4.5, 6.0, 5.1],
    'PetalWidth': [0.2, 0.2, 1.4, 1.5, 2.5, 1.9],
    'Species': ['Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica']
}
df = pd.DataFrame(data)

# Encode Target
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

X = df.drop('Species', axis=1)
y = df['Species']

# Standardize Features BEFORE split
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_9_2 = """# SVC Model
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict & Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM on Iris dataset: {acc*100:.2f}%")"""
create_lab9_ipynb('N:/dmml_lab/sa SEE/9/9.2_Iris_SVM.ipynb', ds_9_2, model_9_2)


# 9.3 Breast Cancer
ds_9_3 = """# Mocking Breast Cancer Data
data = {
    'Radius': [17.99, 20.57, 11.42, 14.45, 13.54, 13.08],
    'Texture': [10.38, 17.77, 20.38, 20.22, 14.36, 15.71],
    'Perimeter': [122.80, 132.90, 77.58, 94.25, 87.46, 85.63],
    'Area': [1001.0, 1326.0, 386.1, 642.7, 566.3, 520.0],
    'Diagnosis': ['Malignant', 'Malignant', 'Benign', 'Benign', 'Malignant', 'Benign']
}
df = pd.DataFrame(data)

# Encode Target
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Standardize Features BEFORE split
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_9_3 = """# SVC Model
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict & Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Classification Accuracy on Test Data: {acc*100:.2f}%")"""
create_lab9_ipynb('N:/dmml_lab/sa SEE/9/9.3_BreastCancer_SVM.ipynb', ds_9_3, model_9_3)


# 9.4 Diabetes
ds_9_4 = """# Mocking Diabetes Data
data = {
    'Glucose': [148, 85, 183, 89, 137, 100, 190, 80],
    'BloodPressure': [72, 66, 64, 66, 40, 70, 80, 60],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.0, 35.5, 22.0],
    'Age': [50, 31, 32, 21, 33, 25, 45, 22],
    'Diabetes': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Encode Target
le = LabelEncoder()
df['Diabetes'] = le.fit_transform(df['Diabetes'])

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# Standardize Features BEFORE split
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_9_4 = """# SVC Model
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict & Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc*100:.2f}%")"""
create_lab9_ipynb('N:/dmml_lab/sa SEE/9/9.4_Diabetes_SVM.ipynb', ds_9_4, model_9_4)


# 9.5 Financial Transactions
ds_9_5 = """# Mocking Financial Transactions Data
data = {
    'Amount': [50, 15000, 20, 12000, 10, 20000, 15, 18000],
    'Distance_from_Home': [5, 500, 2, 800, 1, 1000, 3, 1200],
    'Online_Transaction': [1, 1, 0, 1, 0, 1, 1, 1],
    'Is_Fraudulent': ['Legitimate', 'Fraudulent', 'Legitimate', 'Fraudulent', 'Legitimate', 'Fraudulent', 'Legitimate', 'Fraudulent']
}
df = pd.DataFrame(data)

# Encode Target
le = LabelEncoder()
df['Is_Fraudulent'] = le.fit_transform(df['Is_Fraudulent'])

X = df.drop('Is_Fraudulent', axis=1)
y = df['Is_Fraudulent']

# Standardize Features BEFORE split
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_9_5 = """# SVC Model
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict & Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Classification Accuracy: {acc*100:.2f}%")"""
create_lab9_ipynb('N:/dmml_lab/sa SEE/9/9.5_Financial_SVM.ipynb', ds_9_5, model_9_5)

print("Lab 9 Notebooks created successfully in sa SEE/9/ !")
