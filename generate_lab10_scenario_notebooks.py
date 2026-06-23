import json
import os

def create_lab10_ipynb(filename, dataset_setup, model_setup):
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
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.metrics import accuracy_score, confusion_matrix\n"
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
        "source": ["### b. Train k-NN Classifier & Predict New Sample"]
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


# 10.1 Student
ds_10_1 = """# Mocking Student Performance Data
data = {
    'Study Hours': [5, 2, 6, 3, 8, 1],
    'Attendance %': [85, 60, 95, 55, 90, 45],
    'Internal Marks': [80, 50, 90, 45, 85, 40],
    'Status': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail']
}
df = pd.DataFrame(data)

# Encode Target
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])

X = df.drop('Status', axis=1)
y = df['Status']

# Standardize Features BEFORE split (Shortcut)
sc = StandardScaler()
X = sc.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_10_1 = """# k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict Test Data
y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\\nConfusion Matrix:\\n", confusion_matrix(y_test, y_pred))

# Classify a New Student: 4 hours study, 70% attendance, 65 internal marks
new_student = pd.DataFrame([[4, 70, 65]], columns=['Study Hours', 'Attendance %', 'Internal Marks'])
new_student = sc.transform(new_student)

pred_new = knn.predict(new_student)
print("\\nPredicted Class:", pred_new[0])"""
create_lab10_ipynb('N:/dmml_lab/sa SEE/10/10.1_Student_KNN.ipynb', ds_10_1, model_10_1)


# 10.2 House Price
ds_10_2 = """# Mocking House Price Data
data = {
    'Area (sq ft)': [1500, 800, 2500, 950, 3000, 1100],
    'Rooms': [3, 1, 4, 2, 5, 2],
    'Location Rating': [8, 5, 9, 6, 9, 4],
    'Price Category': ['Medium', 'Low', 'High', 'Low', 'High', 'Low']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['Price Category'] = le.fit_transform(df['Price Category'])

X = df.drop('Price Category', axis=1)
y = df['Price Category']

# Standardize Features BEFORE split (Shortcut)
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_10_2 = """# k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Classify a New House: 1800 sq ft, 3 rooms, rating 7
new_house = pd.DataFrame([[1800, 3, 7]], columns=['Area (sq ft)', 'Rooms', 'Location Rating'])
new_house = sc.transform(new_house)

pred_new = knn.predict(new_house)
print("\\nPredicted Class:", pred_new[0])"""
create_lab10_ipynb('N:/dmml_lab/sa SEE/10/10.2_HousePrice_KNN.ipynb', ds_10_2, model_10_2)


# 10.3 Heart Disease
ds_10_3 = """# Mocking Heart Disease Data
data = {
    'Age': [55, 30, 60, 45, 65, 35],
    'Cholesterol': [250, 180, 280, 210, 300, 190],
    'Blood Pressure': [140, 110, 150, 120, 160, 115],
    'Max Heart Rate': [120, 180, 110, 160, 105, 170],
    'Chest Pain Type': [3, 0, 3, 1, 3, 0],
    'Heart Disease': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Standardize Features BEFORE split (Shortcut)
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_10_3 = """# k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Classify a New Patient: Age 50, Chol 240, BP 130, HR 140, Chest Pain 2
new_patient = pd.DataFrame([[50, 240, 130, 140, 2]], columns=['Age', 'Cholesterol', 'Blood Pressure', 'Max Heart Rate', 'Chest Pain Type'])
new_patient = sc.transform(new_patient)

pred_new = knn.predict(new_patient)
print("\\nPredicted Class:", pred_new[0])"""
create_lab10_ipynb('N:/dmml_lab/sa SEE/10/10.3_HeartDisease_KNN.ipynb', ds_10_3, model_10_3)


# 10.4 Spam Email
ds_10_4 = """# Mocking Email Spam Data
data = {
    'Num Links': [5, 0, 8, 1, 10, 0],
    'Spam Keywords': [10, 1, 15, 2, 20, 0],
    'Message Length': [300, 1500, 200, 1200, 100, 2000],
    'Is Spam': ['Spam', 'Not Spam', 'Spam', 'Not Spam', 'Spam', 'Not Spam']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['Is Spam'] = le.fit_transform(df['Is Spam'])

X = df.drop('Is Spam', axis=1)
y = df['Is Spam']

# Standardize Features BEFORE split (Shortcut)
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_10_4 = """# k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Classify a New Email: 4 Links, 5 Keywords, 800 Length
new_email = pd.DataFrame([[4, 5, 800]], columns=['Num Links', 'Spam Keywords', 'Message Length'])
new_email = sc.transform(new_email)

pred_new = knn.predict(new_email)
print("\\nPredicted Class:", pred_new[0])"""
create_lab10_ipynb('N:/dmml_lab/sa SEE/10/10.4_EmailSpam_KNN.ipynb', ds_10_4, model_10_4)


# 10.5 IBM HR Attrition
ds_10_5 = """# Mocking HR Attrition Data
data = {
    'Age': [41, 35, 45, 25, 50, 28],
    'Monthly Income': [6000, 3000, 8000, 2500, 9000, 2800],
    'Job Role': ['Sales', 'Research', 'Manager', 'Sales', 'Manager', 'Research'],
    'Years at Company': [10, 2, 15, 1, 20, 2],
    'Job Satisfaction': [4, 2, 4, 1, 4, 2],
    'Attrition': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

le = LabelEncoder()
for col in ['Job Role', 'Attrition']:
    df[col] = le.fit_transform(df[col])

X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Standardize Features BEFORE split (Shortcut)
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df.head()"""
model_10_5 = """# k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Classify a New Employee: Age 30, Income 4000, Role (Sales=2), 3 Years, Satisfaction 3
new_emp = pd.DataFrame([[30, 4000, 2, 3, 3]], columns=['Age', 'Monthly Income', 'Job Role', 'Years at Company', 'Job Satisfaction'])
new_emp = sc.transform(new_emp)

pred_new = knn.predict(new_emp)
print("\\nPredicted Class:", pred_new[0])"""
create_lab10_ipynb('N:/dmml_lab/sa SEE/10/10.5_HR_Attrition_KNN.ipynb', ds_10_5, model_10_5)

print("Lab 10 Notebooks created successfully in sa SEE/10/ !")
