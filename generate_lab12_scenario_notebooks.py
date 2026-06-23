import json
import os

def create_ipynb(filename, cells_content):
    cells = []
    for cell in cells_content:
        cells.append({
            "cell_type": cell['type'],
            "metadata": {},
            "execution_count": None if cell['type'] == 'code' else None,
            "outputs": [],
            "source": [line + "\n" for line in cell['source'].split('\n')[:-1]] + [cell['source'].split('\n')[-1]]
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


# 12.1 Breast Cancer: Naive Bayes vs Random Forest
cells_12_1 = [
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"},
    {"type": "markdown", "source": "### a. Preprocessing and Normalization"},
    {"type": "code", "source": "data = load_breast_cancer()\nX = pd.DataFrame(data.data, columns=data.feature_names)\ny = data.target\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n\nscaler = StandardScaler()\nX_train_sc = scaler.fit_transform(X_train)\nX_test_sc = scaler.transform(X_test)"},
    {"type": "markdown", "source": "### b & c. Implement Naive Bayes and Random Forest"},
    {"type": "code", "source": "# Naive Bayes\nnb = GaussianNB()\nnb.fit(X_train_sc, y_train)\nnb_pred = nb.predict(X_test_sc)\n\n# Random Forest\nrf = RandomForestClassifier(random_state=42)\nrf.fit(X_train_sc, y_train)\nrf_pred = rf.predict(X_test_sc)"},
    {"type": "markdown", "source": "### d & e. Compare models and display table"},
    {"type": "code", "source": "nb_acc = accuracy_score(y_test, nb_pred)\nrf_acc = accuracy_score(y_test, rf_pred)\n\nprint(f\"Naive Bayes Accuracy: {nb_acc*100:.2f}%\")\nprint(f\"Random Forest Accuracy: {rf_acc*100:.2f}%\")\nprint(\"\\nRandom Forest generally performs better as an ensemble method.\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/12/12.1_BreastCancer_RF.ipynb', cells_12_1)


# 12.2 SMS Spam: Decision Tree vs AdaBoost
cells_12_2 = [
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.metrics import accuracy_score, confusion_matrix\nimport seaborn as sns\nimport matplotlib.pyplot as plt"},
    {"type": "markdown", "source": "### a. Text Preprocessing and TF-IDF"},
    {"type": "code", "source": "# Mocking SMS Spam Data\ndata = {\n    'Message': ['Win a free iPhone now', 'Hey, are we still meeting later?', 'URGENT: Your account is locked', 'Can you pick up milk?', 'Congratulations, you won 1000 dollars', 'See you tomorrow'],\n    'Label': ['Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham']\n}\ndf = pd.DataFrame(data)\n\ntfidf = TfidfVectorizer()\nX = tfidf.fit_transform(df['Message'])\ny = df['Label']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. Decision Tree and AdaBoost"},
    {"type": "code", "source": "# Decision Tree\ndt = DecisionTreeClassifier(random_state=42)\ndt.fit(X_train, y_train)\ndt_pred = dt.predict(X_test)\n\n# AdaBoost\nada = AdaBoostClassifier(n_estimators=50, random_state=42)\nada.fit(X_train, y_train)\nada_pred = ada.predict(X_test)"},
    {"type": "markdown", "source": "### d & e. Compare results and identify best model"},
    {"type": "code", "source": "print(f\"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred)*100:.2f}%\")\nprint(f\"AdaBoost Accuracy: {accuracy_score(y_test, ada_pred)*100:.2f}%\")\n\nfig, ax = plt.subplots(1, 2, figsize=(10, 4))\nsns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, cmap='Blues', ax=ax[0])\nax[0].set_title('Decision Tree CM')\nsns.heatmap(confusion_matrix(y_test, ada_pred), annot=True, cmap='Greens', ax=ax[1])\nax[1].set_title('AdaBoost CM')\nplt.show()\n\nprint(\"AdaBoost is typically the better model for text classification due to reducing bias.\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/12/12.2_SMSSpam_AdaBoost.ipynb', cells_12_2)


# 12.3 Telco Churn: KNN vs Gradient Boosting
cells_12_3 = [
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder, StandardScaler\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import accuracy_score, f1_score"},
    {"type": "markdown", "source": "### a. Handle missing and encode"},
    {"type": "code", "source": "data = {\n    'Tenure': [1, 45, 2, 60, np.nan, 5],\n    'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70, 99.65],\n    'Contract': ['Month', 'One year', 'Month', 'Two year', 'Month', 'Month'],\n    'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']\n}\ndf = pd.DataFrame(data)\n\ndf['Tenure'].fillna(df['Tenure'].mean(), inplace=True)\n\nle = LabelEncoder()\ndf['Contract'] = le.fit_transform(df['Contract'])\ndf['Churn'] = le.fit_transform(df['Churn'])\n\nX = df.drop('Churn', axis=1)\ny = df['Churn']\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. KNN vs Gradient Boosting"},
    {"type": "code", "source": "knn = KNeighborsClassifier(n_neighbors=3)\nknn.fit(X_train, y_train)\nknn_pred = knn.predict(X_test)\n\ngb = GradientBoostingClassifier(random_state=42)\ngb.fit(X_train, y_train)\ngb_pred = gb.predict(X_test)"},
    {"type": "markdown", "source": "### d & e. Compare models"},
    {"type": "code", "source": "print(f\"KNN -> Accuracy: {accuracy_score(y_test, knn_pred):.2f}, F1: {f1_score(y_test, knn_pred):.2f}\")\nprint(f\"Gradient Boosting -> Accuracy: {accuracy_score(y_test, gb_pred):.2f}, F1: {f1_score(y_test, gb_pred):.2f}\")\nprint(\"\\nGradient Boosting generally provides a strong improvement by correcting errors sequentially.\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/12/12.3_TelcoChurn_GB.ipynb', cells_12_3)


# 12.4 Student Performance: Decision Tree vs Random Forest
cells_12_4 = [
    {"type": "code", "source": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, confusion_matrix\nimport seaborn as sns\nimport matplotlib.pyplot as plt"},
    {"type": "markdown", "source": "### a. Preprocess"},
    {"type": "code", "source": "data = {\n    'Absences': [10, 2, 15, 0, 5, 20],\n    'StudyTime': [2, 10, 1, 15, 5, 0],\n    'Activities': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],\n    'Status': ['Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail']\n}\ndf = pd.DataFrame(data)\n\nle = LabelEncoder()\ndf['Activities'] = le.fit_transform(df['Activities'])\ndf['Status'] = le.fit_transform(df['Status'])\n\nX = df.drop('Status', axis=1)\ny = df['Status']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. DT vs RF"},
    {"type": "code", "source": "dt = DecisionTreeClassifier(random_state=42)\ndt.fit(X_train, y_train)\ndt_pred = dt.predict(X_test)\n\nrf = RandomForestClassifier(random_state=42)\nrf.fit(X_train, y_train)\nrf_pred = rf.predict(X_test)"},
    {"type": "markdown", "source": "### d & e. Compare"},
    {"type": "code", "source": "print(f\"Decision Tree Acc: {accuracy_score(y_test, dt_pred)}\")\nprint(f\"Random Forest Acc: {accuracy_score(y_test, rf_pred)}\")\n\nfig, ax = plt.subplots(1, 2, figsize=(10, 4))\nsns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, cmap='Blues', ax=ax[0])\nax[0].set_title('DT CM')\nsns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, cmap='Greens', ax=ax[1])\nax[1].set_title('RF CM')\nplt.show()\nprint(\"Random Forest is the better classifier as it reduces variance compared to a single Decision Tree.\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/12/12.4_StudentPerf_RF.ipynb', cells_12_4)


# 12.5 Credit Dataset: Logistic Regression vs AdaBoost
cells_12_5 = [
    {"type": "code", "source": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder, StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"},
    {"type": "markdown", "source": "### a. Encode and Normalize"},
    {"type": "code", "source": "data = {\n    'Income': [50000, 20000, 80000, 15000, 100000, 25000],\n    'CreditHistory': ['Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor'],\n    'LoanAmount': [10000, 20000, 15000, 5000, 50000, 30000],\n    'Risk': ['Good Credit', 'Bad Credit', 'Good Credit', 'Bad Credit', 'Good Credit', 'Bad Credit']\n}\ndf = pd.DataFrame(data)\n\nle = LabelEncoder()\ndf['CreditHistory'] = le.fit_transform(df['CreditHistory'])\ndf['Risk'] = le.fit_transform(df['Risk'])\n\nX = df.drop('Risk', axis=1)\ny = df['Risk']\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. LR vs AdaBoost"},
    {"type": "code", "source": "lr = LogisticRegression()\nlr.fit(X_train, y_train)\nlr_pred = lr.predict(X_test)\n\nada = AdaBoostClassifier(random_state=42)\nada.fit(X_train, y_train)\nada_pred = ada.predict(X_test)"},
    {"type": "markdown", "source": "### d & e. Compare"},
    {"type": "code", "source": "lr_acc = accuracy_score(y_test, lr_pred)\nada_acc = accuracy_score(y_test, ada_pred)\n\nprint(f\"Logistic Regression Accuracy: {lr_acc*100:.2f}%\")\nprint(f\"AdaBoost Accuracy: {ada_acc*100:.2f}%\")\nprint(\"\\nAdaBoost is typically better for complex, non-linear relationships compared to Logistic Regression.\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/12/12.5_Credit_AdaBoost.ipynb', cells_12_5)

print("Lab 12 Notebooks created successfully in sa SEE/12/ !")
