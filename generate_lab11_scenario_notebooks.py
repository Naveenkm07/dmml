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


# 11.1 Iris - From Scratch (Version 2 style)
cells_11_1 = [
    {"type": "code", "source": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.metrics import confusion_matrix, accuracy_score\nimport seaborn as sns"},
    {"type": "markdown", "source": "### a. Normalize features and perform one-hot encode labels"},
    {"type": "code", "source": "iris = load_iris()\nX = iris.data\ny = iris.target.reshape(-1, 1)\n\nencoder = OneHotEncoder(sparse_output=False)\ny_encoded = encoder.fit_transform(y)\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Split 70:30\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. Implement feedforward NN with 1 hidden layer & backpropagation from scratch"},
    {"type": "code", "source": "# Weights\nW1 = np.random.randn(4, 5)\nW2 = np.random.randn(5, 3)\n\nlosses = []\nfor _ in range(1000):\n    # Forward\n    hidden = 1 / (1 + np.exp(-(X_train @ W1)))\n    output = 1 / (1 + np.exp(-(hidden @ W2)))\n    \n    # Loss (Cross Entropy)\n    loss = -np.mean(y_train * np.log(output + 1e-9))\n    losses.append(loss)\n    \n    # Backward (Gradients)\n    d_out = (output - y_train) * output * (1 - output)\n    d_hid = d_out @ W2.T * hidden * (1 - hidden)\n    \n    # Update\n    W2 -= hidden.T @ d_out * 0.1\n    W1 -= X_train.T @ d_hid * 0.1"},
    {"type": "markdown", "source": "### d. Report accuracy, plot loss, and confusion matrix"},
    {"type": "code", "source": "# Test Accuracy\ntest_hidden = 1 / (1 + np.exp(-(X_test @ W1)))\ntest_output = 1 / (1 + np.exp(-(test_hidden @ W2)))\n\npred = np.argmax(test_output, axis=1)\ntrue = np.argmax(y_test, axis=1)\nprint(f\"Test Accuracy: {accuracy_score(true, pred) * 100:.2f}%\")\n\n# Plot Loss\nplt.plot(losses)\nplt.title('Loss vs Epochs')\nplt.show()\n\n# Confusion Matrix\ncm = confusion_matrix(true, pred)\nsns.heatmap(cm, annot=True, cmap='Blues')\nplt.show()"}
]
create_ipynb('N:/dmml_lab/sa SEE/11/11.1_Iris_ANN_Scratch.ipynb', cells_11_1)


cells_11_2 = [
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\nfrom sklearn.neural_network import MLPClassifier"},
    {"type": "markdown", "source": "### a. Handle missing values and scale features"},
    {"type": "code", "source": "data = {\n    'Pregnancies': [6, 1, 8, 1, 0, 5],\n    'Glucose': [148, 85, 183, 89, np.nan, 116],\n    'BloodPressure': [72, 66, 64, np.nan, 40, 74],\n    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6],\n    'Outcome': [1, 0, 1, 0, 1, 0]\n}\ndf = pd.DataFrame(data)\n\n# Handle missing values with mean\ndf.fillna(df.mean(), inplace=True)\n\nX = df.drop('Outcome', axis=1)\ny = df['Outcome']\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. Implement ANN with two hidden layers, train with early stopping"},
    {"type": "code", "source": "# MLPClassifier automatically implements ANN with backpropagation\n# hidden_layer_sizes=(12, 8) creates two hidden layers\nmlp = MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=200, early_stopping=True, random_state=42)\nmlp.fit(X_train, y_train)"},
    {"type": "markdown", "source": "### d. Evaluate using accuracy, precision, recall, F1, ROC-AUC"},
    {"type": "code", "source": "y_pred = mlp.predict(X_test)\ny_prob = mlp.predict_proba(X_test)[:, 1]\n\nprint(f\"Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\nprint(f\"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}\")\nprint(f\"Recall: {recall_score(y_test, y_pred, zero_division=0):.2f}\")\nprint(f\"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.2f}\")\nprint(f\"ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/11/11.2_Diabetes_ANN_MLP.ipynb', cells_11_2)


# 11.3 Digits - MLPClassifier (Version 3 style, easy for mini-batch)
cells_11_3 = [
    {"type": "code", "source": "import matplotlib.pyplot as plt\nimport numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import MinMaxScaler, OneHotEncoder\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.metrics import accuracy_score"},
    {"type": "markdown", "source": "### a. Normalize pixel values and one-hot encode"},
    {"type": "code", "source": "digits = load_digits()\nX = digits.data\ny = digits.target\n\n# Normalize pixels (0-1)\nscaler = MinMaxScaler()\nX_norm = scaler.fit_transform(X)\n\n# MLPClassifier internally handles one-hot encoding for multi-class classification\n# but let's do train-test split\nX_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)"},
    {"type": "markdown", "source": "### b & c. Implement neural network with one hidden layer, mini-batch GD"},
    {"type": "code", "source": "# solver='sgd' and batch_size configures mini-batch gradient descent\nmlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd',\n                    batch_size=32, learning_rate_init=0.01, max_iter=300, random_state=42)\n\nmlp.fit(X_train, y_train)"},
    {"type": "markdown", "source": "### d. Report accuracy and show misclassified images"},
    {"type": "code", "source": "train_acc = accuracy_score(y_train, mlp.predict(X_train))\ntest_pred = mlp.predict(X_test)\ntest_acc = accuracy_score(y_test, test_pred)\n\nprint(f\"Training Accuracy: {train_acc * 100:.2f}%\")\nprint(f\"Test Accuracy: {test_acc * 100:.2f}%\")\n\n# Find misclassified\nmisclassified_idx = np.where(test_pred != y_test)[0]\n\n# Plot a few misclassified\nplt.figure(figsize=(10, 4))\nfor i, idx in enumerate(misclassified_idx[:4]):\n    plt.subplot(1, 4, i + 1)\n    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')\n    plt.title(f\"True: {y_test[idx]}\\nPred: {test_pred[idx]}\")\n    plt.axis('off')\nplt.tight_layout()\nplt.show()"}
]
create_ipynb('N:/dmml_lab/sa SEE/11/11.3_Digits_ANN_MLP.ipynb', cells_11_3)


# 11.4 Breast Cancer - MLPClassifier (Version 3 style)
cells_11_4 = [
    {"type": "code", "source": "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.metrics import confusion_matrix, recall_score\nimport seaborn as sns\nimport matplotlib.pyplot as plt"},
    {"type": "markdown", "source": "### a. Standardize features and encode labels (Malignant=1, Benign=0)"},
    {"type": "code", "source": "data = load_breast_cancer()\nX = data.data\n# Original dataset: 0=Malignant, 1=Benign. Invert to match requirement\ny = 1 - data.target \n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)"},
    {"type": "markdown", "source": "### b & c. Build ANN (two hidden layers), train with stratified split"},
    {"type": "code", "source": "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)\n\n# 2 Hidden Layers\nmlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)\nmlp.fit(X_train, y_train)\n\ny_pred = mlp.predict(X_test)\n\n# Sensitivity = Recall\nsensitivity = recall_score(y_test, y_pred)\nprint(f\"Sensitivity (Recall): {sensitivity:.2f}\")\n\n# Confusion Matrix\ncm = confusion_matrix(y_test, y_pred)\nsns.heatmap(cm, annot=True, cmap='Reds', fmt='d')\nplt.title('Confusion Matrix')\nplt.xlabel('Predicted')\nplt.ylabel('True')\nplt.show()"},
    {"type": "markdown", "source": "### d. Perform k-fold cross-validation"},
    {"type": "code", "source": "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\ncv_scores = cross_val_score(mlp, X_scaled, y, cv=skf, scoring='accuracy')\n\nprint(f\"K-Fold CV Accuracies: {cv_scores}\")\nprint(f\"Average Performance: {np.mean(cv_scores):.4f}\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/11/11.4_BreastCancer_ANN_MLP.ipynb', cells_11_4)


# 11.5 Housing Price - MLPRegressor & LinearRegression (Version 3 style)
cells_11_5 = [
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neural_network import MLPRegressor\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import root_mean_squared_error\nimport matplotlib.pyplot as plt"},
    {"type": "markdown", "source": "### a. Data cleaning, standardize features, split 80:20"},
    {"type": "code", "source": "# Mocking Housing Price Data\ndata = {\n    'Rooms': [3, 4, 2, 5, 3, 4],\n    'Area': [1500, 2000, 900, 2500, 1600, 2200],\n    'Age': [10, 5, 20, 2, 15, 8],\n    'Price': [300000, 450000, 150000, 600000, 320000, 500000]\n}\ndf = pd.DataFrame(data)\n\nX = df.drop('Price', axis=1)\ny = df['Price']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nscaler = StandardScaler()\nX_train_sc = scaler.fit_transform(X_train)\nX_test_sc = scaler.transform(X_test)"},
    {"type": "markdown", "source": "### b & c. Implement ANN regressor (1 hidden layer), compare with Linear Regression"},
    {"type": "code", "source": "# ANN Regressor\nann = MLPRegressor(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)\n# Using early stopping to get validation loss across epochs automatically\nann_val = MLPRegressor(hidden_layer_sizes=(10,), max_iter=2000, early_stopping=True, random_state=42)\n\nann.fit(X_train_sc, y_train)\nann_val.fit(X_train_sc, y_train)\n\nann_preds = ann.predict(X_test_sc)\nann_rmse = root_mean_squared_error(y_test, ann_preds)\n\n# Linear Regression\nlr = LinearRegression()\nlr.fit(X_train_sc, y_train)\nlr_preds = lr.predict(X_test_sc)\nlr_rmse = root_mean_squared_error(y_test, lr_preds)\n\nprint(f\"ANN Test RMSE: {ann_rmse:.2f}\")\nprint(f\"Linear Regression Test RMSE: {lr_rmse:.2f}\")\n\nif ann_rmse < lr_rmse:\n    print(\"ANN Model is better.\")\nelse:\n    print(\"Linear Regression is better.\")"},
    {"type": "markdown", "source": "### d. Training vs validation loss plot"},
    {"type": "code", "source": "plt.plot(ann_val.loss_curve_, label='Training Loss')\nplt.plot(ann_val.validation_scores_, label='Validation Score (Not Loss)', color='orange')\nplt.title('Loss Curve during Training')\nplt.xlabel('Epochs')\nplt.ylabel('Loss / Score')\nplt.legend()\nplt.show()\n\nprint(\"Conclusion: The loss curve provides insight into underfitting or overfitting depending on if training loss stabilizes while validation diverges.\")"}
]
create_ipynb('N:/dmml_lab/sa SEE/11/11.5_HousingPrice_ANN.ipynb', cells_11_5)

print("Lab 11 Notebooks created successfully in sa SEE/11/ !")
