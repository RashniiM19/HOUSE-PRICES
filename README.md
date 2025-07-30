
# 🏡 House Price Prediction using Linear Regression

This project demonstrates how to build a machine learning model to predict house sale prices using the Ames Housing dataset. We apply a simple Linear Regression model based on key features and visualize the prediction results.

---

## 📌 Objectives

- Load and explore the housing dataset.
- Train a linear regression model using:
  - Gross Living Area (`GrLivArea`)
  - Number of Bedrooms Above Ground (`BedroomAbvGr`)
  - Number of Full Bathrooms (`FullBath`)
- Evaluate the model using Mean Squared Error (MSE) and R² Score.
- Allow custom user input to predict sale prices.
- Visualize correlations, predictions, and inputs.

---

## 📁 Dataset

We use the `train.csv` file from the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

Ensure the dataset is stored in the following structure:

```
/project-directory
│
├── data/
│   └── train.csv
```

---

## 🚀 Getting Started on Google Colab

1. Open Google Colab: https://colab.research.google.com/
2. Upload your `train.csv` inside a folder named `data`.
3. Run the code blocks below.

---

## 🧪 Install Required Libraries

```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📊 Full Python Code

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('data/train.csv')  # Adjust path if needed

# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = df[features].fillna(0)
y = df['SalePrice'].fillna(0)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("📉 Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("✅ R² Score:", r2_score(y_test, y_pred))

# === USER INPUT SECTION ===
user_input = {
    'GrLivArea': 1800,
    'BedroomAbvGr': 3,
    'FullBath': 2
}

input_df = pd.DataFrame([user_input])
predicted_price = model.predict(input_df)[0]

print(f"\n🔎 Predicted Sale Price for Input: ${predicted_price:,.2f}")
```

---

## 📈 Visualizations

```python
# 1. Heatmap: Feature Correlations
plt.figure(figsize=(8,6))
sns.heatmap(df[features + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with SalePrice")
plt.show()

# 2. Scatter Plot: Living Area vs Sale Price with User Input
plt.figure(figsize=(10,6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, label='Training Data')
plt.scatter(user_input['GrLivArea'], predicted_price, color='red', s=100, label='User Prediction')
plt.title("Living Area vs Sale Price")
plt.xlabel("Gross Living Area")
plt.ylabel("Sale Price")
plt.legend()
plt.grid(True)
plt.show()

# 3. Bar Plot: Input Features
plt.figure(figsize=(6,4))
sns.barplot(x=list(user_input.keys()), y=list(user_input.values()))
plt.title("User Input Features")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# 4. Actual vs Predicted on Test Set
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Test Predictions")
plt.scatter(predicted_price, predicted_price, color='red', s=100, label="User Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📦 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🙋‍♀️ Author

Created by [Your Name]. Feel free to contribute, fork, or open issues.

---

## 📜 License

This project is licensed under the MIT License.
