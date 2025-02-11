import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('C:/Users/nithi/Desktop/nithin.csv')
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicate_count = data.duplicated().sum()
print(f"Duplicate Rows Found: {duplicate_count}")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter plots for feature relationships
features = ["TV", "Radio", "Newspaper"]
plt.figure(figsize=(15, 4))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=data[feature], y=data["Sales"])
    plt.title(f"{feature} vs Sales")
plt.show()

# Create a new feature for total advertising budget
data["Total_Budget"] = data["TV"] + data["Radio"] + data["Newspaper"]
print(data.head())

# Split data into training and testing sets
X = data.drop(columns=['Sales'])
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data successfully split into training and testing sets!")

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Save the model as a pickle file
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("Model saved as 'model.pkl'")
