# Task 1: Predict House Prices using Linear Regression

# Step 1: Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv("train.csv")  # Make sure 'train.csv' is in the same folder

# Step 3: Select relevant features and target
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']]

# Step 4: Drop missing values if any
df.dropna(inplace=True)

# Step 5: Create a new column for Total Bathrooms
df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

# Step 6: Define input features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']]
y = df['SalePrice']

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Step 11: Show coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nModel Coefficients:")
print(coefficients)

# Step 12: Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")
plt.grid(True)
plt.tight_layout()
plt.show()
