import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Generer syntetisk datasæt
def generate_apartment_data(n=500):
    np.random.seed(42)

    # Generer data
    square_meters = np.random.randint(
        30, 150, n
    )  # Lejlighedsstørrelse mellem 30 og 150 kvadratmeter
    floor = np.random.randint(0, 10, n)  # Etage (0 = stue, op til 9. sal)
    balcony = np.random.choice([0, 1], n)  # 0 = ingen altan, 1 = har altan
    rooms = np.random.randint(1, 6, n)  # Antal værelser fra 1 til 5
    build_year = np.random.randint(1900, 2023, n)  # Byggeår mellem 1900 og 2023
    renovation = np.random.choice([0, 1], n)  # 0 = ikke renoveret, 1 = renoveret

    # Skab en syntetisk pris baseret på faktorer
    price = (
        square_meters * 15000  # Hver kvadratmeter er 15000 DKK værd
        + floor * 5000  # Hver etage øger værdien med 5000 DKK
        + balcony * 50000  # Altan tilføjer 50000 DKK til prisen
        + rooms * 10000  # Hvert værelse tilføjer 10000 DKK
        + (2023 - build_year) * -1000  # Ældre bygninger reduceres i værdi
        + renovation * 80000  # Renovering tilføjer 80000 DKK
        + np.random.normal(0, 50000, n)  # Støj i priserne
    )

    # Opret DataFrame
    df = pd.DataFrame(
        {
            "kvadratmeter": square_meters,
            "etage": floor,
            "altan": balcony,
            "værelser": rooms,
            "byggelsesår": build_year,
            "renovering": renovation,
            "pris": price,
        }
    )

    return df


# Generer data
data = generate_apartment_data()

# Dataopdeling i features og målvariabel
X = data[["kvadratmeter", "etage", "altan", "værelser", "byggelsesår", "renovering"]]
y = data["pris"]

# Split data til træning og test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regression Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Forudsigelser med Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluering af Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"R^2 Score: {r2_rf}")

# Lineær Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Forudsigelser med Lineær Regression
y_pred_linear = linear_model.predict(X_test)

# Evaluering af Lineær Regression
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nLineær Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_linear}")
print(f"Mean Squared Error (MSE): {mse_linear}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear}")
print(f"R^2 Score: {r2_linear}")

# 1. Scatter Plot: Kvadratmeter vs. Pris med Lineær Regressionslinje
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data["kvadratmeter"],
    y=data["pris"],
    alpha=0.6,
    color="teal",
    label="Faktiske data",
)
sns.lineplot(
    x=X_test["kvadratmeter"], y=y_pred_linear, color="orange", label="Lineær Regression"
)
plt.title("Forhold mellem kvadratmeter og pris med Lineær Regression")
plt.xlabel("Kvadratmeter")
plt.ylabel("Pris (DKK)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Feature Importance Bar Plot for Random Forest
feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame(
    {"Feature": features, "Importance": feature_importances}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title(
    "Faktorer med størst betydning for lejlighedsprisforudsigelse (Random Forest)"
)
plt.xlabel("Vigtighed")
plt.ylabel("Faktorer")
plt.grid(True)
plt.show()

# 3. Sammenligning af Forudsigelser: Lineær Regression vs. Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(
    y_test,
    y_pred_linear,
    alpha=0.5,
    color="orange",
    label="Lineær Regression Forudsigelser",
)
plt.scatter(
    y_test, y_pred_rf, alpha=0.5, color="green", label="Random Forest Forudsigelser"
)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2, label="Ideel linje")
plt.title("Sammenligning af Faktiske og Forudsigte Priser")
plt.xlabel("Faktiske priser (DKK)")
plt.ylabel("Forudsigte priser (DKK)")
plt.legend()
plt.grid(True)
plt.show()
