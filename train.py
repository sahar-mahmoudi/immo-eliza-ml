import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def train():
    """
    Train an XGBoost regression model to predict real estate prices.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function performs the following steps:
    1. Load and preprocess the dataset.
    2. Train an XGBoost regression model using the best hyperparameters.
    3. Evaluate the model on the training and test sets.
    4. Save the trained model and artifacts.
    5. Visualize model performance and residuals.

    
    Examples
    --------
    train()  # Train the XGBoost regression model.

    """
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["nbr_frontages", 'nbr_bedrooms', "latitude", "longitude", "total_area_sqm",
                     'surface_land_sqm','terrace_sqm','garden_sqm']
    fl_features = ["fl_terrace", 'fl_garden', 'fl_swimming_pool']
    cat_features = ["province", 'heating_type', 'state_building',
                    "property_type", "epc", 'locality', 'subproperty_type','region']

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])


    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

  # Use the best parameters found during RandomizedSearchCV
    best_params = {
        'n_estimators': 130,
        'max_depth': 9,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 1.0,
        'gamma': 5,
        'reg_alpha': 0,
        'reg_lambda': 1.0,
    }

    # Train the final model using the best parameters
    final_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    # Evaluate the final model
    train_score = r2_score(y_train, final_model.predict(X_train))
    test_score = r2_score(y_test, final_model.predict(X_test))

    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")


    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": final_model,
    }
    joblib.dump(artifacts, "models/artifacts_xg.joblib", compress="xz")

    # Visualize predicted vs. actual prices on the test set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, final_model.predict(X_test), alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual Prices vs. Predicted Prices (Test Set)")
    plt.savefig("plots/actual_vs_predicted.png", dpi = 800)
    plt.show()

    # Visualize residuals on the test set
    residuals = y_test - final_model.predict(X_test)

    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(final_model.predict(X_test), residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot (Test Set)")
    plt.savefig("plots/residual_plot.png", dpi = 800)
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals (Test Set)")
    plt.savefig("plots/residual_distribution.png", dpi = 800)
    plt.show()


if __name__ == "__main__":
    train()
