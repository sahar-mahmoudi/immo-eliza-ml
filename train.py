import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ridge_regression, Lasso
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns



 


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/filtered_data.csv")

    # Define features to use
    num_features = ["nbr_frontages",'nbr_bedrooms',"latitude","total_area_sqm"]
    fl_features = ["fl_terrace", 'fl_garden', 'fl_swimming_pool','fl_floodzone']
    cat_features = ["equipped_kitchen", "province",'heating_type' ,'state_building',"property_type", "epc"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Create bins for the 'price' variable
    data['price_bins'] = pd.cut(data['price'], bins=5, labels=False)  # Adjust the number of bins as needed

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=505, stratify=data[['property_type', 'price_bins']]
)

    # Drop the temporary 'price_bins' column
    data.drop('price_bins', axis=1, inplace=True)

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])
    
    
    # Standardize numerical features using StandardScaler
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder(drop='first')
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

    #print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model
    #model = Lasso(alpha=1.0)  # You can adjust the alpha (regularization strength) as needed
    

   # Define the parameter distributions to sample from
    param_dist = {
        'n_estimators': [int(x) for x in range(50, 200, 10)],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Train the model using RandomizedSearchCV
    model = RandomForestRegressor(random_state=505)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=3,       # Number of cross-validation folds
        scoring='r2',
        random_state=505,
        n_jobs=-1    # Use all available CPU cores
    )

    random_search.fit(X_train, y_train)

    # Evaluate the best model
    #train_score = r2_score(y_train, random_search.predict(X_train))
    #test_score = r2_score(y_test, random_search.predict(X_test))
    
    # Print the best parameters and corresponding R2 score
    print("Best parameters found: ", random_search.best_params_)
    print("Best R2 score on cross-validation data: {:.4f}".format(random_search.best_score_))


    best_model = random_search.best_estimator_

    ## After fitting the best model
    train_score = r2_score(y_train, best_model.predict(X_train))
    test_score = r2_score(y_test, best_model.predict(X_test))

    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

   

    # Visualize predicted vs. actual prices on the test set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual Prices vs. Predicted Prices (Test Set)")
    plt.show()
    
    # Visualize residuals on the test set
    residuals = y_test - best_model.predict(X_test)

    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(best_model.predict(X_test), residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot (Test Set)")
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals (Test Set)")
    plt.show()
    
    # Plot feature importances
    features = X_train.columns
    importances = best_model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), features[indices], rotation=45, ha="right")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.show()

    
    

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "scaler": scaler,
        "enc": enc,
        "model": best_model,
    }
    joblib.dump(model, "models/artifacts.joblib", compress="xz")
    
    
if __name__ == "__main__":
    train()


# Best parameters found:  {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
# Best R2 score on cross-validation data: 0.5905
# R2 score on test data: 0.5897


# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# Best parameters found:  {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None}
# Best R2 score on cross-validation data: 0.6304
# R2 score on test data: 0.6935



# Best parameters found:  {'n_estimators': 120, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None, 'bootstrap': True}
# Best R2 score on cross-validation data: 0.6693
# Train R² score: 0.9529238923217082
# Test R² score: 0.6838216853070024
