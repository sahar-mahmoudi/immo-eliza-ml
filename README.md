# 🏰✨ Immo Eliza ML: Predicting Real Estate Prices in Belgium

Welcome to the Real Estate Price Prediction project by Immo Eliza! 🏡 This machine learning endeavor utilizes the powerful XGBoost algorithm to predict real estate prices in Belgium. Follow the comprehensive guide below to explore the project and start predicting property prices with confidence!

## 🚀 Quick Start

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sahar-mahmoudi/immo-eliza-ml.git

2. **Set Up Your Virtual Environment:**

   ```bash
   python -m venv venv

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

4. **Run the Training Script:**

   ```bash
   python train.py
   

## 📊 Project Structure
The project is structured as follows:

**data:** Contains the cleaned dataset, "properties.csv."
**models:** The trained XGBoost model and related artifacts.
**plots:** Visualizations of model performance and residuals.
**train.py, predict.py:** The source code for data preprocessing, model training, and visualization.
**.gitignore:** Specifies files and folders to be ignored by version control.
**README.md:** The detailed guide you are currently reading.
**requirements.txt:** Lists project dependencies for easy setup.

## 🛠️ Data Preprocessing

**Numerical Features:** Handled missing values using SimpleImputer with the mean strategy.
**Categorical Features:** Applied OneHotEncoder for one-hot encoding.
**Model Training Data:** Split into training and testing sets with a stratified approach.

## ⚙️ Model Training

**XGBoost Model:** Trained with the best hyperparameters obtained through RandomizedSearchCV.
**Evaluation**: R² scores calculated for both training and testing sets.


## 📈 Visualizations

**Actual vs. Predicted Prices:** Scatter plot showcasing model predictions on the test set.
**Residual Plot:** Illustrates the distribution of residuals (actual - predicted).
**Residual Distribution:** Histogram depicting the distribution of residuals.

## 🔧 Model Artifacts

**features:** Information about numerical, boolean, and categorical features.
**imputer**: The SimpleImputer object used for handling missing values.
**enc**: The OneHotEncoder object for categorical feature encoding.
**model**: The trained XGBoost model.

## 🤝 Contributing
Contributions are encouraged! Feel free to open issues, propose enhancements, or submit pull requests.
