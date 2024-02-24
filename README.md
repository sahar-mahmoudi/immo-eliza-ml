# 🏰✨ Immo Eliza ML: Predicting Real Estate Prices in Belgium

Welcome to the Real Estate Price Prediction project by Immo Eliza! 🏡 In this machine learning project, we use the XGBoost algorithm to predict real estate prices in Belgium. Follow the guide below to explore the project and start predicting property prices with confidence!


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

4. **Predict for unseen data:**

   ```bash
   python predict.py -i data/properties.csv -o output/predictions.csv

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
**Boolean Features:** Encoded as dummy variables (1/0).              
**Categorical Features:** Applied OneHotEncoder for one-hot encoding.                      
**Feature List:**               
- **Numerical Features:**         
  - ['nbr_frontages', 'nbr_bedrooms', 'latitude', 'longitude', 'total_area_sqm', 'surface_land_sqm', 'terrace_sqm', 'garden_sqm']
- **Boolean Features:**         
  - ['fl_terrace', 'fl_garden', 'fl_swimming_pool']
- **Categorical Features:**                 
  - ['province', 'heating_type', 'state_building', 'property_type', 'epc', 'locality', 'subproperty_type', 'region']




## ⚙️ Model Training

**Model Training Data:** Split the model training data into training (80%) and testing (20%) sets through random sampling.      
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
