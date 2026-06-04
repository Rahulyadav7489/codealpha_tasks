# Credit Scoring System

A machine learning-based credit scoring application that predicts credit risk for loan applicants using multiple classification models.

## Project Overview

This project builds and deploys a credit scoring system trained on the German Credit Dataset. It compares multiple machine learning algorithms to predict whether a loan applicant will default or not, ultimately generating a credit score between 300-900.

### Key Features

- **Multiple ML Models**: Implements Random Forest, XGBoost, and Logistic Regression classifiers
- **Model Evaluation**: Compares models using accuracy and AUC metrics
- **Interactive Web Interface**: Streamlit-based UI for real-time credit scoring predictions
- **Data Preprocessing**: Automated data cleaning and feature scaling
- **Credit Scoring**: Generates standardized credit scores (300-900 range) based on model predictions

## Project Structure

```
├── app.py                      # Streamlit web application
├── main.py                     # Model training and evaluation
├── split.py                    # Data loading and preprocessing
├── random_forest.py            # Random Forest model implementation
├── xg_boost.py                 # XGBoost model implementation
├── logistic_regression.py      # Logistic Regression model implementation
├── predict.py                  # Credit scoring prediction functions
├── requirements.txt            # Python dependencies
├── german_credit_data.csv      # Original dataset
├── cleaned_file.csv            # Preprocessed dataset
├── Untitled2.ipynb             # Jupyter notebook for exploration
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting classifier
- **streamlit**: Web application framework
- **joblib**: Model serialization
- **matplotlib**: Data visualization

## Usage

### Option 1: Run the Web Application

```bash
streamlit run app.py
```

This launches the interactive web interface where you can:
- Input applicant details (checking account status, loan duration, credit history, etc.)
- Get instant credit predictions
- View credit scores

### Option 2: Train and Evaluate Models

```bash
python main.py
```

This script:
- Loads and preprocesses the credit data
- Trains three different models
- Evaluates each model on test data
- Prints accuracy, AUC, and other metrics

### Option 3: Make Predictions Programmatically

```bash
python predict.py
```

This demonstrates credit scoring predictions for sample customers (low-risk and high-risk profiles).

## Dataset

The project uses the **German Credit Dataset** with the following characteristics:

- **Total Records**: 1,000 loan applications
- **Features**: 17 credit-related attributes
- **Target Variable**: Default (Good/Bad credit)
- **Data Split**: 80% training, 20% testing

### Key Features Used:

- Checking Account Status
- Loan Duration (months)
- Credit History
- Loan Amount
- Savings Account Status
- Employment Duration
- Installment Rate
- Personal Status & Sex
- Guarantors
- Property Type
- Age
- Other Installment Plans
- Housing Status
- Existing Credits
- Job Classification
- Telephone
- Foreign Worker Status

## Model Performance

The system trains and compares three classification models:

1. **Random Forest** (Default model used in app.py)
   - n_estimators: 300
   - max_depth: 6
   - min_samples_leaf: 5

2. **XGBoost**
   - Gradient boosting classifier with optimized parameters

3. **Logistic Regression**
   - Linear classification baseline

Each model is evaluated using:
- **Accuracy**: Percentage of correct predictions
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve

## Credit Score Calculation

Credit scores are generated on a standardized scale:

- **Range**: 300 - 900
- **Formula**: `300 + (prediction_probability × 600)`
- **Prediction**: Based on model's probability of "good" credit

## Key Files Description

### app.py
Interactive Streamlit application with form inputs for:
- Applicant financial details
- Loan parameters
- Personal information

Outputs credit prediction and score.

### main.py
Model training and evaluation pipeline that:
- Loads preprocessed data
- Trains all three models
- Generates performance metrics
- Compares model performance

### split.py
Data preprocessing module that:
- Loads the German Credit dataset
- Separates features and target
- Performs train-test split (80-20)
- Applies StandardScaler normalization

### predict.py
Prediction utilities with example usage:
- Low-risk customer profile
- High-risk customer profile
- Credit score calculation function

### Model Files (random_forest.py, xg_boost.py, logistic_regression.py)
Individual model implementations with specific hyperparameters.

## Workflow

1. **Data Preparation** (`split.py`)
   - Load CSV data
   - Split into training (80%) and testing (20%)
   - Scale features using StandardScaler

2. **Model Training** (model files)
   - Train each model on training data
   - Save models using joblib

3. **Evaluation** (`main.py`)
   - Evaluate on test data
   - Compare metrics across models

4. **Prediction** (`app.py` or `predict.py`)
   - Accept new applicant data
   - Scale using saved scaler
   - Generate predictions and credit scores

## Running the Application

### Start Web App:
```bash
streamlit run app.py
```
- Opens at `http://localhost:8501`
- Fill in applicant details
- Click predict to see credit score

### Train Models:
```bash
python main.py
```
- Displays model performance metrics
- Shows accuracy and AUC scores

### Test Predictions:
```bash
python predict.py
```
- Runs examples on predefined customer profiles
- Displays credit scores

## Model Serialization

Trained models are saved as pickle files:
- `credit_rf_model.pkl`: Random Forest model
- `scaler.pkl`: StandardScaler for feature normalization

These are loaded by `app.py` for making predictions without retraining.

## Future Enhancements

- [ ] Add feature importance visualization
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning
- [ ] Create model comparison dashboard
- [ ] Deploy to cloud platform (Streamlit Cloud, Heroku, etc.)
- [ ] Add explainability features (SHAP, LIME)
- [ ] Implement model monitoring and retraining pipeline

## Requirements

See `requirements.txt` for complete dependency list:
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- joblib
- matplotlib
- ipykernel

## Notes

- All numerical features are standardized before model input
- The Random Forest model is set as default in the web app for prediction
- Data should be preprocessed and placed in `cleaned_file.csv` before running
- Models assume consistent feature order and scaling

## License

This project uses the German Credit Dataset for educational purposes.

## Author

Created for credit risk assessment and machine learning demonstration.
