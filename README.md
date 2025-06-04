💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It handles class imbalance using SMOTE and provides a real-time prediction app built with Streamlit.

📁 Project Structure

credit-card-fraud-detection/
├── step1_eda.py # Exploratory Data Analysis
├── step2_preprocessing_smote.py # Preprocessing and SMOTE balancing
├── step3_model_training.py # Model training and evaluation
├── step4_prediction_app.py # Streamlit web app for predictions
├── best_model_fraud.pkl # Trained model
├── model_features.pkl # List of feature columns
├── requirements.txt # Python dependencies
├── README.md # Project documentation

📊 Dataset

- Name: Credit Card Fraud Detection Dataset
- Source: [Kaggle Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Rows: 284,807
- Features: 30 (V1 to V28, Amount, Time) + Class (Target)
- Class Imbalance: Only ~0.17% transactions are fraud

 ⚙️ Workflow

1. Preprocessing:
   - Feature selection
   - Normalization
   - Handled class imbalance with SMOTE

2. Modeling:
   - Trained Logistic Regression & Random Forest
   - Best model selected based on F1-score and ROC AUC

3. Evaluation:
   - Accuracy: 95%
   - F1-score: 0.95
   - ROC AUC Score: 0.9911

4. Deployment:
   - A Streamlit app was built for end users to predict fraud in real time
   

 🚀 Getting Started

1. Clone the repository
git clone https://github.com/salmi002/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run step4_app.py
