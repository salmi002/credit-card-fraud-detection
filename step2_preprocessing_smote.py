import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv('creditcard.csv')

# Step 1: Scale 'Time' and 'Amount'
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])

# Drop original 'Time' and 'Amount'
df = df.drop(['Time', 'Amount'], axis=1)

# Rearranging columns: put scaled features at the start
scaled_features = ['scaled_amount', 'scaled_time']
other_features = [col for col in df.columns if col not in scaled_features + ['Class']]
df = df[scaled_features + other_features + ['Class']]

# Step 2: Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Apply SMOTE to balance the training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("After SMOTE oversampling:")
print(y_train_res.value_counts())

# Save preprocessed data for next steps
joblib.dump((X_train_res, y_train_res, X_test, y_test), 'preprocessed_data.pkl')
print("Preprocessed data saved as 'preprocessed_data.pkl'")
