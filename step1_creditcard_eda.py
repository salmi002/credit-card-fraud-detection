import pandas as pd

# 1. Load the dataset
file_path = 'creditcard.csv'  # Make sure this file is in your working directory
df = pd.read_csv(file_path)

# 2. Shape of the dataset
print(f"Shape of the dataset: {df.shape}\n")

# 3. Data types
print("Data Types:")
print(df.dtypes)
print("\n")

# 4. Check for missing values
print("Missing Values:")
print(df.isnull().sum())
print("\n")

# 5. Preview first 5 rows
print("First 5 Rows:")
print(df.head())
print("\n")

# 6. Check class distribution (imbalanced dataset?)
print("Class Distribution (Target Variable - 'Class'):")
print(df['Class'].value_counts())
print("\n")
print("Class Distribution in Percentage:")
print(df['Class'].value_counts(normalize=True) * 100)
