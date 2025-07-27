import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import re

# Load dataset
df = pd.read_csv("laptop_price.csv", encoding="latin1")

# Drop unnecessary columns
df.drop(columns=['laptop_ID', 'Product'], inplace=True)

# Clean and convert 'Ram' and 'Weight'
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# Clean and convert 'Memory'
def process_memory_robust(mem):
    mem = mem.upper()
    mem = mem.replace("FLASH STORAGE", "").replace("HDD", "").replace("SSD", "").replace("HYBRID", "")
    mem = mem.replace("GB", "").replace("TB", "000")
    total = 0
    for p in re.split(r'\+|\s+', mem):
        try:
            if p.strip():
                total += int(float(p.strip()))
        except:
            continue
    return total

df['Memory'] = df['Memory'].apply(process_memory_robust)

# Convert 'ScreenResolution' to pixel count
def get_resolution(res):
    res = res.split()[-1]
    try:
        width, height = map(int, res.split('x'))
        return width * height
    except:
        return np.nan

df['Resolution'] = df['ScreenResolution'].apply(get_resolution)
df.drop(columns=['ScreenResolution'], inplace=True)

# Rename for clarity
df.rename(columns={'TypeName': 'Type', 'OpSys': 'Operating System'}, inplace=True)

# Drop missing values
df.dropna(inplace=True)
# Label encode categorical columns
label_encoders = {}
categorical_cols = ['Company', 'Type', 'Cpu', 'Gpu', 'Operating System']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Price_euros'])
y = df['Price_euros']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")
