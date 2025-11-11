import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import joblib
import os

# paths
BASEDIR = "C:/DRL-Anomaly-Based-IDS/NSL-KDD"
RESULTS_DIR = "C:/DRL-Anomaly-Based-IDS/results/local"
os.makedirs(RESULTS_DIR, exist_ok=True)

# load training data
train_arff, _ = arff.loadarff(f"{BASEDIR}/KDDTrain+.arff")
train_data = pd.DataFrame(train_arff)

# decode bytes to str
for col in train_data.select_dtypes([object]).columns:
    train_data[col] = train_data[col].str.decode('utf-8')

# ---------------------------
# Numerical features
# ---------------------------
numerical_cols = [0,4,5,7,8,9,10,12,13,14,15,16,17,18,19,
                  22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
scaler = StandardScaler()
train_data.iloc[:, numerical_cols] = scaler.fit_transform(train_data.iloc[:, numerical_cols])

# ---------------------------
# Categorical features
# ---------------------------
categorical_cols = [1,2,3,6,11,20,21,41]  # protocol_type, service, flag, land, etc.
train_data = pd.get_dummies(train_data, columns=train_data.columns[categorical_cols])

# ---------------------------
# Save scaler & column list
# ---------------------------
joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))
joblib.dump(train_data.columns.tolist(), os.path.join(RESULTS_DIR, "train_columns.pkl"))

print("Done! Scaler and train_columns saved locally.")
