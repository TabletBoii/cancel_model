import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def convert_range_to_numeric(x):
    if isinstance(x, str) and '-' in x:
        try:
            parts = x.split('-')
            numbers = [float(part.strip()) for part in parts]
            return sum(numbers) / len(numbers)
        except Exception as e:
            return np.nan

    return x


df_train = pd.read_excel("data_train.xlsx", sheet_name="TrainingData")
numeric_features = [
    "Ваш возраст",
    "Стаж курения",
    "Сколько пачек/штук употребляете за сутки? ",
    "Сколько раз в году болеете ОРВИ?"
]
for col in numeric_features:
    df_train[col] = df_train[col].replace("60 и выше", 60)
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

# Target
df_train["Вы болели раком?"] = df_train["Вы болели раком?"].replace({"Да": 1, "Нет": 0}).fillna(0)

# Define X and y
X_train_full = df_train.drop(columns=["Вы болели раком?"])
y_train_full = df_train["Вы болели раком?"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

categorical_features = [
    "Ваш пол",
    "Онкоанамнез у родителей и близких родственников? ",
    "Употребляете ли Вы сигареты? "
]

# Numeric pipeline: impute with mean, then scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # fills numeric NaNs with mean
    ("scaler", StandardScaler()),
])

# Categorical pipeline: impute with most frequent, then one-hot encode
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fills cat NaNs with mode
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Put them all in a ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# =========================================
# 3) Build the IMB Pipeline: preproc -> SMOTE -> Classifier
# =========================================
model = Pipeline([
    ("preproc", preprocessor),
    ("smote", SMOTE(random_state=42, k_neighbors=2)),  # reduce k_neighbors
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
])

# =========================================
# 4) Fit the pipeline on training data
#    (SMOTE will run AFTER imputation & encoding)
# =========================================
model.fit(X_train, y_train)

# =========================================
# 5) Evaluate on test data
#    (the pipeline will do the same transformations except SMOTE)
# =========================================
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred_test))
print("=== ROC-AUC ===")
print(roc_auc_score(y_test, y_proba_test))

# =========================================
# 6) Save the pipeline
# =========================================
joblib.dump(model, "model.pkl")
print("Model pipeline saved to model.pkl")

# =========================================
# 7) (Optional) Inference on new data
# =========================================
df_new = pd.read_excel("data_train.xlsx", sheet_name="TrainingData")


# Convert any range strings to numeric values in numeric columns
numeric_features = [
    "Ваш возраст",
    "Стаж курения",
    "Сколько пачек/штук употребляете за сутки? ",
    "Сколько раз в году болеете ОРВИ?"
]


for col in numeric_features:
    df_new[col] = df_new[col].replace("60 и выше", "60")
    df_new[col] = df_new[col].apply(convert_range_to_numeric)
    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

# Load the trained model pipeline
loaded_model = joblib.load("model.pkl")

# Predict using the loaded pipeline (the pipeline handles imputation and encoding)
y_pred_new = loaded_model.predict(df_new)
y_proba_new = loaded_model.predict_proba(df_new)[:, 1]

# Append predictions to df_new and save to a new Excel file
df_new["predicted_class"] = y_pred_new
df_new["predicted_proba"] = y_proba_new
df_new.to_excel("results_inference.xlsx", index=False)

print("Inference complete. Results saved to results_inference.xlsx")