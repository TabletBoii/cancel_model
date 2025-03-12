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


df_train["Вы болели раком?"] = df_train["Вы болели раком?"].replace({"Да": 1, "Нет": 0}).fillna(0)


X_train_full = df_train.drop(columns=["Вы болели раком?"])
y_train_full = df_train["Вы болели раком?"]


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


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])


categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fills cat NaNs with mode
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

model = Pipeline([
    ("preproc", preprocessor),
    ("smote", SMOTE(random_state=42, k_neighbors=2)),  # reduce k_neighbors
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
])


model.fit(X_train, y_train)


y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred_test))
print("=== ROC-AUC ===")
print(roc_auc_score(y_test, y_proba_test))


joblib.dump(model, "model.pkl")
print("Model pipeline saved to model.pkl")


df_new = pd.read_excel("data_train.xlsx", sheet_name="TrainingData")

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

loaded_model = joblib.load("model.pkl")

y_pred_new = loaded_model.predict(df_new)
y_proba_new = loaded_model.predict_proba(df_new)[:, 1]

df_new["predicted_class"] = y_pred_new
df_new["predicted_proba"] = y_proba_new
df_new.to_excel("results_inference.xlsx", index=False)

print("Inference complete. Results saved to results_inference.xlsx")