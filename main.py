import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

df_train = pd.read_excel("data_train.xlsx", sheet_name="TrainingData")

numeric_features = ["Ваш возраст","Стаж курения","Сколько пачек/штук употребляете за сутки? ","Сколько раз в году болеете ОРВИ?"]
for col in numeric_features:
    df_train[col] = df_train[col].replace("60 и выше", 60)
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce")


X_train_full = df_train.drop(columns=["Вы болели раком?"])
y_train_full = df_train["Вы болели раком?"]
y_train_full = y_train_full.replace(regex='Да', value=1)
y_train_full = y_train_full.replace(regex='Нет', value=0)
y_train_full = y_train_full.fillna(0)

print("Строк в DataFrame:", len(X_train_full))

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

categorical_features = ["Пол","Онкоанамнез у родственников","Употребляете ли вы сигареты?","Место проживания"]
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

model = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
model.fit(X_train_sm, y_train_sm)

y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_test))

joblib.dump(model, "model.pkl")

df_new = pd.read_excel("data_inference.xlsx", sheet_name="Лист1")
loaded_model = joblib.load("model.pkl")
y_pred_new = loaded_model.predict(df_new)
y_proba_new = loaded_model.predict_proba(df_new)[:, 1]
df_new["predicted_class"] = y_pred_new
df_new["predicted_proba"] = y_proba_new
df_new.to_excel("results_inference.xlsx", index=False)
