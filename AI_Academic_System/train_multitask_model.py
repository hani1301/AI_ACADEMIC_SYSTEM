import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("data/student_burnout_modified.csv")
print("Dataset Loaded Successfully")
print(df.head())

# ==============================
# 2. Encode Target Variables
# ==============================

burnout_encoder = LabelEncoder()
df["Burnout_Level"] = burnout_encoder.fit_transform(df["Burnout_Level"])

mental_encoder = LabelEncoder()
df["Mental_Health_Risk"] = mental_encoder.fit_transform(df["Mental_Health_Risk"])

# ==============================
# 3. Encode Categorical Features
# ==============================

df["Participation_in_Discussions"] = df["Participation_in_Discussions"].map({"Yes":1,"No":0})
df["Use_of_Educational_Tech"] = df["Use_of_Educational_Tech"].map({"Yes":1,"No":0})

df = pd.get_dummies(df, columns=["Gender","Preferred_Learning_Style"])

# ==============================
# 4. Feature Selection
# ==============================

X = df.drop(columns=[
    "Student_ID",
    "Final_Grade",
    "Burnout_Level",
    "Mental_Health_Risk"
])

X = X.select_dtypes(include=["int64","float64"])

y_burnout = to_categorical(df["Burnout_Level"])
y_mental = to_categorical(df["Mental_Health_Risk"])
y_productivity = df["Productivity_Score"].values

# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_burn_train, y_burn_test, y_ment_train, y_ment_test, y_prod_train, y_prod_test = train_test_split(
    X, y_burnout, y_mental, y_productivity,
    test_size=0.2,
    random_state=42
)

# ==============================
# 6. Scaling
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)

# ==============================
# 7. Build Deep Learning Multi-Task Model
# ==============================

input_layer = Input(shape=(X_train.shape[1],))

x = Dense(128, activation="relu")(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)

burnout_output = Dense(3, activation="softmax", name="burnout")(x)
mental_output = Dense(3, activation="softmax", name="mental")(x)
productivity_output = Dense(1, activation="linear", name="productivity")(x)

model = Model(
    inputs=input_layer,
    outputs=[burnout_output, mental_output, productivity_output]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        "burnout": "categorical_crossentropy",
        "mental": "categorical_crossentropy",
        "productivity": "mse"
    },
    metrics={
        "burnout": ["accuracy"],
        "mental": ["accuracy"],
        "productivity": ["mae"]
    }
)

# ==============================
# 8. Train Model
# ==============================

history = model.fit(
    X_train,
    {
        "burnout": y_burn_train,
        "mental": y_ment_train,
        "productivity": y_prod_train
    },
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ==============================
# 9. Save Model & Encoders
# ==============================

encoders = {
    "burnout": burnout_encoder,
    "mental": mental_encoder
}

model.save("models/multitask_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Deep Learning Multi-task Model Saved Successfully")
# Scale panna apram ithu panni terminal la vara list-ah copy pannunga
print("SCALER_COLUMNS:", X.columns.tolist())

import pickle

# ===========================================
# Save training history and test artifacts
# ===========================================

os.makedirs("results", exist_ok=True)

# Save training history
with open("results/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Save test data and class labels
artifacts = {
    "X_test_s": X_test,
    "yb_test_i": np.argmax(y_burn_test, axis=1),
    "ym_test_i": np.argmax(y_ment_test, axis=1),
    "yp_test": y_prod_test,
    "burnout_classes": list(burnout_encoder.classes_),
    "mental_classes": list(mental_encoder.classes_)
}

with open("results/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ history.pkl saved")
print("✅ artifacts.pkl saved")