import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "hand_gesture_dataset.csv"

df = pd.read_csv(DATASET_PATH)

X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Model Accuracy:", acc)

joblib.dump(model, "gesture_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model & Scaler Saved Successfully")
