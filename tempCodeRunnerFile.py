import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# LOAD DATASET
# =====================================

iris = pd.read_csv("IRIS.csv")

print("Dataset Loaded Successfully")
print("\nFirst 5 Rows:\n")
print(iris.head())

# =====================================
# FEATURES AND TARGET
# =====================================

X = iris.drop("species", axis=1)
y = iris["species"]

# =====================================
# ENCODE TARGET VARIABLE
# =====================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# MODEL TRAINING
# =====================================

model = RandomForestClassifier(
    n_estimators=120,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# PREDICTION
# =====================================

predictions = model.predict(X_test)

# =====================================
# EVALUATION
# =====================================

accuracy = accuracy_score(y_test, predictions)

print("\n========================")
print("MODEL PERFORMANCE")
print("========================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# =====================================
# SAMPLE FLOWER PREDICTION
# =====================================

sample_flower = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(sample_flower)

print("\nPredicted Species:")
print(encoder.inverse_transform(result)[0])