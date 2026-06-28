import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


iris = pd.read_csv("IRIS.csv")


print("Dataset Loaded Successfully")
print("\nFirst 5 Rows:\n")
print(iris.head())


X = iris.drop("species", axis=1)

y = iris["species"]



encoder = LabelEncoder()

y = encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



model = RandomForestClassifier(

    n_estimators=120,

    random_state=42

)



model.fit(

    X_train,

    y_train

)



predictions = model.predict(X_test)



accuracy = accuracy_score(

    y_test,

    predictions

)




print("MODEL PERFORMANCE")





print(f"Accuracy: {accuracy:.4f}")



print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)



sample_flower = [[5.1,3.5,1.4,0.2]]

result = model.predict(sample_flower)


print("\nPredicted Species:")

print(
    encoder.inverse_transform(result)[0]
)



comparison = pd.DataFrame({

    "Actual":

    encoder.inverse_transform(y_test),


    "Predicted":

    encoder.inverse_transform(predictions)

})


print("\nActual vs Predicted:")

print(comparison.head(10))



plt.figure(figsize=(10,5))


plt.plot(

    comparison["Actual"][:30],

    marker="o",

    label="Actual"

)


plt.plot(

    comparison["Predicted"][:30],

    marker="x",

    label="Predicted"

)



plt.title("Actual vs Predicted Iris Species")


plt.xlabel("Samples")


plt.ylabel("Species")


plt.legend()


plt.grid()


plt.show()