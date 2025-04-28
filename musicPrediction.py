#first AI project :O
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas
import numpy 

data = {
    "Age": numpy.random.randint(10, 70, 500),
    "genre": numpy.random.choice(["rock", "pop", "classical"], 500)
}

dataFrame = pandas.DataFrame(data)
dataFrame.loc[dataFrame["Age"] <= 25, "Genre"] = "pop"
dataFrame.loc[(dataFrame["Age"] > 25) & (dataFrame["Age"] <= 45), "Genre"] = "Rock"
dataFrame.loc[dataFrame["Age"] > 45, "Genre"] = "Classical"

X = dataFrame["Age"]
y = dataFrame["Genre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state= 42)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

def predict_genre(age):
    predictionData = pandas.DataFrame({"Age": [age]})
    return model.predict(predictionData)[0]

userAge = int(input("What is your age: "))
predictedGenre = predict_genre(userAge)
print(f"Based on your age, you most likely like {predictedGenre} music")