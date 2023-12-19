import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
x = data[["House Age","Nearest Train Station","Number Convenience Stores"]].values
y = data["Price"].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

model = LinearRegression().fit(xtrain, ytrain)

coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y), 2)

print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)

# Predicted values

predict = model.predict(xtest)
predict = np.around(predict, 2)
print(predict)

print("*******************\nTesting Results")
for i in range(len(xtest)):
    actual = ytest[i]
    predicted_y = predict[i]
    x_coord = xtest[i]
    print(f"House Age: {x_coord[0]} Nearest Train Station: {x_coord[1]} Number of Convenience Stores: {x_coord[2]} Actual: {actual} Predicted: {predicted_y}")
    print()
 
 
# Graphs

x_1 = data["House Age"]
x_2 = data["Nearest Train Station"]
x_3 = data["Number Convenience Stores"]
y = data["Price"]

fig, graph = plt.subplots(3)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("House Age")
graph[0].set_ylabel("Price")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("Nearest Train Station")
graph[1].set_ylabel("Price")

graph[2].scatter(x_3, y)
graph[2].set_xlabel("Number of Convenience Stores")
graph[2].set_ylabel("Price")

print("Correlation between House Age and Price Population:",round(x_1.corr(y),2))
print("Correlation between Nearest Train Station and Price Population:",round(x_2.corr(y),2))
print("Correlation between Number of Convenience Stores and Price Population:",round(x_3.corr(y),2))

plt.tight_layout()
plt.show()