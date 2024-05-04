import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(0)
n_samples = 1000
weather = np.random.rand(n_samples, 1) * 50
occupancy = np.random.rand(n_samples, 1) * 100
energy_demand = 10 * weather + 5 * occupancy + np.random.randn(n_samples, 1) * 5


X = np.concatenate((weather, occupancy), axis=1)
y = energy_demand.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
