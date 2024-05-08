import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5  # Adding some noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)  # Linear regression with single neuron
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using mean squared error as loss function

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)  # Adjust epochs and batch_size as needed

# Evaluate the model on the testing data
mse = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Mean Squared Error on test set:", mse)

# Make predictions
predictions = model.predict(X_test_scaled)

# Display a few predictions
for i in range(5):
    print("Predicted:", predictions[i][0], "Actual:", y_test[i][0])
