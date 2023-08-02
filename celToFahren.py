import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

celsius_q    = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit_a = np.array([-40,14,32,46,59,72,100],dtype=float)

list = enumerate(celsius_q)
for i,c in list:
    print("{} degrees Celsius = {} degrees Fahrenheit" .format(c,fahrenheit_a[i]))

# Define the layers. Here we are using one layer with just one neuron.      
l0 = tf.keras.layers.Dense(units = 4,input_shape = [1])
l1 = tf.keras.layers.Dense(units = 4)
l2 = tf.keras.layers.Dense(units = 1)

# Define the model from the layers above.
model = tf.keras.Sequential([l0,l1,l2])

# Compile the model defined above.
model.compile(loss = 'mean_squared_error',optimizer = tf.keras.optimizers.Adam(0.1))

# Train our model using 500 epochs.
history = model.fit(celsius_q,fahrenheit_a,epochs = 500, verbose = False)

plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history["loss"])

# Now we can validate our model using the predict function.
print(model.predict([100.0]))
