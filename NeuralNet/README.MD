## bASIC nEURAL NET- NO LIBRARY
![image](https://user-images.githubusercontent.com/2136211/120102918-df2e9c00-c16a-11eb-8594-f4af03dab5b7.png)

```

init_range = 0.1
weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1)) # WHEN INPUT IS (1000000, 2)
biases = np.random.uniform(low=-init_range, high=init_range, size=1)
print (weights)
print (biases)
learning_rate = 0.02
for i in range (100):
    outputs = np.dot(inputs,weights) + biases
    
    
    deltas = outputs - targets
    
    
    loss = np.sum(deltas ** 2) / 2 / observations
    
    
    print (loss)

    
    deltas_scaled = deltas / observations
	
weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
biases = biases - learning_rate * np.sum(deltas_scaled)
```

## basic nn with TF
note: The Huber loss is more appropriate than the L2-norm when we have outliers, as it is less sensitive to them 
```
#gen data
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations,1))
generated_inputs = np.column_stack((xs,zs))
noise = np.random.uniform(-1, 1, (observations,1))
generated_targets = 2*xs - 3*zs + 5 + noise
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

## training
training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size,

                                                 kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                                 bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
                                                 )
                            ])
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)
model.layers[0].get_weights()
model.predict_on_batch(training_data['inputs']).round(1)
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
```
