## bASIC nEURAL NET- NO LIBRARY
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
