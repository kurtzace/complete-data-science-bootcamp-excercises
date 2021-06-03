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
model.compile(optimizer=custom_optimizer, loss='mean_squared_error') #metrics accuracy
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)
model.layers[0].get_weights()
model.predict_on_batch(training_data['inputs']).round(1)
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
```

![image](https://user-images.githubusercontent.com/2136211/120599093-b3f6c600-c464-11eb-92a9-b88f07d0def8.png)

## Deal with dataset from TF and extract validation out of it
directly from TensorFlow_MNIST_Part3_with_comments in https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/learn/lecture/15089010
```
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True, data_dir="D:\Reference\AI\completeDsUdemy\dataDir")

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)
def scale(image, label):
    
    image = tf.cast(image, tf.float32)
    
    
    image /= 255.
    return image, label
scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)
BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)
validation_inputs, validation_targets = next(iter(validation_data))
```
