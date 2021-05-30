
#Remove the duplicate index column from the dataset.

either
```
df = (df.reset_index()

        .drop_duplicates(subset='index', keep='last')

        .set_index('index').sort_index())
        
```
or
```
data = data.drop(['Unnamed: 0'], axis = 1)
```

## WCSS
```
# Create an empty list
wcss=[]

# Create all possible cluster solutions with a loop
for i in range(1,7):
    # Cluster solution with i clusters
    kmeans = KMeans(i)
    # Fit the data
    kmeans.fit(x)
    # Find WCSS for the current iteration
    wcss_iter = kmeans.inertia_
    # Append the value to the WCSS list
    wcss.append(wcss_iter)
    
# Create a variable containing the numbers from 1 to 6, so we can use it as X axis of the future plot
number_clusters = range(1,7)
# Plot the number of clusters vs WCSS
plt.plot(number_clusters,wcss)
``` 
makes elbow
