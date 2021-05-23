## Simple Logistic Regression
```
x1=sm.add_constant(x)

resPre = sm.Logit(y,x1)

res=resPre.fit()

Optimization terminated successfully.
         Current function value: 0.546118
         Iterations 7

res.summary()
res.pred_table() ## givrs confusion matrix

df = df.append(pd.DataFrame({'duration': [130,120],'ans':[None, None]}))

df.tail()

new_x = df.loc[df.ans.isnull(), ['duration']]

print(new_x.head())

res.predict(sm.add_constant(new_x))
looks like probabilities
   duration
0       130
1       120

0    0.262015
1    0.252251
dtype: float64

```

### at times- errata could occur while using ol version of sm library
when summary returns an error

```
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
```
### pseudo r2

```
values of 0.2 to 0.4 for ρ2 represent EXCELLENT fit." So basically, ρ2 can be interpreted like R2, but don't expect it to be as big. And values from 0.2-0.4 indicate (in McFadden's words) excellent model fit.
```

## from stats model return both Confusion matrix and accuracy
```
def confusion_matrix(data,actual_values,model):
        #Predict the values using the Logit model
        pred_values = model.predict(data)
        # Specify the bins 
        bins=np.array([0,0.5,1])
        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
        # if they are between 0.5 and 1, they will be considered 1
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        # Calculate the accuracy
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        # Return the confusion matrix and the accuracy
        return cm, accuracy
cm = confusion_matrix(test_data,test_actual,res)
```


