## Simple Logistic Regression
```
x1=sm.add_constant(x)

resPre = sm.Logit(y,x1)

res=resPre.fit()

Optimization terminated successfully.
         Current function value: 0.546118
         Iterations 7

res.summary()


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

##
