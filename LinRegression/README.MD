## Linear Regression Exercises

```
def adjustedR2(x,y reg):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
```

## Override fit of sklearn to have p value and t-statistic 
```
  def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        
        # SSE (sum of squared errors)
        #  SE (standard error)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        # t-statistic
        self.t = self.coef_ / se
        # p-value for each feature
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self
```

## Calculate the univariate p-values of the variables
```
from sklearn.feature_selection import f_regression

freg=f_regression(x,y)

p=freg[1]

print(p.round(3))
```



### Create a summary table with your findings¶

```
reg_summary = pd.DataFrame([['size'],['year']],columns =['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p.round(3)
reg_summary.head()

	Features 	Coefficients 	p-values
0 	size 	227.700854 	0.000
1 	year 	2916.785327 	0.357
```

P value is more than 0.05 and fstat is very low - year is insignificant. can be dropped



## Scaled
```
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler().fit(x) 
rescaledX = scaler.transform(x) 
rescaledX
```


