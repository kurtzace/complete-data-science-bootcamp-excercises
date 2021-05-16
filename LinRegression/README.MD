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

later to predict
```
actualpredictX = pd.DataFrame({'size':[750], 'year':[2009]})
rescaledpredictX = scaler.transform(actualpredictX) 
reg.predict(rescaledpredictX)
```

## Vif

https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
https://statisticalhorizons.com/multicollinearity

snippet from above
```
obtaining the R2 from that regression. The VIF is just 1/(1-(R^2)).
Estimates how much the variance of a coefficient is “inflated” because of linear dependence with other predictors. Thus, a VIF of 1.8 tells us that the variance (the square of the standard error) of a particular coefficient is 80% larger than it would be if that predictor was completely uncorrelated with all the other predictors.

The VIF has a lower bound of 1 but no upper bound. Authorities differ on how high the VIF has to be to constitute a problem. Personally, I tend to get concerned when a VIF is greater than 2.50, which corresponds to an R2 of .60 with the other variables.

above talks about three situations in which a high VIF is not a problem and can be safely ignored:
-  (dummy) variables
-  If you specify a regression model with both x and x2, there’s a good chance that those two variables will be highly correlated. 
-  high VIFs are control variables: the sample consists of U.S. colleges, the dependent variable is graduation rate, and the variable of interest is an indicator (dummy) for public vs. private. Two control variables are average SAT scores and average ACT scores for entering freshmen. 
```
in general
![image](https://user-images.githubusercontent.com/2136211/118392467-871a7480-b657-11eb-95de-261274ff2865.png)

```
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
```


## What are the reference/benchmark categories for each cat?
```
data_cleaned['name_of_categorical_variable'].unique() gives some missing.
```
The benchmark categories are as follows:
reorder the categorical variables prior to using .get_dummies() to customize

## All steps
![AllStepsLinearReg](https://user-images.githubusercontent.com/2136211/118398745-a2957780-b677-11eb-8720-ea3dfb02e11b.png)



