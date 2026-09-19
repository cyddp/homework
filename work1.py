import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ISLP import load_data

Carseats = load_data('Carseats')
df = Carseats[['Sales','Price','Income','Advertising','ShelveLoc']]

X = df[['Price','Income','Advertising','ShelveLoc']]
y = df['Sales']
X_dummy = pd.get_dummies(X, columns=['ShelveLoc'], drop_first=True)
X_dummy = sm.add_constant(X_dummy)

model = sm.OLS(y, X_dummy).fit()
print(model.summary())

vif_data = pd.DataFrame()
vif_data["变量"] = X_dummy.columns
vif_data["VIF"] = [variance_inflation_factor(X_dummy.values, i) for i in range(X_dummy.shape[1])]
print(vif_data)
