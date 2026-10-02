import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("Hitters.csv")
df = df.dropna(subset=["Salary"])
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 4, 100), scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_scaled, y_train)
print("Ridge最优alpha：", ridge_cv.alpha_)
ridge_pred = ridge_cv.predict(X_test_scaled)
print("Ridge测试集MSE：", mean_squared_error(y_test, ridge_pred))

lasso_cv = LassoCV(alphas=np.logspace(-3, 4, 100), cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
print("Lasso最优alpha：", lasso_cv.alpha_)
lasso_pred = lasso_cv.predict(X_test_scaled)
print("Lasso测试集MSE：", mean_squared_error(y_test, lasso_pred))

coef = pd.Series(lasso_cv.coef_, index=X.columns)
print("\nLasso变量系数：")
print(coef)

alpha_range = np.logspace(-3,4,100)
ridge_coefs = []
lasso_coefs = []

for a in alpha_range:
    r = Ridge(alpha=a)
    r.fit(X_train_scaled,y_train)
    ridge_coefs.append(r.coef_)
    
    l = Lasso(alpha=a,max_iter=10000)
    l.fit(X_train_scaled,y_train)
    lasso_coefs.append(l.coef_)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))

ax1.plot(alpha_range, ridge_coefs)
ax1.set_xscale("log")
ax1.set_title("Ridge 系数路径图")
ax1.set_xlabel("alpha")
ax1.set_ylabel("系数值")
ax1.grid(alpha=0.3)

ax2.plot(alpha_range, lasso_coefs)
ax2.set_xscale("log")
ax2.set_title("Lasso 系数路径图")
ax2.set_xlabel("alpha")
ax2.set_ylabel("系数值")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
