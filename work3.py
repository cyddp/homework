import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

df = pd.read_csv("netflix.csv")
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("imdb_rating", axis=1)
y = df["imdb_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

elastic_cv = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9], alphas=np.logspace(-3,3,80), cv=5, random_state=42)
elastic_cv.fit(X_train_scaled, y_train)
print("ElasticNet最优alpha：", elastic_cv.alpha_)
print("ElasticNet最优l1_ratio：", elastic_cv.l1_ratio_)
elastic_pred = elastic_cv.predict(X_test_scaled)
print("ElasticNet测试集MSE：", mean_squared_error(y_test, elastic_pred))

coef = pd.Series(elastic_cv.coef_, index=X.columns)
print("\nElasticNet变量系数：")
print(coef)

alpha_range = np.logspace(-3,3,80)
elastic_coefs = []

for a in alpha_range:
    en = ElasticNet(alpha=a, l1_ratio=elastic_cv.l1_ratio_, max_iter=10000)
    en.fit(X_train_scaled,y_train)
    elastic_coefs.append(en.coef_)

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(alpha_range, elastic_coefs)
ax.set_xscale("log")
ax.set_title("ElasticNet 系数路径图")
ax.set_xlabel("alpha")
ax.set_ylabel("系数值")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
