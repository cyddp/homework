import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",header=None)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_svm = SVC(kernel='linear',C=1)
linear_svm.fit(X_train,y_train)
y_pred_linear = linear_svm.predict(X_test)

rbf_svm = SVC(kernel='rbf',C=10,gamma=0.01)
rbf_svm.fit(X_train,y_train)
y_pred_rbf = rbf_svm.predict(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)

print("线性SVM准确率",accuracy_score(y_test,y_pred_linear))
print("RBF核SVM准确率",accuracy_score(y_test,y_pred_rbf))
print("逻辑回归准确率",accuracy_score(y_test,y_pred_lr))
print("\n线性SVM分类报告")
print(classification_report(y_test,y_pred_linear))
print("\nRBF SVM分类报告")
print(classification_report(y_test,y_pred_rbf))
print("\n逻辑回归分类报告")
print(classification_report(y_test,y_pred_lr))

param_grid = {"C":[0.01,0.1,1,10,100],"gamma":[0.0001,0.001,0.01,0.1,1]}
grid = GridSearchCV(SVC(kernel="rbf"),param_grid,cv=5)
grid.fit(X_train,y_train)
print("\n网格搜索最优参数",grid.best_params_)
print("交叉验证最优得分",grid.best_score_)
best_model = grid.best_estimator_
y_best_pred = best_model.predict(X_test)
print("最优参数模型测试集准确率",accuracy_score(y_test,y_best_pred))

support_count = best_model.n_support_
print("每一类支持向量数量",support_count)
