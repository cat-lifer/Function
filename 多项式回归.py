# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:33:30 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

## 多项式回归
import Error as E
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge
import Data as D
from sklearn.model_selection import  train_test_split
from sklearn.metrics import r2_score

## 导入数据
path =r"E:\科研\公式化20210414\蠕变寿命\Version 2.0\结果.xlsx"  #按需更改
name = '前5个最强特征'
num = 5 #几个输入特征
Data_set,Data,X,y = D.data_input(path,name,num)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)

##  多项式回归
lr = LinearRegression()
pf=PolynomialFeatures(degree=1,include_bias=False)
lr.fit(pf.fit_transform(X_train),y_train)
xx_quadratic = pf.transform(X_test)
y_p=lr.predict(xx_quadratic)
score1 = r2_score(y_test, y_p)

coef =lr.coef_
intercept = lr.intercept_
print(lr.coef_)
print('\n')
print(lr.intercept_)
print('\n')
xx = pf.transform(X)
y_pre=lr.predict(xx)
score = r2_score(y, y_pre)
mse = E.get_mse(y, y_pre)
rmse = E.get_rmse(y, y_pre)
mae = E.get_mae(y, y_pre)
p_cc = E.get_PCC(y, y_pre)
print("rr = %.*f"%(2,score))
print("mse = %.*f"%(2,mse))
print("rmse = %.*f"%(2,rmse))
print("mae = %.*f"%(2,mae))
print("pcc = %.*f"%(2,p_cc))
print('\n')

##Ridge
ridge = Ridge(alpha=0.01,max_iter=100000).fit(X_train,y_train)
y2_pre = ridge.predict(X)

ridge_score = ridge.score(X,y)
r_score = r2_score(y, y2_pre)
r_mse = E.get_mse(y, y2_pre)
r_rmse = E.get_rmse(y, y2_pre)
r_mae = E.get_mae(y, y2_pre)
r_p_cc = E.get_PCC(y, y2_pre)

r_coef =ridge.coef_
r_intercept = ridge.intercept_
print(r_coef)
print('\n')
print(r_intercept)
print('\n')

print("r_rr = %.*f"%(2,r_score))
print("r_mse = %.*f"%(2,r_mse))
print("r_rmse = %.*f"%(2,r_rmse))
print("r_mae = %.*f"%(2,r_mae))
print("r_pcc = %.*f"%(2,r_p_cc))