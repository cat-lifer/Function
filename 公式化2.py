# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:42:47 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

'''
整体思想：1. 将所有特征进行乘、除、log、sqrt计算，扩充特征
         2. 进行pearson相关性计算，挑出符合要求的特征
         3. Lasso 回归、多项式回归
'''

import Data as D
import Error as E
import numpy as np
import pandas as pd
import seaborn as sns
import Create_Features as C
import matplotlib.pyplot as plt 
from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


############################### 数据导入 ###############################
path = r"E:\科研\公式化20210414\蠕变寿命\Version 2.0\新表.xlsx"  
Sheet ='Sheet1'
feature_num = 49 #有几个特征
Data_set,Data,X,y = D.data_input(path,Sheet,feature_num)

########################### 1.生成特征，特征组合 ###########################
feats = [i for i in Data_set.columns if i not in ['logt']]

## 两两组合
new_feats=[]
for i in combinations(feats, 2):
    new_feats.append(i)

## 乘法项
data_mul=pd.DataFrame()
for k in range(len(new_feats)):
    new = list(new_feats[k]) 
    data1 = C.mul_feats(Data_set,new)
    data_mul = pd.concat([data_mul,data1],axis=1) 
    
## 除法项
data_div=pd.DataFrame()
for l in range(len(new_feats)):
    new = list(new_feats[l]) 
    data2 = C.div_feats(Data_set,new)
    data_div = pd.concat([data_div,data2],axis=1)   
    
## log项
data_log=pd.DataFrame()
for m in range(len(feats)): 
    data3 = C.log_feats(Data_set,feats[m])
    data_log = pd.concat([data_log,data3],axis=1)   

## sqrt项
data_sqrt=pd.DataFrame()
for n in range(len(feats)): 
    data4 = C.sqrt_feats(Data_set,feats[n])
    data_sqrt = pd.concat([data_sqrt,data4],axis=1)
    
## square项
data_square=pd.DataFrame()
for o in range(len(feats)): 
    data5 = C.square_feats(Data_set,feats[o])
    data_square = pd.concat([data_square,data5],axis=1)    

## 所有特征拼在一起
Final_set0 = pd.concat([data_mul,data_div,data_log,data_sqrt,data_square,Data_set],axis=1)  
Final_set1 = Final_set0.replace(np.inf, np.nan) #替换正inf为NaN  
Final_set = Final_set1.dropna(axis=1)   #删掉含有NaN的列

########################### 1.相关性分析 ###########################  
# # 计算全局相关性
p = Final_set.astype(float).corr()
# 过滤掉与目标性能极不相关的特征
p1 = p[((p.logt>0.5)|(p.logt<-0.5))]
p11 = p1[p1.index]
low2high = p11.iloc[p11['logt'].abs().argsort()] #按绝对值从小到大排列

low2high = low2high.drop('logt',axis=1)
# 过滤掉互相关的特征
n = low2high.shape[1]
p2 = low2high
for i in range(n):
    dex = low2high.index[i]
    if(((p2.loc[dex]>0.8)|(p2.loc[dex]<-0.8))&(p2.loc[dex]<1)).any():
        p2 = p2.drop(dex)
        p2 = p2.drop(dex,axis=1)

p3 = p2.drop(p2.index[len(p2)-1])
remain_feature = p3.index  #剩余符合要求的特征
#画图
colormap = plt.cm.RdBu 
plt.rcParams['font.sans-serif']=['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False    #显示负号  
sns.heatmap(p2,
            linewidths=0.1,
            vmax=1.0,
            square=True,
            center=0,
            cmap=colormap,
            linecolor='white',annot=True,annot_kws={"size": 9})  

########################### 3.多项式特征 + Lasso ########################## 
# 整理后的数据表 
remain_set = Final_set[remain_feature]
#remain_set = Final_set
New_X = np.array(remain_set) 
# 添加特征
pf=PolynomialFeatures(degree=1,include_bias=False)
New_XX = pf.fit_transform(New_X)

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(New_XX,y,test_size=.1,random_state=38)

#使用线性回归模型
lr = Lasso(alpha=0.0015).fit(x_train,y_train)
y_pre = lr.predict(New_XX) 

print('\n')
print('lr training set score:{:.2f}'.format(lr.score(x_train,y_train)))
print('lr testing set score:{:.2f}'.format(lr.score(x_test,y_test))) 
print('\n')
print('使用的特征数：{}'.format(np.sum(lr.coef_!=0)))

# 系数和截距
lr_coef = lr.coef_
lr_intercept = lr.intercept_
# 全局评价参数
rr = lr.score(New_XX, y)
mse = E.get_mse(y, y_pre)
rmse = E.get_rmse(y, y_pre)
mae = E.get_mae(y, y_pre)
p_cc = E.get_PCC(y, y_pre)

print(lr_coef)
print('\n')
print(lr_intercept)
print('\n') 

print("rr = %.*f"%(2,rr))
print("mse = %.*f"%(2,mse))
print("rmse = %.*f"%(2,rmse))
print("mae = %.*f"%(2,mae))
print("pcc = %.*f"%(2,p_cc))
print('\n')

#计算画图
max_logt = max(max(y),max(y_pre))
min_logt = min(min(y),min(y_pre))
plt.figure()
plt.scatter(y,y_pre)
plt.xlim(min_logt,max_logt) #修改x y的坐标轴范围
plt.ylim(min_logt,max_logt)
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.title('logt的真实值与计算值,alpha=0.0015',fontsize=20) #设置标题
plt.xlabel('真实值',fontsize=14)
plt.ylabel('计算值',fontsize=14)
plt.show()