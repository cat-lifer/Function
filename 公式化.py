# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:00:22 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

# 元素特征/材料参数————目标性能 公式化
'''
整体思路：
1. 导入已计算的特征
2. 相关性过滤：a) 用 PCC 选出与目标性能大于0.3（按情况）的特征
              b) 计算选出的特征之间的相关性，如果大于0.8，删除与目标性能相关性小的一项
3. LR：首先用多元线性回归尝试，输出整体的 MRE MSE R^2 PCC,作图
4. 符号回归尝试：gplearn库 ，+ - × ÷ log sqrt 等，输出公式，评价函数
'''

## 导入包
import Data as D
import Error as E
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

########################## 1.导入已计算的特征 ##########################
path =r"C:\Users\Uaena_HY\Desktop\公式化20210414\蠕变寿命\Version 2.0\新表.xlsx"  #按需更改
Sheet = 'Sheet3'
feature_num = 13 #长宽比是10  厚度是6  体积分数是6  （有几个特征）
Data_set,Data,X,y = D.data_input(path,Sheet,feature_num)

############################# 2.相关性过滤 #############################
# 计算全局相关性
p = Data_set.astype(float).corr()
# 过滤掉与目标性能极不相关的特征
p1 = p[((p.logt>0.2)|(p.logt<-0.2))]
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
sns.heatmap(p2,
            linewidths=0.1,
            vmax=1.0,
            square=True,
            center=0,
            cmap=colormap,
            linecolor='white',annot=True,annot_kws={"size": 22})

########################### 3.多项式特征 + Lasso ########################## 
# 整理后的数据表  
New_X = np.array(Data_set[remain_feature]) 
# 添加特征
pf=PolynomialFeatures(degree=2,include_bias=False)
New_XX = pf.fit_transform(New_X)

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(New_XX,y,test_size=.2,random_state=38)

#使用线性回归模型
lr = Lasso(alpha=0.01).fit(x_train,y_train)
y_pre = lr.predict(New_XX) 

print('\n')
print('lr training set score:{:.2f}'.format(lr.score(x_train,y_train)))
print('lr testing set score:{:.2f}'.format(lr.score(x_test,y_test))) 
print('\n')
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
plt.title('logt的真实值与计算值',fontsize=24) #设置标题
plt.xlabel('真实值',fontsize=14)
plt.ylabel('计算值',fontsize=14)
plt.show()

########################### 4.符号回归 ##########################
NX_train,NX_test,ny_train,ny_test = train_test_split(New_X,y,test_size=0.1,random_state=0)


function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log']
        
re_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           metric='rmse',
                           function_set=function_set,
                           parsimony_coefficient=0.01, random_state=38)

re_gp.fit(NX_train, ny_train)
y_gp_pre = re_gp.predict(New_X)
gp_pcc = E.get_PCC(y, y_gp_pre)

print('\n')
print(re_gp._program)
print('\n')
#print(re_gp.score(NX_test, ny_test))
print("pcc = %.*f"%(2,gp_pcc))