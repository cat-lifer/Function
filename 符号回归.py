# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:27:04 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

################################ 用遗传算法实现符号回归 ###############################

# Genetic Programming in Python, with a scikit-learn inspired API: gplearn
# https://gplearn.readthedocs.io/en/latest/index.html

'''
参数介绍
population_size : 整数，可选(默认值=1000)种群规模(每一代个体数目即初始树的个数)。

generations : 整数，可选(默认值=20)要进化的代数。

tournament_size : 整数，可选(默认值=20)进化到下一代的个体数目(从每一代的所有公式中，tournament_size个公式会被随机选中，其中适应度最高的公式将被认定为生存竞争的胜利者，进入下一代。tournament_size的大小与进化论中的选择压力息息相关：tournament_size越小，选择压力越大，算法收敛的速度可能更快，但也有可能错过一些隐藏的优秀公式)。

stopping_criteria : 浮点数，可选(默认值=0.0)停止条件。

const_range : 两个浮点数组成的元组，或none，可选(默认值=(-1，1))公式中所要包含的常量取值范围。如果设为none，则无常数。

init_depth : 两个整数组成的元组，可选(默认值=(2，6))用来表示原始公式初始总体的树深度范围，树的初始深度将处在(min_depth, max_depth)的区间内(包含端点)。原始公式初始总体的树深度范围，单个树将随机选择此范围内的最大深度。

init_method : 字符串, 可选(默认值=‘half and half’)控制每棵公式树的初始化方式，有三种策略：
grow：公式树从根节点开始生长。在每一个子节点，gplearn会从所有常数、变量和函数中随机选取一个元素。如果它是常数或者变量，那么这个节点会停止生长，成为一个叶节点。如果它是函数，那么它的两个子节点将继续生长。用grow策略生长得到的公式树往往不对称，而且普遍会比用户设置的最大深度浅一些；在变量的数量远大于函数的数量时，这种情况更明显。
full：除了最后一层外，其他所有层的所有节点都是内部节点——它们都只是随机选择的函数，而不是变量或者常数。最后一层的叶节点则是随机选择的变量和常数。用full策略得到的公式树必然是perfect binary tree。
half and half：一半的公式树用grow策略生成，另一半用full策略生成。因为种群的多样性有利于生存，所以这是init_method参数的默认值。
function_set : 字符串, 用于符号回归的函数，包括gplearn原始提供以及自定义
metric : 字符串, 目标函数(损失函数) (默认值=‘MAE’(平均绝对误差))，此外还包括gplearn提供的mse等，也可以自定义。

parsimony_coefficient (简约系数): 浮点数或 “auto”, 可选 (默认值=0.001)用于惩罚过于复杂的公式。简约系数往往由实践验证决定。如果过于吝啬（简约系数太大），那么所有的公式树都会缩小到只剩一个变量或常数；如果过于慷慨（简约系数太小），公式树将严重膨胀。不过，gplearn已经提供了’auto’的选项，能自动控制节俭项的大小。
类似

p_crossover : 浮点数, 可选 (默认值=0.9)对胜者进行交叉的概率，用于合成新的树

p_subtree_mutation : 浮点数, 可选 (默认值=0.01)控制胜者中进行子树变异的比例(优胜者的一棵子树将被另一棵完全随机的全新子树代替)所选值表示进行子树突变的部分。

p_hoist_mutation : 浮点数, 可选 (默认值=0.01) 控制进行hoist变异的比例，hoist变异是一种对抗公式树膨胀（bloating，即过于复杂）的方法：从优胜者公式树内随机选择一个子树A，再从A里随机选择一个子树B，然后把B提升到A原来的位置，用B替代A。hoist的含义即「升高、提起」。
p_point_mutation : 浮点数, 可选 (默认值=0.01)控制点进行突变的比例

p_point_replace : 浮点数, 可选 (默认值=0.05)对于点突变时控制某些点突变的概率。

max_samples : 浮点数, 可选 (默认值=1.0)从样本中抽取的用于评估每个树(成员)的百分比

feature_names : list(列表), 可选 (默认值=None)因子名(或特征名)若为none则用x0，x1等表示。

warm_start : 布尔型, 可选 (默认值=False)用于选择是否使用之前的解决方案

low_memory : 布尔型, 可选 (默认值=False)用于选择是否只保留当前一代

n_jobs : 整数，可选(默认值=1)用于设置并行计算的操作
'''
'''
‘generation’ : The generation index.
‘average_length’ : The average program length of the generation.
‘average_fitness’ : The average program fitness of the generation.
‘best_length’ : The length of the best program in the generation.
‘best_fitness’ : The fitness of the best program in the generation.
‘best_oob_fitness’ : The out of bag fitness of the best program in the generation (requires max_samples < 1.0).
‘generation_time’ : The time it took for the generation to evolve.
'''

import numpy as np
import Data as D
import Error as E
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from gplearn.genetic import SymbolicTransformer,SymbolicRegressor
from sklearn.model_selection import  train_test_split

# 数据准备
path =r"C:\Users\Uaena_HY\Desktop\公式化20210414\蠕变寿命\Version 2.0\新表.xlsx"  #按需更改
name = 'Sheet5'
num = 5 #长宽比是10  厚度是6  体积分数是6
Data_set,Data,X,y = D.data_input(path,name,num)
#X_norm, y_norm = D.data_norm(X,y,X,y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=38)
#X_train,X_test,y_train,y_test = train_test_split(X_norm,y_norm,test_size=0.1,random_state=1)

function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log']
        
##SymbolicRegressor
re_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           function_set=function_set,
                           parsimony_coefficient=0.01, random_state=38)

re_gp.fit(X_train, y_train)
y_gp_pre = re_gp.predict(X)
gp_pcc = E.get_PCC(y, y_gp_pre)

print('\n')
print(re_gp._program)
print('\n')
#print(re_gp.score(NX_test, ny_test))
print("pcc = %.*f"%(2,gp_pcc))



# ## SymbolicTransformer
# tf_gp = SymbolicTransformer(generations=20, population_size=5000,
#                          hall_of_fame=100, n_components=3,
#                          function_set=function_set,
#                          parsimony_coefficient=0.0005,
#                          max_samples=0.9, verbose=1,
#                          random_state=38, n_jobs=3)

# tf_gp.fit(X_train, y_train)
# gp_features = tf_gp.transform(X)
# new_boston = np.hstack((X, gp_features))

# est = Lasso()
# est.fit(new_boston[:40, :], y[:40])
# print(est.score(new_boston[40:, :], y[40:]))
# print(tf_gp)