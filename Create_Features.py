# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:48:30 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

## 生成特征:根据元特征，通过加减乘除创造新特征

## 导入包

import numpy as np



# 乘法 ，只支持两列
def mul_feats(data,feats):
    tmp = data.copy(deep=True)
    feats_name = '*'.join(feats)
    tmp[feats_name] = tmp[feats[0]] * tmp[feats[1]]
    #columns = [i for i in tmp.columns if i not in feats]
    return tmp[feats_name]

# 除法，只支持两列
def div_feats(data,feats):
    tmp = data.copy(deep=True)
    feats_name = '/'.join(feats)
    tmp[feats_name] = tmp[feats[0]] / tmp[feats[1]]
    #columns = [i for i in tmp.columns if i not in feats]
    return tmp[feats_name]

# 取log
def log_feats(data,feats):
    tmp = data.copy(deep=True)
    aa=list(feats)
    aa.insert(0,'log')
    feats_name = ''.join(aa)
    tmp[feats_name] = np.log(tmp[feats])
    #columns = [i for i in tmp.columns if i not in feats]
    return tmp[feats_name]

# 开方sqrt
def sqrt_feats(data,feats):
    tmp = data.copy(deep=True)
    bb=list(feats)
    bb.insert(0,'sqrt')
    feats_name = ''.join(bb)
    tmp[feats_name] = np.sqrt(tmp[feats])
    #columns = [i for i in tmp.columns if i not in feats]
    return tmp[feats_name]

# 平方square
def square_feats(data,feats):
    tmp = data.copy(deep=True)
    bb=list(feats)
    bb.insert(0,'square')
    feats_name = ''.join(bb)
    tmp[feats_name] = np.square(tmp[feats])
    #columns = [i for i in tmp.columns if i not in feats]
    return tmp[feats_name]