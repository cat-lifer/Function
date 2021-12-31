# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:21:31 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

### 计算模型评价参数
import math
import numpy as np
from scipy.stats import pearsonr

def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_PCC(records_real, records_predict):
    '''
    真实值与预测值的相关性
    '''
    records_real = np.squeeze(records_real)
    records_predict = np.squeeze(records_predict)
    P_CC = pearsonr(records_real,records_predict)
    return P_CC[0]

 