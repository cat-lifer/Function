# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:35:01 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_input(path,Sheet,feature_num):
    Data_set = pd.read_excel(path, sheet_name = Sheet).drop('num',axis=1)
    Data =np.array(Data_set)
    X = Data[:,:feature_num]
    y = Data[:,-1].reshape(-1,1)
       
    return Data_set,Data,X,y

def data_norm(X,y,X1,y1):
    scaler = StandardScaler()
    X_norm = scaler.fit(X).transform(X1)
    y_norm = scaler.fit(y).transform(y1)
    return X_norm, y_norm
