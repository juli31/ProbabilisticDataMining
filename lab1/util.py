import numpy as np
import pandas as pd
import os
from sklearn import mixture

def df_fromarr(names, *args):
    df = pd.DataFrame()
    for i in range(len(args)):
        df[names[i]] = args[i]
    return df

def get_data(pathto, files, plot_ready = False):
    flag = 0
    for item in files:
        if flag == 0:
            data = np.loadtxt(pathto+item)
            flag = 1
        else:
            data = np.vstack((data, np.loadtxt(pathto+item)))
    return data

def prep_work_GMM():
    lab1 = [1]*150
    lab2 = [2]*350
    x,y = np.random.multivariate_normal([-3,0], [[5,-2],[-2,1]], size = 150).T
    x2, y2 = np.random.multivariate_normal([3,0], [[5,2],[2,2]], 350).T
    xs = list(x) + list(x2)
    ys = list(y) + list(y2)
    labs = lab1 + lab2
    return df_fromarr(['x','y','class'], xs,ys,labs)
