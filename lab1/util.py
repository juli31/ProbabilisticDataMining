import numpy as np
import pandas as pd
import os
from sklearn import mixture

def df_fromarr(names, *args):
    df = pd.DataFrame()
    for i in range(len(args)):
        df[names[i]] = args[i]
    return df

def plot_letter_data(letter):
    xs = []
    ys = []
    files = [f for f in os.listdir("./Unistroke/") if ((f[0] == letter) and (f[1] != 'm'))]
    for item in files:
        with open("./Unistroke/"+item,'r') as f:
            for line in f.readlines():
                data = line.split('\t')
                for i in range(len(data)):
                    data[i] = float(data[i])
                xs.append(data[0])
                ys.append(data[1])
    return df_fromarr(['x','y'],xs,ys)

def get_letter_data(letter, amerge = False):
    data = np.empty((1,2))
    files = [f for f in os.listdir("./Unistroke/") if ((f[0] == letter) and (f[1] != 'm'))]
    for item in files:
        data = np.vstack((data, np.loadtxt("./Unistroke/"+item)))
    if amerge:
        data = np.loadtxt("./Unistroke/Amerge.txt")
    return data

def fitpredict_gmm(data):
    model = mixture.GaussianMixture(n_components = 2)
    labs = model.fit_predict(data)
    return labs

def prep_work_GMM():
    mean1 = [-3,0]
    mean2 = [3,0]
    lab1 = [1]*150

    cov1 = [[5,-2],[-2,1]]
    cov2 = [[5,2],[2,2]]
    lab2 = [2]*350

    x,y = np.random.multivariate_normal(mean1, cov1, size = 150).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 350).T
    xs = list(x) + list(x2)
    ys = list(y) + list(y2)
    labs = lab1 + lab2
    return df_fromarr(['x','y','class'], xs,ys,labs)
