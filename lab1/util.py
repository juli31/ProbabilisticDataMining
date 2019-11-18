import numpy as np
import pandas as pd
import os
from sklearn import mixture
from matplotlib import pyplot as plt

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

def plot_pdf(model, values, labels, comp):
    x = np.linspace(-1.5,1.5)
    y = np.linspace(-1.5,1.5)
    X, Y = np.meshgrid(x,y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -model.score_samples(XX)
    Z = Z.reshape(X.shape)
    # plot contour
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.scatter(data[:, 0], data[:, 1],.8, c=labels)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
