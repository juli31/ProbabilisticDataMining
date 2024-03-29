import numpy as np
import pandas as pd
import statistics as stat
import os
from sklearn import mixture
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import altair as alt

def plot(*args, **kwargs):
    charts = []
    for item in args:
        if type(item) is str:
            charts[-1].title = item
        elif len(list(item)) > 2:
            charts.append(alt.Chart(item, height = 250, width = 250).mark_point(filled = True).encode(
                    alt.X("x:Q"), alt.Y("y:Q"), alt.Color("label:N")).interactive())
        else:
            charts.append(alt.Chart(item,height = 250, width = 250).mark_point(filled = True).encode(
                    alt.X('x:Q'),alt.Y('y:Q')).interactive())
    for i in range(len(charts)):
        if i==0: chart = charts[i]
        else: chart = chart | charts[i]
    chart.display()

def df_fromarr(*args, **kwargs):
    df = pd.DataFrame()
    for key, val in kwargs.items():
        df[key] = val
    return df

def get_data(pathto, *args):
    flag = 0
    for item in args:
        if flag == 0:
            data = np.loadtxt(pathto+item)
            flag = 1
        else:
            data = np.vstack((data, np.loadtxt(pathto+item)))
    return data

def mean_variance(df, **kwargs):
    name = kwargs['name'] if 'name' in kwargs else 'Input Model'
    zeromean = [round(i,2) for i in [stat.mean([x for x in df['x'][df['label']==0]]),\
                stat.mean([x for x in df['y'][df['label']==0]])]]
    onemean = [round(i,2) for i in [stat.mean([x for x in df['x'][df['label']==1]]),\
                stat.mean([x for x in df['y'][df['label']==1]])]]
    covone = np.cov([x for x in df['x'][df['label']==1]],[x for x in df['y'][df['label']==1]])
    covzero = np.cov([x for x in df['x'][df['label']==0]],[x for x in df['y'][df['label']==0]])
    print("Parameters for {}:\n".format(name),"="*25,"\n\
Mean of class 0 = {}, (co)-variance of class 0 = {}\n\
Mean of class 1 = {}, (co)-variance of class 1 = {}\n".format(zeromean,np.round(covone,2).tolist(),onemean,np.round(covzero,2).tolist()))

def two_dim_contour(model, data):
    max_x= max([x[0] for x in data])
    min_x = min([x[0] for x in data])
    max_x += max_x/8
    min_x -= min_x/8
    max_y= max([x[1] for x in data])
    min_y = min([x[1] for x in data])
    max_y += max_y/8
    min_y -= min_y/8
    x, y = np.meshgrid(np.linspace(min_x,max_x),np.linspace(min_y,max_y))
    XX = np.array([x.ravel(), y.ravel()]).T
    z = -model.score_samples(XX)
    z = z.reshape(x.shape)
    return x,y,z
