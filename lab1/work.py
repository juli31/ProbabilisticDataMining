from sklearn import mixture
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import os
import util

def scatter_from_df(*args):
    charts = []
    for item in args:
        if len(list(item))>2:
            charts.append(alt.Chart(item, height = 500, width = 500).mark_point(filled = True).encode(
                    alt.X("x:Q"), alt.Y("y:Q"), alt.Color("class:N")).interactive())
        else:
            charts.append(alt.Chart(item,height = 500, width = 500).mark_point(filled = True).encode(
                    alt.X('x:Q'),alt.Y('y:Q')).interactive())
    for i in range(len(charts)):
        if i==0: chart = charts[i]
        else: chart = chart | charts[i]
    chart = chart.configure_legend(clipHeight = 10, titleFontSize= 17, \
                            labelFontSize = 17, symbolSize = 50, \
                            symbolStrokeWidth = 3)\
            .configure_axis(labelFontSize=17, titleFontSize=17)
    chart.serve()

def main():
    '''
    #1. Preliminary Questions
    scatter_from_df(util.prep_work_GMM())
    A_data = util.get_data("./Unistroke/",[f for f in os.listdir("./Unistroke/") if ((f[0] == 'A') and (f[1] != 'm'))])
    scatter_from_df(util.df_fromarr(['x','y'],[x[0] for x in A_data],[x[1] for x in A_data]))

    #2. Data Analysis, Gaussian Models
    scatter_from_df(util.df_fromarr(['x','y'],[x[0] for x in A_data],[x[1] for x in A_data]),\
                    util.df_fromarr(['x','y','class'],[x[0] for x in A_data],[x[1] for x in A_data],\
                    list(mixture.GaussianMixture(n_components = 2).fit_predict(A_data))))
    '''
    Amerge_data = util.get_data("./Unistroke/", ["Amerge.txt",])
    '''
    scatter_from_df(util.df_fromarr(['x','y'],[x[0] for x in Amerge_data],[x[1] for x in Amerge_data]),\
                    util.df_fromarr(['x','y','class'],[x[0] for x in Amerge_data],[x[1] for x in Amerge_data],\
                    list(mixture.GaussianMixture(n_components = 2).fit_predict(Amerge_data))))
    '''
    Amerge_df =  util.df_fromarr(['x','y','class'],[x[0] for x in Amerge_data],[x[1] for x in Amerge_data],\
                                    list(mixture.GaussianMixture(n_components = 2).fit_predict(Amerge_data)))
    '''
    ones = [[x,y] for x,y,z in zip(Amerge_df['x'],Amerge_df['y'],Amerge_df['class']) if z==1]
    zeros = [[x,y] for x,y,z in zip(Amerge_df['x'],Amerge_df['y'],Amerge_df['class']) if z!=1]
    onemean = [sum([x[0] for x in ones])/len(ones), sum([x[1] for x in ones])/len(ones)]
    zeromean = [sum([x[0] for x in zeros])/len(zeros), sum([x[1] for x in zeros])/len(zeros)]
    covone = np.cov(np.array(ones).T)
    covzero = np.cov(np.array(zeros).T)
    print("Parameters for the bivariate GMM on the letter A data:\n","="*25,"\n\
Mean of class 0 = {}\n\
Mean of class 1 = {}\n\
Covariance of class 0 = {}\n\
Covariance of class 1 = {}".format(zeromean,onemean,covone,covzero))
    '''
    GMM = mixture.GaussianMixture(n_components = 2)
    GMM.fit_predict(Amerge_data)
    util.plot_pdf(GMM, 0, 0, 0)


if __name__=='__main__':
    main()
