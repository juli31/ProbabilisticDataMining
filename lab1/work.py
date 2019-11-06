import numpy as np
import pandas as pd
import altair as alt
import util
from sklearn import mixture
import os

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
    '''1. Preliminary Questions'''
    scatter_from_df(util.prep_work_GMM())
    A_data = util.get_data("./Unistroke/",[f for f in os.listdir("./Unistroke/") if ((f[0] == 'A') and (f[1] != 'm'))])
    scatter_from_df(util.df_fromarr(['x','y'],[x[0] for x in A_data],[x[1] for x in A_data]))

    '''2. Data Analysis, Gaussian Models'''
    scatter_from_df(util.df_fromarr(['x','y'],[x[0] for x in A_data],[x[1] for x in A_data]),\
                    util.df_fromarr(['x','y','class'],[x[0] for x in A_data],[x[1] for x in A_data],\
                    list(mixture.GaussianMixture(n_components = 2).fit_predict(A_data))))
    Amerge_data = util.get_data("./Unistroke/", ["Amerge.txt",])
    scatter_from_df(util.df_fromarr(['x','y'],[x[0] for x in Amerge_data],[x[1] for x in Amerge_data]),\
                    util.df_fromarr(['x','y','class'],[x[0] for x in Amerge_data],[x[1] for x in Amerge_data],\
                    list(mixture.GaussianMixture(n_components = 2).fit_predict(Amerge_data))))


if __name__=='__main__':
    main()
