import numpy as np
import pandas as pd
import altair as alt
import util
from sklearn import mixture

def plot_from_df(df):
    if len(list(df))>2:
        chart = alt.Chart(df, height = 500, width = 500).mark_point().encode(
                alt.X("x:Q"),
                alt.Y("y:Q"),
                alt.Color("class:N")).interactive()
    else:
        chart = alt.Chart(df,height = 500, width = 500).mark_point().encode(
                alt.X('x:Q'),
                alt.Y('y:Q')).interactive()
    chart.serve()

def main():
    #plot_from_df(util.prep_work_GMM())
    #plot_from_df(util.plot_letter_data('A'))
    data = util.get_letter_data('A')
    data = util.get_letter_data('A',amerge = True)
    #for item in data:
    #    print(item)
    labs = util.fitpredict_gmm(data)
    xs = []
    ys = []
    for item in data:
        xs.append(float(item[0]))
        ys.append(float(item[1]))
    df = util.df_fromarr(['x','y','label'],xs,ys,list(labs))
    plot_from_df(util.df_fromarr(['x','y','class'],xs,ys,list(labs)))

if __name__=='__main__':
    main()
