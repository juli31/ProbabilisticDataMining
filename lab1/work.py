import numpy as np
import pandas as pd
import altair as alt
from os import listdir

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

def df_fromarr(names, *args):
    df = pd.DataFrame()
    for i in range(len(args)):
        df[names[i]] = args[i]
    return df

def get_A_data():
    xs = []
    ys = []
    '''
    with open("./Unistroke/Amerge.txt", 'r') as f:
        for line in f.readlines():
            data = line.split(' ')
            for i in range(len(data)):
                data[i] = float(data[i])
            xs.append(data[0])
            ys.append(data[1])

    ''' 
    files = [f for f in listdir("./Unistroke/") if ((f[0] == 'A') and (f[1] != 'm'))]
    for item in files:
        print(item)
        with open("./Unistroke/"+item,'r') as f:
            for line in f.readlines():
                data = line.split('\t')
                for i in range(len(data)):
                    data[i] = float(data[i])
                xs.append(data[0])
                ys.append(data[1])
    df = df_fromarr(['x','y'],xs,ys)
    print(df)
    plot_from_df(df)

def prep_GMM():
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
    
    df = df_fromarr(['x','y','class'], xs,ys,labs)
    
    plot_from_df(df)

def main():
    #prep_GMM()
    get_A_data()

if __name__=='__main__':
    main()
