from sklearn import mixture
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import os
import util

def main():
    A = util.get_data("./Unistroke/",*[f for f in os.listdir("./Unistroke/") if ((f[0] == 'A') and (f[1] != 'm'))])
    flag = 0
    for item in list(A):
        if item[0] < 0:
            continue
        elif flag == 0:
            new = np.array(item)
            flag = 1
        else:
            new = np.vstack((new, item))
    A = new

    A_GMM = mixture.GaussianMixture(n_components = 2)
    A_labeled_df = util.df_fromarr(x = [x[0] for x in A],\
                                    y = [x[1] for x in A],label= A_GMM.fit_predict(A))
    Amerge = util.get_data("./Unistroke/", "Amerge.txt")
    Amerge_GMM = mixture.GaussianMixture(n_components = 2)
    Amerge_labeled_df = util.df_fromarr(x = [x[0] for x in Amerge],\
                                        y = [x[1] for x in Amerge],\
                                        label= Amerge_GMM.fit_predict(Amerge))

    '''1. Preliminary Questions'''
    # util.plot((util.prep_work_GMM()))
    # util.plot((util.df_fromarr(x = [x[0] for x in A],\
    #                                 y = [x[1] for x in A])))
    # '''2. Data Analysis, Gaussian Models'''
    util.mean_variance(A_labeled_df)


if __name__=='__main__':
    main()
