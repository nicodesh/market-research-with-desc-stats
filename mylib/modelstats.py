import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as st

#from descstats import MyPlot, Univa

import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")


###############################################################
# Linear Regression Class
###############################################################

class LinReg():
    """ A class to realize linear regressions. """

    def __init__(x, y):
        """ Class constructor.
        Args:
            x (Pandas Series): The first variable
            y (Pandas Series): The unique feature
        """

        self.x = x
        self.X = x[:, np.newaxis]
        self.y = y

        self.sklearn_lr = LinearRegression()
        self.sklearn_lr = self.sklearn_lr.fit(X, y)

        self.y_pred = self.sklearn_lr.predict(X)
        r2 = r2_score(self.y, self.y_pred)

        self.residuals = self.y - self.y_pred

        self.st_slope, self.st_intercept, self.st_r_value, self.st_p_value, self.st_std_err = st.linregress(x,y)

    def plot():
        """ Plot the scatterplot and the linear regression. """

        fig, ax = plt.subplots(figsize=[7,5])        
        MyPlot.scatter(ax, self.x, self.y)
        ax.plot(x, y_pred, linewidth=1, color="#fcc500")
        MyPlot.bg(ax)
        MyPlot.title(ax, "Scatterplot + Linear regression")
        MyPlot.border(ax)
        plt.show()

    def residuals_distribution():
        """ Plot the distribution of the residuals. """

        univ = Univa(self.residuals)
        univ.describe()
        univ.distribution(bins=9)