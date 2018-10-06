import numpy as np
import pandas as pd
import scipy.stats as stats
#import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression


boston = load_boston()

'''print boston.keys()
print boston.data.shape
print boston.DESCR
print boston.data.head
print boston.feature_names'''

bos = pd.DataFrame(boston.data)

bos.columns = boston.feature_names

bos['PRICE'] = boston.target


X = bos.drop('PRICE', axis = 1)

lm = LinearRegression()
lm.fit(X, bos.PRICE)
print lm.intercept_
print lm.coef_

