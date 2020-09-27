#!/usr/bin/env python
# coding: utf-8

# In[21]:


# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:59:57 2020

@author: Hamid
"""

from sklearn import linear_model
import yfinance as yf

# Download stock data for JP Morgan and the S&P 500
jpm=yf.download("JPM", start="2018-02-01", end="2018-12-30")
spx=yf.download("^GSPC", start="2018-02-01", end="2018-12-30")

# 
X = spx.iloc[:, spx.columns == 'Adj Close']
Y = jpm['Adj Close']

# Perform Linear Regression on the data
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Get values for Linear Regression Equation
print('slope:', regr.coef_)
print('intercept:', regr.intercept_)


# In[ ]:




