#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np


# In[2]:


#load csv files into dataframes
data_frames = {}
for file in os.listdir():
    if file.find('.csv')!=-1:
        try:
            data_frames[file] = pd.read_csv(file)
        except:
            print(file)


# In[3]:


#load CAD/USD exchange rate
exchange_rate = data_frames['AEXCAUS.csv']
exchange_rate['DATE']=exchange_rate['DATE'].astype(str).str[0:4].astype(int)
exchange_rate = exchange_rate.set_index('DATE')


# In[4]:


#test unit root of CAD/USD exchange
from statsmodels.tsa.stattools import adfuller

#Test1
dftest_1 = adfuller(exchange_rate['AEXCAUS'].diff().dropna())
dftest_1_output = pd.Series(dftest_1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest_1[4].items():
    dftest_1_output['Critical Value (%s)'%key] = value
print(dftest_1_output)


# In[5]:


#get cpi data
cpi_ir = data_frames['DP_LIVE_10102020213754370.csv'][data_frames['DP_LIVE_10102020213754370.csv'].LOCATION!='OECD'].pivot(index='TIME',columns='LOCATION',values='Value').dropna()


# In[6]:


cpi_ir['CPI_IR']=cpi_ir['CAN']-cpi_ir['USA']


# In[7]:


#test unit root  of consumer price index 
dftest_1 = adfuller(cpi_ir['CPI_IR'].diff().dropna())
dftest_1_output = pd.Series(dftest_1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest_1[4].items():
    dftest_1_output['Critical Value (%s)'%key] = value
print(dftest_1_output)


# In[8]:


#get real itnerest rate data
ri_ir = data_frames['DP_LIVE_10102020214019137.csv'][data_frames['DP_LIVE_10102020214019137.csv'].LOCATION!='OECD'].pivot(index='TIME',columns='LOCATION',values='Value').dropna()
ri_ir['RI_IR']=ri_ir['CAN']-ri_ir['USA']


# In[9]:


#test unit root of real interest rate
dftest_1 = adfuller(ri_ir['RI_IR'].diff().dropna())
dftest_1_output = pd.Series(dftest_1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest_1[4].items():
    dftest_1_output['Critical Value (%s)'%key] = value
print(dftest_1_output)


# In[10]:


#get terms of trade data
tot = data_frames['DP_LIVE_10102020213123395.csv'][data_frames['DP_LIVE_10102020213123395.csv'].LOCATION!='OECD'].pivot(index='TIME',columns='LOCATION',values='Value').dropna()
tot['tot']=tot['CAN']


# In[11]:


#test unit root of difference of terms of trade CAN
dftest_1 = adfuller(tot['tot'].diff().dropna())
dftest_1_output = pd.Series(dftest_1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest_1[4].items():
    dftest_1_output['Critical Value (%s)'%key] = value
print(dftest_1_output)


# In[12]:


# difference between CAD and USA federal debt as percentage of gdp
debt = pd.concat([data_frames['GGGDTACAA188N.csv'].set_index('DATE'),
                  data_frames['DEBTTLUSA188A.csv'].set_index('DATE')],axis=1).dropna()
debt['rel_debt']=debt['GGGDTACAA188N']-debt['DEBTTLUSA188A']
debt = debt.reset_index()
debt['index']=debt['index'].astype(str).str[:4].astype(int)
debt=debt.set_index('index')
#Test1
dftest_1 = adfuller(debt['rel_debt'].diff().dropna())
dftest_1_output = pd.Series(dftest_1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest_1[4].items():
    dftest_1_output['Critical Value (%s)'%key] = value
print(dftest_1_output)


# In[13]:


#oil data
oil = data_frames['DCOILWTICO.csv']
oil['DATE']=oil['DATE'].astype(str).str[:4].astype(int)
oil = oil.set_index('DATE')
#Test1
dftest_1 = adfuller(np.log(oil['DCOILWTICO']).diff().dropna())
dftest_1_output = pd.Series(dftest_1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest_1[4].items():
    dftest_1_output['Critical Value (%s)'%key] = value
print(dftest_1_output)


# In[28]:


#merge data on year/date/time index
data = pd.concat([exchange_rate['AEXCAUS'],
           np.log(oil['DCOILWTICO']),
           tot['tot'],
           cpi_ir['CPI_IR'],
           debt['rel_debt'],
           ri_ir['RI_IR']
          ],
          axis=1).dropna()


# In[29]:


from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.vector_ar.var_model import VAR
#trace rank
#k_ar_diff set to 1 since all variables are unit root at diff = 1
vec_rank = select_coint_rank(data, det_order = 1, k_ar_diff = 1, method = 'trace', signif=0.05)
print(vec_rank.summary())


# In[30]:


#eigen rank
#k_ar_diff set to 1 since all variables are unit root at diff = 1
vec_rank2 = select_coint_rank(data, det_order = 1, k_ar_diff = 1, method = 'maxeig', signif=0.05)
print(vec_rank2.summary())


# In[33]:


#fit model and print predictions (i have no idea what predictions is returning, future values, in-sample values?)
#k_ar_diff set to 1 since all variables are unit root at diff = 1
#coint_rank - i have set at one but this may be incorrect
vecm = VECM(endog = data, k_ar_diff = 1, coint_rank =1, deterministic = 'cili')
vecm_fit = vecm.fit()
vecm_fit.predict()


# In[18]:


#we may not need this
model = VAR(endog=data.diff().dropna())
res = model.select_order(3)
res.summary()


# In[19]:


data


# In[20]:




#export to other file formats
get_ipython().system('jupyter nbconvert --to html "multivariate_analysis.ipynb"')
get_ipython().system('jupyter nbconvert --to python "multivariate_analysis.ipynb"')


# In[ ]:




