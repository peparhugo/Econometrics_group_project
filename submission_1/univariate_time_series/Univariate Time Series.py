#!/usr/bin/env python
# coding: utf-8

# # Univariate Time Series
# 
# ## Requirements
# 
# This notebook follows the outline provided in MScFE 610 Econ Group Work Projects:
# 
# With this data, do the following using R or Python languages: 
# 1. Forecast S&P/Case-Shiller U.S. National Home Price Index using an ARMA model. 
# 2. Implement the Augmented Dickey-Fuller Test for checking the existence of a unit root in Case-Shiller Index series. 
# 3. Implement an ARIMA(p,d,q) model. Determine p, d, q using Information Criterion or Box-Jenkins methodology. Comment the results.  
# 4. Forecast  the  future  evolution  of  Case-Shiller  Index  using  the  ARMA  model.  Test  model  using in-sample forecasts.
# 
# ## Data
# 
# Data source: [https://fred.stlouisfed.org/series/CSUSHPISA](https://fred.stlouisfed.org/series/CSUSHPISA)
# 
# Period considered in the analysis: January 1987 – latest data
# 
# Frequency: monthly data
# 

# ## Import Libraries

# In[1]:


import requests
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
plt.style.use('fivethirtyeight')
#library to provide seasonal decompositon of data
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
#library to run auto-correlation and partial autocorrelation plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#library for Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
#library for information criterion
import pmdarima as pm
#library for ARMIMA forecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA


# ## Get Data
# I am pulling the csv data directly through a request. I used the browser `Inspect` function on [this page](https://fred.stlouisfed.org/series/CSUSHPISA) to see what url request downloaded the csv data. I copied that link and parased out the request parameters below into a dictionary called `params`. The params are passed in with csv url using `requests.get()` to download in the csv data.

# In[42]:


params={'bgcolor': '%23e1e9f0',
 'chart_type': 'line',
 'drp': '0',
 'fo': 'open%20sans',
 'graph_bgcolor': '%23ffffff',
 'height': '450',
 'mode': 'fred',
 'recession_bars': 'on',
 'txtcolor': '%23444444',
 'ts': '12',
 'tts': '12',
 'width': '1168',
 'nt': '0',
 'thu': '0',
 'trc': '0',
 'show_legend': 'yes',
 'show_axis_titles': 'yes',
 'show_tooltip': 'yes',
 'id': 'CSUSHPISA',
 'scale': 'left',
 'cosd': '1987-01-01',
 'coed': '2020-06-01',
 'line_color': '%234572a7',
 'link_values': 'false',
 'line_style': 'solid',
 'mark_type': 'none',
 'mw': '3',
 'lw': '2',
 'ost': '-99999',
 'oet': '99999',
 'mma': '0',
 'fml': 'a',
 'fq': 'Monthly',
 'fam': 'avg',
 'fgst': 'lin',
 'fgsnd': '2020-02-01',
 'line_index': '1',
 'transformation': 'lin',
 'vintage_date': '2020-09-23',
 'revision_date': '2020-09-23',
 'nd': '1987-01-01'}
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
resp = requests.get(url=url,params=params)


# ## Read data into a pandas dataframe & visualize the data
# Pandas `read_csv` needs a string input output function `StringIO` from library `io` to read the csv data loaded into the response from csv request above. Usually the `read_csv` funtions takes a path to a csv file but I by passed save and reloading the data using the `StringIO` function.

# In[44]:


data= pd.read_csv(StringIO(resp.text),sep=',',parse_dates=['DATE'], index_col='DATE')
data.plot(figsize=(10,4))
plt.title('S&P/Case-Shiller U.S. National Home Price Index')
plt.show()


# In[45]:


data.describe()


# ## Run a seasonal decomposition to see trend, seaonal and residual components
# This section of the code passes the pandas dataframe `data` into an additive seasonal decomposition.
# 
# Notes:
# There is little to no seaonal component since it ranges between -0.05 to 0.05 relative to the range of actual values from 64 to 218. The residual also has a wave pattern, which is most likely negatively correlated with seasonal component. I will take the decomposition output and run a correlation matrix to see if the seasonal component and the residual component are correlated.

# In[53]:


decomp = seasonal_decompose(data, model='additive')
decomp.plot()
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 100 
plt.show()


# ## Correlation of seaonal
# This portion of the notebook will see if 

# In[73]:


from sklearn.preprocessing import MinMaxScaler


# In[82]:


#all data
seasonal_data = decomp.seasonal.dropna()
resid_data = decomp.resid.dropna()
seasonal_resid_data=pd.DataFrame(MinMaxScaler().fit_transform(pd.concat([seasonal_data,resid_data],axis=1).dropna()),
                                columns=['seasonal','resid'])
#this is inefficient but it was easiest method to carry the index forward once I added the min max scaler
seasonal_resid_data.index = pd.concat([seasonal_data,resid_data],axis=1).dropna().index
seasonal_resid_data.corr()


# In[86]:





# In[87]:


#2016 to now
seasonal_resid_data.loc['2016-01-01':'2020-06-01'].corr()


# In[88]:


#pre-internet
pd.concat([decomp.seasonal.dropna(),decomp.resid.dropna()],axis=1).dropna().loc['1987-01-01':'1998-12-01'].corr()


# ## Run an acf and pacf plot
# The code below runs a acf and pacf.
# 
# Results:
# The acf decays overtime following a linear trend and the pacf spikes through 2 lags. This means this is AR(2).
# The ARMA model will be (2,0,0).
# 
# This assumes the data is stationary.

# In[138]:


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
plot_acf(data, lags=50, ax=ax1)
plot_pacf(data, lags=50, ax=ax2)
plt.show()


# # ARMA Forecast
# The initial ARMA forecast using (2,0,0).
# 
# Results:
# 
# The forecast does not look right. I will try an Augmented Dickey-Fuller Test to see if the data is stationary. The null hypthesis of the Augmented Dickey-Fuller Test is the data is not stationary.

# In[ ]:


model_1 = ARIMA(data, order=(2,0,0),freq='MS')
model_fit = model_1.fit()
output_1 = model_fit.forecast(steps=100,freq='MS')


# In[134]:


forecast_1 = data.copy()
forecast_1 = pd.concat([forecast_1,output_1],axis=1)
forecast_1.columns = [forecast_1.columns[0], 'Forecast']
forecast_1.plot(figsize=(10,4))
plt.title('S&P/Case-Shiller U.S. National Home Price Index')
plt.show()


# ## Augmented Dickey-Fuller Test 
# 
# Results:
# 
# **Test 1**
# 
# We failed to reject the null hypthesis because p=0.903413>0.05. This means the data is not stationary.
# 
# **Test 2**
# 
# We rejected the null hypthesis because p=0.049176>0.05. This means the data is stationary at p<0.05 but it is not stationary at 0.01.

# In[135]:


#Test1
dftest = adfuller(data)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# In[136]:


#Test2
dftest = adfuller(data.diff().dropna())
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# ## ARIMA Forecast
# Steps:
# 
# 1. Plot acf and pacf for differencing 1.
# 2. Examine plots to see if AR, MA or AR & MA for p and q values.
# 3. Create ARIMA model with (p,d,q).
# 4. Forecast values.
# 5. Plot Actuals and Forecast.
# 6. Grid search for p and q using AIC minimzing objective.
# 7. Forecast and plot AIC criterion.

# In[139]:


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
plot_acf(data.diff().dropna(), lags=50, ax=ax1)
plot_pacf(data.diff().dropna(), lags=50, ax=ax2)
plt.show()


# In[143]:


model_2 = ARIMA(data, order=(2,1,0),freq='MS')
model_fit2 = model_2.fit()
output_2 = model_fit2.forecast(steps=100,freq='MS')
forecast_2 = pd.concat([forecast_1,output_2],axis=1)
forecast_2.columns = [forecast_1.columns[0], 'Forecast (2,0,0)','Forecast (2,1,0)']
forecast_2.plot(figsize=(10,4))
plt.title('S&P/Case-Shiller U.S. National Home Price Index')
plt.show()


# In[140]:


model = pm.auto_arima(data, d=1, D=1,
                       trend='c', seasonal=False, 
                      start_p=0, start_q=0, max_order=12, test='adf', max_p=12,
                      error_action='ignore',  # don't want to know if an order does not work
                    suppress_warnings=True,
                      stepwise=True, trace=True)


# In[146]:


model_3 = ARIMA(data, order=(2,1,0),freq='MS')
model_fit3 = model_3.fit()
output_3 = model_fit3.forecast(steps=100,freq='MS')
forecast_3 = pd.concat([forecast_2,output_3],axis=1)
forecast_3.columns = [forecast_1.columns[0], 'Forecast (2,0,0)','Forecast (2,1,0)','Forecast (2,1,1)']
forecast_3.plot(figsize=(10,4))
plt.title('S&P/Case-Shiller U.S. National Home Price Index')
plt.show()


# ## Forecast In-sample
# This forecast will take 80% of the past data and 20% of the data back from 2020-06-01 to use a future data to compare 1 month incremental predictions.
# 
# eg. At time period June 01, 2016 will use 1987-01-01 to 2016-06-01 as data for an ARMA model and then predict the `CSUSHPISA` value for 2016-07-01, 1 month a head. Then the actual value will be appended to the data for an ARMA model so the historical data will be 1987-01-01 to 2016-07-01 and the next prediction will be for 2016-08-01.

# In[216]:


#set size of historical data and size of "future" data
size = int(data.shape[0] * 0.80)
#split data based on size
train, test = data[0:size], data[size:data.shape[0]]
#extract list of values
history = [x[1].CSUSHPISA for x in train.iterrows()]
#create empty list for storing incremental predictions
predictions = list()
#loop through test data for incremental predictions
for t in range(len(test)):
    #create ARMA model using current history
    model = ARIMA(history, order=(2,0,0))
    #fit model
    model_fit = model.fit()
    #create forecast
    output = model_fit.forecast(steps=1)
    #append next month forecast
    predictions.append(output[0])
    #append this month to history to simulate time moving forward
    history.append(test.iloc[t].values[0])


# In[217]:


ppd_pred = test.copy()


# In[218]:


ppd_pred['Predictions (2,0,0)']=predictions


# In[219]:


ppd_pred


# In[220]:


ppd_pred['resid']=ppd_pred['Predictions (2,0,0)']-ppd_pred['CSUSHPISA']


# In[222]:


ppd_pred[['Predictions (2,0,0)','CSUSHPISA']].plot()
plt.title('S&P/Case-Shiller U.S. National Home Price Index')
plt.show()


# In[226]:


plt.plot(ppd_pred[['resid']])
plt.title('Residuals')
plt.show()


# In[224]:



ppd_pred['resid_perc']=ppd_pred['resid']/ppd_pred['CSUSHPISA']
ax = ppd_pred['resid_perc'].plot()
type(ax)  # matplotlib.axes._subplots.AxesSubplot

# manipulate
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
plt.title('Percentage Residuals')
plt.show()


# In[211]:


ppd_pred['resid_perc'].describe()


# In[ ]:




