#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import plotly.express as px


# In[2]:


#read in data
df= pd.read_csv('JPM.csv')


# In[4]:


#view first 5 rows of JPM data
df.head()


# In[5]:


#plot the adjusted close field overtime
fig = px.line(df, x=df['Date'], y=df['Adj Close'])
fig.show()


# In[6]:


#output summary of data distribution
df['Adj Close'].describe(percentiles=[0.05,0.1,0.25,.33,0.5,0.67,0.75,0.9,0.95,0.99])


# In[7]:


#calcalate mean stock value
df['mean_stock_value']=df['Adj Close'].mean(0)


# In[8]:


#calculate daily stock return using log returns and plot daily return data
df['Daily Return Log'] = np.log(df['Adj Close']/(df['Adj Close'].shift(1)-1))
fig = px.line(df, x=df['Date'], y=df['Daily Return Log'])
fig.show()


# In[9]:


#calculate and plot the daily return percentage
df['Daily Return Percent']=df['Daily Return Log']*100
fig = px.line(df, x=df['Date'], y=df['Daily Return Percent'])
fig.show()


# In[10]:


#display first 5 and last 5 rows of data
df


# In[11]:


#calculate volatility
volatility=df['Daily Return Log'].std()*100


# In[12]:


#output volatility
volatility


# In[13]:


df['Volatility_2nd']= ((df['Adj Close']-df['mean_stock_value'])**2)/df.shape[0]


# In[14]:


fig = px.line(df, x=df['Date'], y=df['Volatility_2nd'])
fig.show()


# In[16]:


df[['Volatility_2nd']].describe()


# In[17]:


#export to other file formats
get_ipython().system('jupyter nbconvert --to html "basic_stats.ipynb"')
get_ipython().system('jupyter nbconvert --to python "basic_stats.ipynb"')


# In[ ]:




