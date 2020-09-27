#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import plotly.express as px


# In[2]:


df= pd.read_csv('JPM.csv')


# In[3]:


# df.set_index('Date',inplace=True)


# In[4]:


df.head()


# In[5]:


fig = px.line(df, x=df['Date'], y=df['Adj Close'])
fig.show()


# In[6]:


df['Adj Close'].describe(percentiles=[0.05,0.1,0.25,.33,0.5,0.67,0.75,0.9,0.95,0.99])


# In[7]:


df['mean_stock_value']=df['Adj Close'].mean(0)


# In[8]:


df['Daily Return Log'] = np.log(df['Adj Close']/(df['Adj Close'].shift(1)-1))
fig = px.line(df, x=df['Date'], y=df['Daily Return Log'])
fig.show()


# In[9]:


df['Daily Return Percent']=df['Daily Return Log']*100
fig = px.line(df, x=df['Date'], y=df['Daily Return Percent'])
fig.show()


# In[10]:


df


# In[11]:


volatility=df['Daily Return Log'].std()*100


# In[12]:


volatility


# In[13]:


df['Volatility_2nd']= ((df['Adj Close']-df['mean_stock_value'])**2)/df.shape[0]


# In[14]:


fig = px.line(df, x=df['Date'], y=df['Volatility_2nd'])
fig.show()


# In[16]:


df[['Volatility_2nd']].describe()


# In[ ]:




