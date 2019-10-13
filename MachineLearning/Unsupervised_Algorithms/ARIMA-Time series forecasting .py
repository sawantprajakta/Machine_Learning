#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


# In[2]:


#matplotlib.rcParams['axes.labelsize']=14
#matplotlib.rcParams['axes.labelsize']=14
#matplotlib.rcParams['axes.labelsize']=14
#matplotlib.rcParams['axes.labelsize']=14


# In[3]:


df=pd.read_excel('/home/dell/Documents/datasets/Superstore.xls')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='cool')


# In[7]:


furniture=df.loc[df['Category']=='Furniture']


# In[8]:


furniture['Order Date'].min(),furniture['Order Date'].max()


# In[9]:


furniture.columns


# In[10]:


cols=['Row ID', 'Order ID','Ship Date', 'Ship Mode','Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State','Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category','Product Name','Quantity', 'Discount', 'Profit']
furniture.drop(cols,axis=1,inplace=True)
furniture=furniture.sort_values('Order Date')
#furniture.isnull.sum()


# In[11]:


furniture['Sales']
furniture['Order Date']


# In[12]:


furniture=furniture.groupby('Order Date')['Sales'].sum().reset_index()


# In[13]:


furniture


# In[14]:


furniture=furniture.set_index('Order Date')
furniture.index


# In[15]:


y=furniture['Sales'].resample('MS').mean()
y


# In[16]:


plt.figure(figsize=(11,5))
plt.plot(y,c='magenta')
plt.title("Time series plotting")
plt.xlabel("Date")
plt.ylabel("Furniture Sales")
plt.show()


# In[17]:


#Arima=(p,q,d)seasonality,trend,noise
p=d=q=range(0,2)
pdq=list(itertools.product(p,d,q))
seasonal_pdq=[(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]

seasonal_pdq
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1],seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1],seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2],seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2],seasonal_pdq[4]))


# In[18]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod=sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results=mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except:
            continue
    


# In[19]:


#the above output suggests that SARIMAX(1,1,1)x(1,1,0,12) yields the lowest AIC value of 297.78.therefore we should consider this to be optimal option.
#AIC is Akaike Information Criteria
#- goodness of model fit
#- simplicity of model in linear star


# In[20]:


mod=sm.tsa.statespace.SARIMAX(y,order=(1,1,1),seasonal_order=(1,1,0,12),enforce_stationaryity=False,enforce_invertibility=False)
results=mod.fit()
print(results.summary().tables[1])


# In[21]:


results.plot_diagnostics(figsize=(16,8))
plt.show()


# In[22]:


#validating forecasts
pred=results.get_prediction(start=pd.to_datetime('2017-01-01'),dynamic=False)
pred_ci=pred.conf_int()
ax=y['2014':].plot(label='observed',c='magenta')
pred.predicted_mean.plot(ax=ax,label='One-step ahead forecast',c='k',alpha=.7,figsize=(14,7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='k',alpha=.2)
plt.title("Time series Plotting")
plt.xlabel('Date')
plt.ylabel('Furniture Sales')
plt.legend()


# In[23]:


#MSE

y_forecasted=pred.predicted_mean
y_truth=y['2017-01-01':]
mse=((y_forecasted-y_truth)**2).mean()
print('The mean squared error of our forecasts is {}'.format(round(mse,2)))


# In[24]:


#RMSE
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse),2)))


# In[ ]:


#producing and visualizing forecasts
#pred_uc=results.get_forecast(steps=100)
#pred_ci=pred_uc.conf_int()
#ax=y.plot(label='observed',figsize=(14,7),c='magenta')


# In[ ]:




