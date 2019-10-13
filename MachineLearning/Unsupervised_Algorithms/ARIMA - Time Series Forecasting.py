#!/usr/bin/env python
# coding: utf-8

# ## ARIMA - Time Series Forecasting for Sales Data

# * Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values.

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


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# ** Step 1: Read the Dataset using Pandas **

# In[3]:


df = pd.read_excel("E:\RDataSet\Superstore.xls")


# In[4]:


furniture = df.loc[df['Category'] == 'Furniture']


# In[5]:


furniture['Order Date'].min(), furniture['Order Date'].max()


# ** Step2: Data Preprocessing **

# In[6]:


furniture.columns


# In[7]:


furniture.info()


# In[8]:


sns.heatmap(furniture.isnull(),yticklabels=False,cbar=False,cmap='cool')
plt.show()


# In[9]:


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()


# In[10]:


furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()


# ** Step 3: Indexing with Time Series Data **

# In[11]:


furniture = furniture.set_index('Order Date')
furniture.index


# In[12]:


y = furniture['Sales'].resample('MS').mean()


# In[13]:


y['2017':]


# In[14]:


sns.set_style('darkgrid')
plt.figure(figsize=(11, 5))
plt.plot(y, c='magenta')
plt.title("Time Series Plotting")
plt.xlabel("Date")
plt.ylabel("Furniture Sales")
plt.show()


# ### Observations:
# 
# * Some distinguishable patterns appear when we plot the data. The time-series has seasonality pattern, such as sales are always low at the beginning of the year and high at the end of the year. There is always an upward trend within any single year with a couple of low months in the mid of the year.

# ** Step 4: Time series forecasting with ARIMA **

# ARIMA models are denoted with the notation ** ARIMA(p, d, q) **. These three parameters account for seasonality, trend, and noise in data.

# In[15]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[16]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# ** The above output suggests that SARIMAX(1, 1, 1)x(1, 1, 0, 12) yields the lowest AIC value of 297.78. Therefore we should consider this to be optimal option. **
# 
# AIC - Akaike Information Critera
# - Goodness of model fit
# - Simplicity of model in linear statistics

# In[17]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1]) 


# In[18]:


results.plot_diagnostics(figsize=(16, 8))
plt.show() 


# ## Validating forecasts

# ** To help us understand the accuracy of our forecasts, we compare predicted sales to real sales of the time series, and we set forecasts to start at 2017–01–01 to the end of the data. **

# In[19]:


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), 
                              dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed', c='magenta')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', 
                         c= 'k',alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.title("Time Series Plotting")
plt.xlabel('Date')
plt.ylabel('Furniture Sales')
plt.legend()
plt.show()


# ** MSE **

# In[20]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# ** RMSE **

# In[21]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round
                                                                  (np.sqrt(mse), 2)))


# ** Producing and visualizing forecasts **

# In[22]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7) , c = 'magenta')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', c = 'k' )
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


# ## Time Series of Furniture vs. Office Supplies

# ** According to our data, there were way more number of sales from Office Supplies than from Furniture over the years. **

# In[23]:


furniture = df.loc[df['Category'] == 'Furniture']
office = df.loc[df['Category'] == 'Office Supplies']
furniture.shape, office.shape


# In[24]:


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
office.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
office = office.sort_values('Order Date')
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
office = office.set_index('Order Date')
y_furniture = furniture['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()
furniture = pd.DataFrame({'Order Date':y_furniture.index, 'Sales':y_furniture.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})
store = furniture.merge(office, how='inner', on='Order Date')
store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)
store.head()


# In[30]:


plt.figure(figsize=(20, 8))
plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label = 'furniture')
plt.plot(store['Order Date'], store['office_sales'], 'k-', label = 'office supplies')
plt.xlabel('Date') 
plt.ylabel('Sales') 
plt.title('Sales of Furniture and Office Supplies')
plt.legend()
plt.show()


# In[31]:


first_date = store.ix[np.min(list(np.where(store['office_sales'] > store['furniture_sales'])[0])), 'Order Date']
print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))


# In[ ]:




