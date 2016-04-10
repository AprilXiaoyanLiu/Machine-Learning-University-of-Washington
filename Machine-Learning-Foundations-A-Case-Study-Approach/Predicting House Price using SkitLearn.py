
# coding: utf-8

# In[16]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# In[18]:

sales = pd.read_csv('home_data.csv')


# In[19]:

sales.head()


# In[21]:

sales_X = sales['sqft_living']


# In[48]:

x = []
for i in sales_X:
    x.append(i)
xarray = np.array(x)
xarray


# In[68]:

sales_X_train = xarray[:-20]
sales_X_train.reshape(len(sales_X_train),1)


# In[53]:

sales_X_test = xarray[-20:]


# In[54]:

sales_X_test


# In[33]:

sales_Y = sales['price']


# In[55]:

y = []
for m in sales_Y:
    y.append(m)
yarray = np.array(y)
yarray


# In[58]:

sales_y_train = yarray[:-20]
sales_y_test = yarray[-20:]


# In[66]:

sales_y_train.reshape(len(sales_y_train),1)


# In[59]:

regr = linear_model.LinearRegression()


# In[70]:

regr.fit(sales_X_train.reshape(len(sales_X_train),1), sales_y_train.reshape(len(sales_y_train),1))


# In[71]:

print("Coefficients: \n", regr.coef_)


# In[ ]:

print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(sales_X_test.reshape(len(sales_X_test),1)) - sales_y_test.reshape(len(sales_y_test),1)) ** 2))


# In[ ]:

print('Variance score: %.2f' % regr.score(sales_X_test.reshape(len(sales_X_test),1), sales_y_test.reshape(len(sales_y_test),1)))


# In[ ]:

plt.scatter(sales_X_test.reshape(len(sales_X_test),1), sales_y_test.reshape(len(sales_y_test),1),  color='black')
plt.plot(sales_X_test.reshape(len(sales_X_test),1), sales_y_test.reshape(len(sales_y_test),1), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:



