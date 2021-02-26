#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[54]:


ILPD= pd.read_csv('ILPD.csv', names=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphatase', 
'Alanine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin', 'Albumin_and_Globulin_Ratio',
'Selector']) 
ILPD


# # Preprocessing

#  Dropping the "Gender" column to only retain numerical values

# In[55]:


data=ILPD.drop(['Gender'], axis=1)
data


# In[56]:


data.isna().sum()


# In[57]:


data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(0)
data.isna().sum()


# In[58]:


ILPD


# # Creating labels

# In[59]:


ILPD['Gender'].value_counts()


# In[60]:


ILPD['Gender'].replace(['Male','Female'],['0','1'],inplace=True)
ILPD['Gender'].value_counts()
ILPD


# ### Test-train split

# In[61]:


x_tr = data.iloc[0:460, :]
x_te = data.iloc[460: , :]

x_train=np.asarray(x_tr).astype('float32')
x_test=np.asarray(x_te).astype('float32')

x_trn = ILPD.iloc[0:460, :]#Using the modified dataset with '1' and '0' replacing gender values of female and male to create labels
x_tes = ILPD.iloc[460: , :]

#y_train=np.array(x_trn['Gender'])
#y_test=np.array(x_tes['Gender'])
y_train=np.array(x_trn['Selector'])
y_test=np.array(x_tes['Selector'])

print(y_train)
plt.scatter(x_train[:,0],y_train, c=y_train, cmap='rainbow')
plt.show()


# In[62]:


LR=LogisticRegression(solver='liblinear')
LR.fit(x_train,y_train)


# In[73]:


y_pred=LR.predict(x_test)
plt.scatter(x_test[:,0],x_test[:,9],c=y_pred)
plt.colorbar()
plt.show()


# # References

# 1. https://www.youtube.com/watch?v=VK6v9Ure8Lk
# 2. https://www.youtube.com/watch?v=HYcXgN9HaTM
# 3. https://www.kaggle.com/randyrose2017/for-beginners-using-keras-to-build-models

# In[ ]:




