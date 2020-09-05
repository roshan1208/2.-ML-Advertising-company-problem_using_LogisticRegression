#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


ad = pd.read_csv('advertising.csv')


# In[14]:


ad


# In[15]:


ad['Ad Topic Line'].nunique()


# In[16]:


ad['City'].nunique()


# In[17]:


ad['Country'].nunique()


# In[21]:


sns.set_palette('GnBu_r')
sns.set_style('whitegrid')
sns.countplot(ad['Age'])
plt.tight_layout()


# In[40]:


type(ad['Timestamp'].iloc[0])


# In[41]:


ad['Timestamp'] = pd.to_datetime(ad['Timestamp'])
ad['Hour'] = ad['Timestamp'].apply(lambda time: time.hour)
ad['Month'] = ad['Timestamp'].apply(lambda time: time.month)
ad['Day of Week'] = ad['Timestamp'].apply(lambda time: time.dayofweek)


# In[43]:


ad.drop('Year',axis=1,inplace=True)


# In[44]:


ad.head()


# In[45]:


ad.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)


# In[46]:


ad.head()


# In[47]:


sns.heatmap(pd.isnull(ad),yticklabels=False,cbar=False,cmap='viridis')


# In[48]:


ad


# In[49]:


ad.head()


# In[50]:


X = ad.drop('Clicked on Ad',axis=1)


# In[51]:


y = ad['Clicked on Ad']


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[57]:


from sklearn.linear_model import LogisticRegression


# In[58]:


logmodal = LogisticRegression()


# In[59]:


logmodal.fit(X_train,y_train)


# In[60]:


predictions = logmodal.predict(X_test)


# In[61]:


predictions


# In[62]:


from sklearn.metrics import classification_report


# In[63]:


print(classification_report(y_test,predictions))


# In[64]:


from sklearn.metrics import confusion_matrix


# In[65]:


confusion_matrix(y_test, predictions)


# In[ ]:




