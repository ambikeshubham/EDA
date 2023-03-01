#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


titanic = sns.load_dataset('titanic')


# In[3]:


titanic


# In[4]:


titanic.shape


# In[5]:


titanic.head(2)


# In[6]:


titanic.tail(2)


# In[7]:


titanic.info()


# In[8]:


titanic.isnull().sum()


# In[ ]:





# In[9]:


titanic.duplicated().sum()


# In[10]:


sns.heatmap(titanic.isnull(),yticklabels=False, cmap='viridis')


# In[11]:


titanic.drop('deck',inplace=True,axis=1)


# In[12]:


titanic


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='survived',data=titanic,palette='RdBu_r')


# In[14]:


sns.countplot(x='survived',hue='sex',data=titanic,palette='RdBu_r')


# In[15]:


titanic['fare'].hist(color='red',bins=40,figsize=(8,6))


# In[16]:


plt.figure(figsize=(12,7))
sns.boxplot(x='pclass',y='age',data = titanic,palette='RdBu_r')


# In[17]:


def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass ==1:
            return 37
        elif pclass ==2:
            return 29
        else:
            return 24
    else:
        return age
    


# In[18]:


titanic.isnull().sum()


# In[19]:


titanic['age']=titanic[['age','pclass']].apply(impute_age,axis=1)


# In[20]:


sns.heatmap(titanic.isnull(),yticklabels=False, cmap='viridis',cbar=False)


# In[21]:


titanic.shape


# In[22]:


titanic.info()


# In[23]:


sex_new=pd.get_dummies(titanic['sex'],drop_first=True)
sex_new


# In[24]:


titanic.embarked.unique()


# In[25]:


embarked_new=pd.get_dummies(titanic['embarked'],drop_first=True)
embarked_new


# In[26]:


sns.heatmap(titanic.isnull(),yticklabels=False, cmap='viridis',cbar=False)


# In[27]:


titanic.drop(['sex','embarked'],axis=1,inplace=True)


# In[28]:


titanic.isnull().sum()


# In[29]:


titanic.info()


# In[30]:


titanic.drop(['embark_town'],axis=1,inplace=True)


# In[31]:


titanic


# In[32]:


titanic.describe()


# In[33]:


plt.figure(figsize=(20,10))
corr_columns = titanic.corr()
sns.heatmap(corr_columns,annot=True, fmt = ".2f", cmap = "coolwarm")
plt.title(' Titanic EDA')


# In[34]:


titanic.drop(['class' , 'who','adult_male', 'alive'],axis=1,inplace=True)
titanic


# In[35]:


titanic.drop(['alone'],axis=1,inplace=True)
titanic


# In[36]:


X= titanic.drop(["survived"],axis=1)
print(X)
y=titanic["survived"]
print(y)


# In[37]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split


# In[38]:


scaler = StandardScaler()


# In[39]:


scaler = StandardScaler()


# In[40]:


arr=scaler.fit_transform(X)
print(arr)


# In[41]:


df1=pd.DataFrame(arr)


# In[42]:


df1


# In[44]:


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.30, random_state=42)


# In[45]:


X_train = scaler.fit_transform(X_train)


# In[46]:


print(X_train)


# In[47]:


X_test=scaler.transform(X_test)
X_test


# In[48]:


X_train.shape


# In[49]:


y_train.shape


# In[50]:


from sklearn.linear_model import LinearRegression
##cross validation
from sklearn.model_selection import cross_val_score


# In[51]:


regression = LogisticRegression()


# In[52]:


regression.fit(X_train,y_train)


# In[53]:


mse = cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
mse


# In[54]:


np.mean(mse)


# In[55]:


import pickle


# In[56]:


pickle.dump(regression,open('Titanic_lr_model.pickle','wb'))


# In[57]:


ls


# In[58]:


model=pickle.load(open('Titanic_lr_model.pickle','rb'))


# In[59]:


model


# In[60]:


titanic


# In[61]:


test1=scaler.transform([[3,22.0,1,0,7.2500]])


# In[62]:


model.predict(test1)


# In[63]:


regression.score(X_test,y_test)


# In[64]:


regression.score(X_train,y_train)


# In[65]:


regression.coef_


# In[66]:


regression.intercept_


# In[67]:


from sklearn.metrics import classification_report, plot_confusion_matrix,plot_roc_curve


# In[68]:


plot_confusion_matrix(regression,X_test,y_test)


# In[69]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[70]:


y_pred = regression.predict(X_test)
y_pred


# In[71]:


print(classification_report(y_test,y_pred))


# In[72]:


plot_roc_curve(regression,X_test,y_test)


# In[ ]:





# In[ ]:




