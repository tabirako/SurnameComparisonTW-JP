#!/usr/bin/env python
# coding: utf-8

# # A comparison between the degree of concentration of last names between Japan and Taiwan

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

jp = pd.read_csv('./utf_8_japan.csv')
print(jp.shape)
print(jp.head)


# ## As usual, first we import the files and make them into the same format

# In[2]:


jp_format = jp.drop('rank',axis=1)


# In[3]:


jp_format['ranking']=range(1,jp_format.shape[0]+1)
jp_format = jp_format.rename(columns={"sei": "lastname", "number": "population"})
jp_format=jp_format.reindex(columns=['ranking','lastname','population'])
jp_format


# In[4]:


tw = pd.read_csv('./taiwan.csv')
print(tw.shape)


# In[5]:


tw=tw.drop(['statistic_yyy'],axis=1)
tw


# In[6]:


tw_format = tw.groupby(['ranking','lastname']).agg(sum).reset_index().drop(['age'],axis=1)
tw_format


# ## After that, we making them into numpy arrays and calculate the hhi indexes of them

# In[7]:


t = tw_format.to_numpy()
j = jp_format.to_numpy()


# In[8]:


t


# In[9]:


j


# In[10]:


def hhi(data):
    temp = data*1
    p_sum = sum(temp[:,2])
    temp[:,2] /= p_sum
    return sum(temp[:,2]**2)

def get_percent(data):
    temp = data*1
    p_sum = sum(temp[:,2])
    temp[:,2] /= p_sum
    return temp


# In[11]:


p_j = get_percent(j)
p_t = get_percent(t)
p_j_1k = get_percent(j[:1000])
p_t_1k = get_percent(t[:1000])


print(f'The hhi index of Japan: {hhi(j[:1000])}')
print(f'The hhi index of Taiwan: {hhi(t[:1000])}')


# ## As you can see, Taiwan has a higher level of concentration of last names
# ### The hhi index of Japan: 0.004079500854915622
# ### The hhi index of Taiwan: 0.03864669496413481

# ## Now we tried to draw the individual percents of last names in on a log scale in a decending order
# ### Japan being red and Taiwan being blue curves

# In[12]:


plt.plot(p_j[:,0],p_j[:,2],color='r')
plt.plot(p_t[:,0],p_t[:,2],color='b')
plt.yscale('log')


# ## This is a bit no easy to read, so lets just take the first 1000 entries of each sets

# In[13]:


plt.plot(p_j_1k[:1000,0],p_j_1k[:1000,2],color='r')
plt.plot(p_t_1k[:1000,0],p_t_1k[:1000,2],color='b')
plt.yscale('log')


# ## Which clearly shows Taiwan does have a higher level of concentration

# In[14]:


def acc(data):
    acc = 0
    rev_data = data[::-1]
    acc_list = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        acc += rev_data[i]
        acc_list[i] = acc
    return acc_list


# ## Now we tried to calculate the gini cofficient of the two data sets
# ### Since this is gini cofficient, we also include a gray line for comparison

# In[15]:


plt.plot(p_j_1k[:,0],acc(p_j_1k[:,2]),color='r')
plt.plot(p_t_1k[:,0],acc(p_t_1k[:,2]),color='b')
plt.plot(np.linspace(0,999,1000),np.linspace(0,1,1000),color='gray')


# ### examples taken from https://github.com/oliviaguest/gini

# In[16]:


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


# In[17]:


print(f'The gini index of Japan: {gini(j[:,2])}')
print(f'The gini index of Taiwan: {gini(t[:,2])}')


# ## As you can see again, Taiwan really does have higher level of concentration in last names
# ### The gini index of Japan: 0.8707163603956866
# ### The gini index of Taiwan: 0.9833302879456073

# In[ ]:




