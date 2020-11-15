#!/usr/bin/env python
# coding: utf-8

# In[43]:


#Import Pandas, Numpy, Scipy, and Matplotlib Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
import os
from sklearn import preprocessing
#Read balancescale dataaset into a Pandas data frame

source_directory = "/Users/Brady/Documents/datasets/"


# In[27]:


data_set=[]
season=[]

for x in os.listdir(source_directory):
    data_set.append(x)
    season.append(x.strip('.csv'))

data_set.sort()
season.sort()


# In[28]:


df=pd.read_csv(source_directory+data_set[0])
df.head()


# In[29]:


# Isolate club names from dataframe
club_list=df['HomeTeam'].unique()


# In[30]:


clubs=[]
for i in club_list:
    if i in clubs:
        continue
    else:
        clubs.append(i)

clubs.sort()
clubs


# In[31]:


# Isolate referee names from dataframe
referees=df['Referee'].unique()
referees


# In[32]:


# Drop create an array with of referee names free of duplicates
refs=[]
for i in referees:
    lname=i[2::].strip()
    if lname in refs:
        continue
    else:
        refs.append(lname)

refs.sort()
refs


# In[33]:


# Create empty 2d array to hold yellows and reds for each ref
yc_data = np.zeros((len(clubs),len(refs)), int)
rc_data = np.zeros((len(clubs),len(refs)), int)


# In[34]:


# Fill the 2d array with the card count for each ref and club
for i in range(0,len(df['AY'])):
    ref_name=df['Referee'][i][2::].strip()
    home_club=df['HomeTeam'][i]
    away_club=df['AwayTeam'][i]
    yc_data[clubs.index(home_club),refs.index(ref_name)]+=df['HY'][i]
    yc_data[clubs.index(away_club),refs.index(ref_name)]+=df['AY'][i]


# In[35]:


ref_club_yellows=pd.DataFrame(data=yc_data, index=clubs, columns=refs)


# In[36]:


ref_club_yellows


# In[37]:


ref_club_yellows.describe()


# In[38]:


for i in range(0,len(df['AR'])):
    ref_name=df['Referee'][i][2::].strip()
    home_club=df['HomeTeam'][i]
    away_club=df['AwayTeam'][i]
    rc_data[clubs.index(home_club),refs.index(ref_name)]+=df['HR'][i]
    rc_data[clubs.index(away_club),refs.index(ref_name)]+=df['AR'][i]


# In[39]:


ref_club_reds=pd.DataFrame(data=rc_data, index=clubs, columns=refs)


# In[40]:


ref_club_reds


# In[41]:


ref_club_reds.describe()


# In[44]:


ref_club_reds_normalized = preprocessing.normalize(ref_club_reds, norm='l2')


# In[45]:


ref_club_reds_normalized


# In[46]:


ref_club_reds_normalized=pd.DataFrame(data=rc_data, index=clubs, columns=refs)


# In[47]:


ref_club_reds_normalized


# In[48]:


ref_club_yellows_normalized = preprocessing.normalize(ref_club_yellows, norm='l2')


# In[49]:


ref_club_yellows_normalized


# In[51]:


ref_club_yellows_normalized=pd.DataFrame(data=rc_data, index=clubs, columns=refs)


# In[52]:


ref_club_yellows_normalized


# In[53]:


min_max_scaler = preprocessing.MinMaxScaler()
ref_club_yellows_minmax = min_max_scaler.fit_transform(ref_club_yellows)


# In[54]:


ref_club_yellows_minmax


# In[55]:


min_max_scaler = preprocessing.MinMaxScaler()
ref_club_reds_minmax = min_max_scaler.fit_transform(ref_club_reds)


# In[56]:


ref_club_reds_minmax


# In[ ]:




