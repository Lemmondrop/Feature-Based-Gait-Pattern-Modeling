#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns
sns.set()

import scipy as sp
from scipy import stats, integrate
from scipy.optimize import minimize_scalar

import warnings
warnings.filterwarnings(action='ignore')

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Defining the method of finding a gait point(P1, P2, P3)

# In[ ]:


def find_p1(df, index : int, ran : int):
    p1_peaks = []
    for i in range(index, len(df), ran):
        find_peak = (df['step'][i-10:i]).idxmin()
        p1_peaks.append(find_peak)

    return p1_peaks

def find_p2(df, index : int, ran : int): # the highest point in time
    peaks = []
    for i in range(index, len(df), ran):
        find_peak = (df['step'][i-10:i+15]).idxmax()
        peaks.append(find_peak)
    
    #print(peaks)
    return peaks

def find_p3(df, index : int, ran : int):
    peaks = []

    for i in range(index, len(df), ran):
        find_peak = (df['step'][i-10:i+15]).idxmax()
        peaks.append(find_peak)

    p3_peaks = []
    
    for peak_index in peaks:  # Traversing the indexes stored in the peaks list
        for i in range(peak_index + 5, min(peak_index + 13, len(df))):  # Traverse the range within 11 after peak_index
            diff = abs(df['step'][i - 1] - df['step'][i])
            if diff >= 15: # Finds the difference between the previous index and the value of step over 15
                p3_peaks.append(i)

    return p3_peaks


# # Define a walking point visualization method

# In[ ]:


def p1_show(start : int, end : int, df, step : int, ran : int, index : int):
    p1_seri = pd.Series(find_p1(df, index, ran))
    
    p1_seri_list = []
    for i in range(-start, end, 1):
        p1_data = df['step'][p1_seri[:step] + i]
        p1_seri_list.append(p1_data)
    
    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(p1_seri_list)
    
def p2_show(heel_strike : int, toe_off : int, df, step : int , ran : int, index :int):
    p2_seri = pd.Series(find_p2(df, index, ran))
    
    p2_seri_list = []
    for i in range(-heel_strike, toe_off, 1):
        step_data = df['step'][p2_seri[:step] + i]
        p2_seri_list.append(step_data)
        
    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(p2_seri_list)
    
def p3_show(start : int, end : int, df, step : int, ran : int, index : int):
    p3_seri = pd.Series(find_p3(df, index, ran))
    
    p3_seri_list = []
    
    for i in range(-start, end, 1):
        p3_data = df['step'][p3_seri[:step] + i]
        p3_seri_list.append(p3_data)
    
    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(p3_seri_list)


# In[ ]:


def peaks_show_left(heel_strike : int, toe_off : int, df, step : int , ran : int, index :int):
    peaks_seri = pd.Series(find_p2(df, index, ran))
    
    peak_seri_list = []
    for i in range(-heel_strike, toe_off, 1):
        step_data = df['step'][peaks_seri[1:step:2] + i]
        peak_seri_list.append(step_data)
        
    plt.figure(figsize=(6,4))
    plt.plot(peak_seri_list)
    
def peaks_show_right(heel_strike : int, toe_off : int, df, step : int , ran : int, index :int):
    peaks_seri = pd.Series(find_p2(df, index, ran))
    
    peak_seri_list = []
    for i in range(-heel_strike, toe_off, 1):
        step_data = df['step'][peaks_seri[:step:2] + i]
        peak_seri_list.append(step_data)
        
    plt.figure(figsize=(6,4))
    plt.plot(peak_seri_list)


# # Start Code

# In[ ]:


df = pd.read_excel('./file address/excel name', usecols='B')
df


# In[ ]:


peaks = []

for i in range(3000, len(df), 30):
    find_peak = (df['step'][i-10:i+15]).idxmax()
    peaks.append(find_peak)


# In[ ]:


peaks # The value of the index address where the value of step is highest.


# In[ ]:


peaks[0]


# In[ ]:


df.loc[0]['step']


# In[ ]:


for i in peaks:
    print(df.loc[i]['step'])


# In[ ]:


for i in peaks:
    print((df.loc[i]['step'] * 0.8))


# In[ ]:


peaks = []

for i in range(3000, len(df), 30):
    find_peak = (df['step'][i-10:i+15]).idxmax()
    peaks.append(find_peak)
    
p3_peaks = []

for i in range(3000, len(df) - 1):  # Traverse to before last index
    diff = abs(df['step'][i] - df['step'][i + 1])
    if diff >= 10:  # The difference between the value of the current index and the next index is 15 or more.
        p3_peaks.append(i)


# # Verify the extracted values

# In[ ]:


p3_peaks


# In[ ]:


p3_peak_result = [] ## Duplicate value removal code

for value in p3_peaks:
    if value not in p3_peak_result:
        p3_peak_result.append(value)

print(p3_peak_result)


# In[ ]:


print(len(p3_peaks))
print(len(p3_peak_result))


# In[ ]:


gait_df = pd.read_excel('./file address/excel name', usecols='B')
gait_df


# In[ ]:


p3_peak = find_p3(gait_df, 10, 30)
p3_peak


# In[ ]:


p3_peak = gait_df.loc[p3_peak]
p3_peak = p3_peak[p3_peak.step != 0]
p3_peak


# In[ ]:


p3_show(start=20,
       end=10,
       df=gait_df,
       step=10,
       ran=30,
       index=1500)

