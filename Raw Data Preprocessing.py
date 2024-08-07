#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


url = './file address/excel name'
file_name =  'insert your excel file name'


# In[ ]:


df = pd.read_excel( url , names=['step'])
df


# # Remove negative values
# * Removes negative values because they are symmetrical in the form of Sine

# In[ ]:


plus = df['step'] > 0


# In[ ]:


df = df[plus]
df


# In[ ]:


df = df.reset_index(drop=True)
df


# In[ ]:


prac = df[::5].reset_index(drop=True)
prac


# * Find the starting point of a walk

# In[ ]:


plt.plot(df)


# In[ ]:


plt.figure(figsize=(20, 8))
plt.plot(df)


# In[ ]:


plt.plot(df)


# In[ ]:


plt.plot(df[30000:30200])


# # Envelope progression
# * Extraction of End Points

# In[ ]:


# Extract endpoints to advance the envelope
peaks = []

for i in range(9200, len(df), 20):
    find_peak = (df['step'][i-15 : i+15].idxmax())
    peaks.append(find_peak)
    


# In[ ]:


peaks


# In[ ]:


peaks = df.loc[peaks]
peaks


# In[ ]:


peaks = peaks.reset_index(drop=True)
peaks


# * Understanding Data Flows

# In[ ]:


for i in range(0, 1000, 100):
    plt.figure(figsize=(50, 8))
    plt.plot(peaks[i : i + 100])
    plt.show()


# In[ ]:


plt.plot(peaks[100:1000])


# * Save Envelope Processed Data

# In[ ]:


peaks.to_excel('./file address/excel name'.format(file_name))


# # Applying the Kalman Filter , Defining the Kalman Filter

# In[ ]:


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state  # Initial State
        self.covariance = initial_covariance  # Initial Covariance Matrix
        self.process_noise = process_noise  # System model noise
        self.measurement_noise = measurement_noise  # Noise in observation models
    
    def predict(self):
        # Prediction Step
        # Prediction of health based on system model
        self.state = self.state
        
        self.covariance += self.process_noise  # Covariance update
        
    def update(self, measurement):
        # Update step
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        self.state += kalman_gain * (measurement - self.state)  # update states
        self.covariance *= (1 - kalman_gain)  # Covariance update
    
    def get_state(self):
        return self.state


# In[ ]:


# 초Initial state and initial covariance settings
initial_state = peaks['step'][0]
initial_covariance = 1

# Noise settings for system and observational models
process_noise = 0.1
measurement_noise = 1

# Create Kalman Filter Object
kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

# Performing predictions and updates
filtered_values = []
for measurement in peaks['step']:
    kf.predict()
    kf.update(measurement)
    filtered_values.append(kf.get_state())

# Adding smoothed values to a dataframe
peaks['smoothed'] = filtered_values

# print result
print(peaks)


# In[ ]:


peaks['smoothed'] = round(peaks['smoothed'], 2)


# In[ ]:


peaks


# In[ ]:


plt.figure(figsize=(50, 8))
plt.plot(peaks.smoothed[:600])


# * Save Smoothing Processed Data

# In[ ]:


peaks['smoothed'].to_excel('./file address/excel name'.format(file_name))

