#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from matplotlib import font_manager,rc


# In[ ]:


font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


# In[ ]:


from tableone import TableOne


# # Average the three walking points P1, P2, and P3
#  * After checking the status of the values for the three walk points by BMI group, create a table.

# In[ ]:


df = pd.read_excel("./excel file name")


# In[ ]:


df.columns


# In[ ]:


df


# In[ ]:


# Create a new column by dividing 'bmi' into sections
bins = [20,25,30,99]
labels = ['20~25','25~30','30~']
df['bmi_group'] = pd.cut(df['bmi'], bins=bins, labels=labels)

# select columns
columns = ['bmi_group', 'bmi' , 'p1_mean', 'p2_mean', 'p3_mean']

# create tableone
table = TableOne(data=df, columns=columns, groupby='bmi_group')

# print result
print(table.tabulate(tablefmt='grid'))


# In[ ]:


table


# In[ ]:


df_2 = df.copy()
df_2


# In[ ]:


# Extract only subjects whose bmi range from df to bmi is 20 to 25 with a normal weight
df_2 = df_2[df_2['bmi_group'] == '20~25']
df_2


# In[ ]:


columns = ['bmi', 'p1_mean', 'p2_mean','p3_mean']
table_2 = TableOne(data=df_2, columns=columns, groupby='sex')

table_2


# * Compare only those with bmi section 25 to 27

# In[ ]:


filtered_data = df[(df['bmi'] >= 25) & (df['bmi'] <= 27)]


# In[ ]:


filtered_data


# In[ ]:


columns = ['p1_mean', 'p2_mean', 'p3_mean']
table_3 = TableOne(data=filtered_data, columns=columns, groupby='sex')

table_3


# # Determine the Effect of Age on Gait Points
# 

# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'bmi']]
y = bmi_categori_df[['p1_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'bmi']]
y = bmi_categori_df[['p2_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'bmi']]
y = bmi_categori_df[['p3_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# # Comparison by BMI Section

# ## Boxplot
#  * Check the difference in Gait points data by bmi section

# In[ ]:


bins = [20,25,30,99]
labels = ['20~25','25~30','30~']


# In[ ]:


df_melt = df.melt(id_vars='bmi_group', value_vars=['p1_mean', 'p2_mean', 'p3_mean'])


# In[ ]:


plt.figure(figsize=(16,9), dpi=300)
sns.boxplot(x='variable',y = 'value',hue='bmi_group',data = df_melt)
plt.ylabel('value', fontsize=15)
plt.xlabel('gait point', fontsize=15)

plt.show()


# ## Kruskal-Wallis Test
#  * Briefly verify that BMI is significant for each point

# In[ ]:


# Data frame extraction conditions
bmi_groups = {
    '20미만': (0, 20),
    '20~25': (20, 25),
    '25~29': (25, 30),
    '30이상': (30, np.inf)
}

# List to store results for each group
group_results = []

# Perform a calculation for each group
for group, (lower, upper) in bmi_groups.items():
    group_df = df[(df['bmi'] >= lower) & (df['bmi'] < upper)]
    num_people = len(group_df)
    p1_mean = group_df['p1_mean'].mean()
    p2_mean = group_df['p2_mean'].mean()
    p3_mean = group_df['p3_mean'].mean()
    p1_std = group_df['p1_mean'].std()
    p2_std = group_df['p2_mean'].std()
    p3_std = group_df['p3_mean'].std()
    mean_bmi = group_df['bmi'].mean()

    group_results.append({
        'Group': group,
        'Num_People': num_people,
        'P1_Mean': p1_mean,
        'P2_Mean': p2_mean,
        'P3_Mean': p3_mean,
        'P1_Std': p1_std,
        'P2_Std': p2_std,
        'P3_Std': p3_std,
        'Mean_BMI': mean_bmi
    })

# Generating the resulting data frame
result_df = pd.DataFrame(group_results)

# print dataframe
print(result_df)


# In[ ]:


result_df


# In[ ]:


from scipy.stats import kruskal

# Extract only the columns you need from the data frame
df_subset = result_df[['Mean_BMI', 'P1_Mean', 'P2_Mean', 'P3_Mean']]

# Kruskal-Wallis test
statistic, p_value = kruskal(df_subset['Mean_BMI'], df_subset['P1_Mean'], df_subset['P2_Mean'], df_subset['P3_Mean'])

print("Kruskal-Wallis Test")
print("Statistic:", statistic)
print("P-value:", p_value)


# ## P_value calculation, OLS analysis
#  * Proceed with calculation for each dependent variable

# In[ ]:


df.columns


# In[ ]:


data


# In[ ]:


from scipy.stats import ttest_ind

# 'Perform a t-test between 'P1_mean', 'P2_mean', 'P3_mean' and 'BMI'
p_values = {}
for col in ['p1_mean', 'p2_mean', 'p3_mean']:
    _, p_value = ttest_ind(data[col], data['bmi'])
    p_values[col] = p_value

# print result
for col, p_value in p_values.items():
    print(f"p-value for {col}: {p_value}")


# In[ ]:


import statsmodels.api as sm


# ## univariate comparison
#  * Classification X by bmi section, continuous comparison

# In[ ]:


# Performing linear regression
X = data[['bmi']]
y = data[['p1_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()
# p_values = results.pvalues
# print(p_values)


# In[ ]:


# Performing linear regression
X = data[['bmi']]
y = data[['p2_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()
# p_values = results.pvalues
# print(p_values)


# In[ ]:


# Performing linear regression
X = data[['bmi']]
y = data[['p3_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()
# p_values = results.pvalues
# print(p_values)


# ## BMI Split for Comparison
#  * Multivariate Analysis Comparison

# In[ ]:


bmi_categori_df = data.copy()
bmi_categori_df


# In[ ]:


bmi_categori_df['sex'] = bmi_categori_df['sex'].apply(lambda x: 0 if x == 'M' else 1)
bmi_categori_df


# In[ ]:


# Convert 'bmi' column to categorical variable
categories = pd.get_dummies(pd.cut(bmi_categori_df['bmi'], bins=[20, 25, 30, float('inf')]), prefix='bmi', drop_first=True)
categories.columns = ['bmi_1', 'bmi_2']
bmi_categori_df = pd.concat([bmi_categori_df, categories], axis=1)

# print result
print(bmi_categori_df)


# In[ ]:


bmi_categori_df.columns


# In[ ]:


bmi_categori_df = bmi_categori_df[['p1_mean','p2_mean','p3_mean','bmi','age','sex','bmi_1','bmi_2']]

bmi_categori_df


# * When BMI changes continuously

# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'sex', 'bmi']]
y = bmi_categori_df[['p1_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'sex', 'bmi']]
y = bmi_categori_df[['p2_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'sex', 'bmi']]
y = bmi_categori_df[['p3_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# * Values per walking zone when BMI is divided into zones

# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'sex', 'bmi_1', 'bmi_2']]
y = bmi_categori_df[['p1_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'sex', 'bmi_1', 'bmi_2']]
y = bmi_categori_df[['p2_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# In[ ]:


# Performing linear regression
X = bmi_categori_df[['age', 'sex', 'bmi_1', 'bmi_2']]
y = bmi_categori_df[['p3_mean']]
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)
results = model.fit()

# Check Regression Results
results.summary()


# # Check the difference between left and right feet
# 

# In[ ]:


left_df = pd.read_excel('./left gait points data excel file name')
right_df = pd.read_excel('./right gait points data excel file name')


# In[ ]:


left_df


# In[ ]:


right_df


# In[ ]:


diff_p1 = left_df.p1_mean_l - right_df.p1_mean_r

print(diff_p1)


# In[ ]:


diff_p2 = left_df.p2_mean_l - right_df.p2_mean_r

print(diff_p2)


# In[ ]:


diff_p3 = left_df.p3_mean_l - right_df.p3_mean_r

print(diff_p3)


# In[ ]:


diff_df = pd.concat([diff_p1, diff_p2, diff_p3], axis=1)


# In[ ]:


diff_df = diff_df.rename(columns={0: 'p1_mean_diff', 1: 'p2_mean_diff', 2: 'p3_mean_diff'}).abs()

diff_df


# In[ ]:


print('p1 diff avg: {:.2f}, p2 diff avg: {:.2f}, p3 diff avg: {:.2f}'.format(np.mean(diff_df.p1_mean_diff), np.mean(diff_df.p2_mean_diff),
                                                          np.mean(diff_df.p3_mean_diff)))


# In[ ]:




