#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import numpy as np 
import pandas as pd 
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import copy
import math
from scipy.stats import chi2


import warnings
from pandas.core.common import SettingWithCopyWarning

import Regression

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dcor 
from scipy.stats import pearsonr


# ### Reading the data

# In[2]:


claims = pd.read_excel('claim_history.xlsx')

claims.head()


# In[3]:


claims.info()


# In[4]:


claims.isna().sum()


# In[5]:


claims.describe()


# In[6]:


claims['RED_CAR'] = claims['RED_CAR'].replace(['yes'], 'Yes')
claims['RED_CAR'] = claims['RED_CAR'].replace(['no'], 'No')


# #### Question 1

# In[7]:


#Severity = CLM_AMT / CLM_COUNT if CLM_COUNT > 0
claims['SEVERITY'] = np.where(claims['CLM_COUNT']>0, claims['CLM_AMT']/claims['CLM_COUNT'], 0)


# In[8]:


target = 'SEVERITY'
int_pred = ['AGE', 'BLUEBOOK', 'CAR_AGE', 'HOME_VAL', 'HOMEKIDS', 'INCOME', 'YOJ', 'KIDSDRIV', 'MVR_PTS', 'TIF', 'TRAVTIME', 'CAR_TYPE', 'CAR_USE', 'EDUCATION', 'GENDER', 'MSTATUS', 'PARENT1', 'RED_CAR', 'REVOKED', 'URBANICITY']
cat_cols = ['CAR_TYPE', 'CAR_USE', 'EDUCATION', 'GENDER', 'MSTATUS', 'PARENT1', 'RED_CAR', 'REVOKED', 'URBANICITY']

claims[['BLUEBOOK', 'HOME_VAL', 'INCOME']] = claims[['BLUEBOOK', 'HOME_VAL', 'INCOME']]/1000
train_data = claims[claims['CLM_COUNT'] > 0.0] # Only positive claims
train_data = train_data[[target] + int_pred]   # Only necessary variables
train_data = train_data.dropna().reset_index(drop=True)              # Remove missing values
train_data.shape


# In[9]:


n_sample = train_data.shape[0]
y_train = train_data[target]

# Build a model with only the Intercept term
X_train = train_data[[target]]
X_train.insert(0, 'Intercept', 1.0)
X_train = X_train.drop(columns = target)

result = Regression.GammaRegression(X_train, y_train)

outCoefficient = result[0]
outCovb = result[1]
outCorb = result[2]
llk = result[3]
nonAliasParam = result[4]
outIterationTable = result[5]
y_pred_intercept_only = result[6]


# ##### a) Please generate a histogram and a horizontal boxplot to show the distribution of Severity. For the histogram, use a bin-width of 500 and put the number of policies on the vertical axis. Put the two graphs in the same chart where the histogram is above the boxplot.

# In[10]:


# Create a figure with two subplots, sharing the x-axis
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))

# Plot the histogram on the top subplot
binwidth = 500
ax1.hist(train_data['SEVERITY'], bins=int((max(train_data['SEVERITY']) - min(train_data['SEVERITY'])) / binwidth), color='maroon')
ax1.set_ylabel('Number of policies')
ax1.set_title('Distribution of Severity')

# Plot the boxplot on the bottom subplot
ax2.boxplot(train_data['SEVERITY'], vert=False, widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgray'))
ax2.set_xlabel('SEVERITY')

# Adjust the layout and save the figure
plt.subplots_adjust(hspace=0.3)
plt.savefig('histogram_boxplot.png', dpi=300)
plt.show()


# 
# ##### b) What is the log-likelihood value, the Akaike Information Criterion (AIC) value, and the Bayesian Information Criterion (BIC) value of the Intercept-only model?

# In[11]:


print('Log-likelihood value : ', llk)


# In[12]:


def compute_aic_bic(llk, len_nonAliasParam, n_sample) :
    AIC = -2*llk + 2*len_nonAliasParam
    print('Akaike Information Criterion (AIC) value : ', AIC)
    BIC = -2*llk + len_nonAliasParam*math.log(n_sample)
    print('Bayesian Information Criterion (BIC) value : ', BIC)


# In[13]:


compute_aic_bic(llk, len(nonAliasParam), n_sample)


# #### Question 2

# Use the Forward Selection method to build our model. The Entry Threshold is 0.01.
# 
# ##### a) Please provide a summary report of the Forward Selection in a table. 
# 
# The report should include : 
# 1. the step number, 
# 2. the predictor entered,
# 3. the number of non-aliased parameters in the current model, 
# 4. the log-likelihood value of the current model, 
# 5. the Deviance Chi-squares statistic between the current and the previous models, 
# 6. the corresponding Deviance Degree of Freedom, and
# 7. the corresponding Chi-square significance.
# 

# In[14]:



def create_term_var(col) :
    if col in cat_cols :
        # Reorder the categories in ascending order of frequencies of the target field
        u = trainData[col].astype('category')
        u_freq = u.value_counts(ascending = True)
        pm = u.cat.reorder_categories(list(u_freq.index))
        term_var = pd.get_dummies(pm)
    else :
        term_var = trainData[[col]]
    return term_var


def update_step_summary(preds, train_model, llk_0, df_0):
    
    # Find the predictor
    step_detail = []
    for i in preds :
        X = train_model.join(create_term_var(i),rsuffix="_"+i)
        outList = Regression.GammaRegression(X, y_train)
        llk_1 = outList[3]
        df_1 = len(outList[4])

        deviance_chisq = 2 * (llk_1 - llk_0)
        deviance_df = df_1 - df_0
        deviance_sig = chi2.sf(deviance_chisq, deviance_df)
        step_detail.append([i, df_1, llk_1, deviance_chisq, deviance_df, deviance_sig, outList])
    step_detail_df = pd.DataFrame(step_detail, columns=columns+['output'])
    min_index = step_detail_df['Chi-Square Significance'].idxmin()
    min_row = step_detail_df.iloc[min_index].tolist()
    return min_row


def forward_selection() :
    preds = int_pred.copy()
    y_train = trainData[target]

    # Intercept only model
    X_train = trainData[[target]].copy()
    X_train.insert(0, 'Intercept', 1.0)
    X_train.drop(columns = [target], inplace = True)

    step_summary = []

    outList = Regression.GammaRegression(X_train, y_train)
    llk_0 = outList[3]
    df_0 = len(outList[4])
    step_summary.append(['INTERCEPT', df_0, llk_0, np.nan, np.nan, np.nan])

    chi_sig = 0
    threshold = 0.01
    while chi_sig < threshold :
        if len(preds) == 0 :
            break
        else :
            row = update_step_summary(preds, X_train, llk_0, df_0)
            llk_0 = row[2]
            df_0 = row[1]
            chi_sig = row[-2]
            if chi_sig < threshold :
                step_summary.append(row[:-1])
                X_train = X_train.join(create_term_var(row[0]),rsuffix="_"+row[0])
                out_latest_pr = row[-1]
            preds.remove(row[0])

    return step_summary, out_latest_pr
    


# In[15]:


trainData = train_data.copy()
columns = ["Predictor", "Non-Aliased Parameters", "Log-Likelihood", "Deviance Chi-Squares",
                       "Degrees of Freedom", "Chi-Square Significance"]

report_data, out_pr = forward_selection()


report_df = pd.DataFrame(report_data, columns=columns).reset_index(drop=False)
report_df.rename(columns={'index': 'Step'}, inplace=True)
y_pred = out_pr[6]

report_df


# ##### b) Our final model is the model when the Forward Selection ends. What are the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC) of your final model?
# 

# In[16]:



last_row = report_df.iloc[-1].to_dict()
llk = last_row['Log-Likelihood']
len_nonAliasParam = last_row['Non-Aliased Parameters']
n_sample = trainData.shape[0]


# In[17]:


compute_aic_bic(llk, len_nonAliasParam, n_sample)


# 
# ##### c) Please show a table of the complete set of parameters of your final model (including the aliased parameters). Besides the parameter estimates, please also include the standard errors, the 95% asymptotic confidence intervals, and the exponentiated parameter estimates. Conventionally, aliased parameters have zero standard errors and confidence intervals.

# In[18]:


out_pr[0]


# #### Question 3

# In[19]:



def PearsonCorrelation (x, y):
    '''Compute the Pearson correlation between two arrays x and y with the
       same number of values

       Argument:
       ---------
       x : a Pandas Series
       y : a Pandas Series

       Output:
       -------
       rho : Pearson correlation
       '''

    dev_x = x - np.mean(x)
    dev_y = y - np.mean(y)
   
    ss_xx = np.mean(dev_x * dev_x)
    ss_yy = np.mean(dev_y * dev_y)

    if (ss_xx > 0.0 and ss_yy > 0.0):
        ss_xy = np.mean(dev_x * dev_y)
        rho = (ss_xy / ss_xx) * (ss_xy / ss_yy)
        rho = np.sign(ss_xy) * np.sqrt(rho)
    else:
        rho = np.nan

    return (rho)

def RankOfValue (v):
    '''Compute the ranks of the values in an array v. For tied values, the
    average rank is computed.

    Argument:
    ---------
    v : a Pandas Series

    Output:
    -------
    rankv : Ranks of the values of v, minimum has a rank of zero
    '''

    uvalue, uinv, ucount = np.unique(v, return_inverse = True, return_counts = True)
    urank = []
    ur0 = 0
    for c in ucount:
        ur1 = ur0 + c - 1
        urank.append((ur0 + ur1)/2.0)
        ur0 = ur1 + 1

    rankv = []
    for j in uinv:
        rankv.append(urank[j])

    return (rankv)

def SpearmanCorrelation (x, y):
    '''Compute the Spearman rank-order correlation between two arrays x and y
    with the same number of values

    Argument:
    ---------
    x : a Pandas Series
    y : a Pandas Series

    Output:
    -------
    srho : Spearman rank-order correlation
    '''

    rank_x = RankOfValue(x)
    rank_y = RankOfValue(y)

    srho = PearsonCorrelation(rank_x, rank_y)
    return (srho)

def KendallTaub (x, y):
    '''Compute the Kendall's Tau-b correlation between two arrays x and y
    with the same number of values

    Argument:
    ---------
    x : a Pandas Series
    y : a Pandas Series

    Output:
    -------
    taub : Kendall's tau-b correlation
    '''

    nconcord = 0
    ndiscord = 0
    tie_x = 0
    tie_y = 0
    tie_xy = 0

    x_past = []
    y_past = []
    for xi, yi in zip(x, y):
        for xj, yj in zip(x_past, y_past):
            if (xi > xj):
                if (yi > yj):
                    nconcord = nconcord + 1
                elif (yi < yj):
                    ndiscord = ndiscord + 1
                else:
                    tie_y = tie_y + 1
            elif (xi < xj):
                if (yi < yj):
                    nconcord = nconcord + 1
                elif (yi > yj):
                    ndiscord = ndiscord + 1
                else:
                    tie_y = tie_y + 1
            else:
                if (yi == yj):
                    tie_xy = tie_xy + 1
                else:
                    tie_x = tie_x + 1

        x_past.append(xi)
        y_past.append(yi)

    denom = (nconcord + ndiscord + tie_x) * (nconcord + ndiscord + tie_y)
    if (denom > 0.0):
        taub = (nconcord - ndiscord) / np.sqrt(denom)
    else:
        taub = np.nan

    return (taub)

def AdjustedDistance (x):
    '''Compute the adjusted distances for an array x

    Argument:
    ---------
    x : a Pandas Series

    Output:
    -------
    adj_distance : Adjusted distances
    '''

    a_matrix = []
    row_mean = []

    for xi in x:
        a_row = np.abs(x - xi)
        row_mean.append(np.mean(a_row))
        a_matrix.append(a_row)
    total_mean = np.mean(row_mean)

    adj_m = []
    for row, rm in zip(a_matrix, row_mean):
        row = (row - row_mean) - (rm - total_mean)
        adj_m.append(row)

    return (np.array(adj_m))

def DistanceCorrelation (x, y):
    '''Compute the Distance correlation between two arrays x and y
    with the same number of values

    Argument:
    ---------
    x : a Pandas Series
    y : a Pandas Series

    Output:
    -------
    dcorr : Distance correlation
    '''

    adjD_x = AdjustedDistance (x)
    adjD_y = AdjustedDistance (y)

    v2sq_x = np.mean(np.square(adjD_x))
    v2sq_y = np.mean(np.square(adjD_y))
    v2sq_xy = np.mean(adjD_x * adjD_y)
    

    if (v2sq_x > 0.0 and v2sq_y > 0.0):
        dcorr = (v2sq_xy / v2sq_x) * (v2sq_xy / v2sq_y)
        dcorr = np.power(dcorr, 0.25)
    else :
        dcorr = None

    return (dcorr)


# In[20]:



def compute_error_metrics(y_true, y_pred) :
    
    # Simple Residual
    y_simple_residual = y_true - y_pred

    # Root Mean Squared Error
    mse = np.mean(np.power(y_simple_residual, 2))
    rmse = np.sqrt(mse)
    print("RMSE :", rmse)

    # Relative Error
    relerr = mse / np.var(y_true, ddof = 0)
    print("Relative Error :", relerr)

    # Pearson Correlation
    pearson_corr = PearsonCorrelation (y_true, y_pred)
    print("Pearson Correlation:", pearson_corr)

    # Distance Correlation
    distance_corr = DistanceCorrelation (y_true, y_pred)
    print("Distance Correlation :", distance_corr)
    
    # Mean Absolute Proportion Error
    ape = np.abs(y_simple_residual) / y_train
    mape = np.mean(ape)
    print("Mean Absolute Proportion Error : ", mape)

    


# 
# ##### a) Calculate the Root Mean Squared Error, the Relative Error, the Pearson correlation, the Distance correlation, and the Mean Absolute Proportion Error for the Intercept-only model.
# 

# In[21]:


compute_error_metrics(y_train, y_pred_intercept_only)  


# ##### b) Calculate the Root Mean Squared Error, the Relative Error, the Pearson correlation, the Distance correlation, and the Mean Absolute Proportion Error for our final model in Question 2.

# In[22]:


compute_error_metrics(y_train, y_pred)  


# 
# ##### c) We will compare the goodness-of-fit of your model with that of the saturated model. We will calculate the Pearson Chi-Squares and the Deviance Chi-Squares statistics, their degrees of freedom, and their significance values. Based on the results, do you think your model is statistically the same as the saturated Model?
# 

# In[23]:


# Pearson Residual
y_simple_residual = y_train - y_pred
y_pearson_residual = y_simple_residual / np.sqrt(y_pred)
# Deviance Residual
r_vec = y_train / y_pred
di_2 = 2 * (r_vec - np.log(r_vec) - 1)
y_deviance_residual = np.where(y_simple_residual > 0, 1.0, -1.0) * np.sqrt(di_2)

pearson_chisq = np.sum(np.power(y_pearson_residual, 2.0))
deviance_chisq = np.sum(np.power(y_deviance_residual, 2.0))

df_chisq = n_sample - len_nonAliasParam

pearson_sig = chi2.sf(pearson_chisq, df_chisq)
deviance_sig = chi2.sf(deviance_chisq, df_chisq)

pd.DataFrame(data = [['Pearson', pearson_chisq, df_chisq, pearson_sig],['Deviance', deviance_chisq, df_chisq, deviance_sig]],
            columns = ['Type', 'Statistic', 'Degrees of Freedom', 'Significance (p-value)'])


# From the computed statistics we can clearly see that our model is not the statistically same as the saturated model.

# #### Question 4

# 
# You will visually assess your final model in Question 2. Please color-code the markers according to the magnitude of the Exposure value. You must properly label the axes, add grid lines, and choose appropriate tick marks to receive full credit.
# 
# ##### 1. Plot the Pearson residuals versus the observed Severity.
# 

# In[24]:


exp_train_data = claims[claims['CLM_COUNT'] > 0.0] # Only positive claims
exp_train_data = exp_train_data[[target] + int_pred + ['EXPOSURE']]   # Only necessary variables
exp_train_data = exp_train_data.dropna().reset_index(drop=True)              # Remove missing values


# In[25]:


# Plot Pearson residuals

y_resid = y_train - y_pred
pearsonResid = np.where(y_pred > 0.0, y_resid / np.sqrt(y_pred), np.NaN)
plt.figure(figsize = (8,4), dpi = 200)
sg = plt.scatter(y_train, pearsonResid, c = exp_train_data['EXPOSURE'], marker = 'o')
plt.xlabel('Observed SEVERITY')
plt.ylabel('Pearson Residual')
plt.grid(axis = 'both', linestyle = 'dotted')
plt.colorbar(sg, label = 'Exposure')
plt.show()


# ##### 2. Plot the Deviance residuals versus the observed Severity.
# 

# In[26]:


# Plot Deviance residuals

plt.figure(figsize = (8,4), dpi = 200)
sg = plt.scatter(y_train, y_deviance_residual, c = exp_train_data['EXPOSURE'], marker = 'o')
plt.xlabel('Observed SEVERITY')
plt.ylabel('Deviance Residual')
plt.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()


# In[ ]:




