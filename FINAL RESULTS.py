#!/usr/bin/env python
# coding: utf-8

# # Step 0: Prepare Environment

# In[8]:


import numpy as np
import numpy as np
import pandas as pd
import os
import random
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
register_matplotlib_converters()

from statsmodels.tsa.vector_ar.var_model import VAR

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.eval_measures import rmse, aic
import seaborn as sns
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import datetime


# # Step 1: Data Collection

# In[2]:


df = pd.read_csv('unemployment_data.csv', parse_dates=['date'])
df.dropna(axis=1, inplace=True, how='all')
df.dropna(inplace=True)

df['date'] = pd.to_datetime(df.date , format = '%d/%m/%Y')
date = df['date'].values
df.set_index('date', inplace=True)

df.columns = [col.split('(')[0] for col in df.columns]

print(df.shape)
df.head()


# # Step 2: Data Cleaning & Structuring

# In[117]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


# Checking the percentage of missing data in each column
per_missing = df.isna().sum()*100/len(df)
per_missing.sort_values(ascending=False)


# # Step 3: Exploratory Data Analysis

# In[8]:


df.plot(x='date', y=['Temp layoff', 'Perm layoff', 'Job leavers','J searchers worked','J searchers didn’t work', 
                     'Future starts', 'not worked last yr', 'Total unemployed'], figsize=(20, 10))


# In[9]:


###CORRELATION ANALYSIS ###

corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':10})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[10]:


plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('Correlation', size=15)
plt.colorbar()
plt.show()


# In[15]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))

for ycol, ax in zip(['Temp layoff', 'Perm layoff'], axes):

    df.plot(kind='line', x='date', y=ycol, ax=ax, alpha=0.5, color='r')


# In[20]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(40,10))

for ycol, ax in zip(['Job leavers', 'J searchers worked','J searchers didn’t work'], axes):

    df.plot(kind='line', x='date', y=ycol, ax=ax, alpha=0.5, color='r')


# In[21]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(35,10))

for ycol, ax in zip(['Future starts','not worked last yr'], axes):

    df.plot(kind='line', x='date', y=ycol, ax=ax, alpha=0.5, color='r')


# In[27]:


import plotly.offline as py
import plotly.graph_objs as go
unemployed_graph = go.Scatter(x=df.date, y=df['Total unemployed'], name= 'Total unemployed')
py.iplot([unemployed_graph])


# In[28]:


### CHECK FOR NORMALITY AND GAUSSIAN DISTRIBUTION OF THE DATA ###

from scipy import stats
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
df["Total unemployed"].hist(bins=50)
plt.title('Total unemployed')
plt.subplot(1,2,2)
stats.probplot(df['Total unemployed'], plot=plt);
df["Total unemployed"].describe().T


# In[29]:


from scipy import stats
    
stat,p = stats.normaltest(df["Temp layoff"])
print('Temp layoff Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('Temp layoff looks Gaussian (fail to reject H0)')
else:
    print('Temp layoff do not look Gaussian (reject H0)')
    
stat,p = stats.normaltest(df["Perm layoff"])
print('Perm layoff Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('Perm layoff looks Gaussian (fail to reject H0)')
else:
    print('Perm layoff do not look Gaussian (reject H0)')
    
stat,p = stats.normaltest(df["Job leavers"])
print('Job leavers Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('Job leavers looks Gaussian (fail to reject H0)')
else:
    print('Job leavers do not look Gaussian (reject H0)')
    
stat,p = stats.normaltest(df["J searchers worked"])
print('J searchers worked Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('J searchers worked looks Gaussian (fail to reject H0)')
else:
    print('J searchers worked do not look Gaussian (reject H0)')
    
stat,p = stats.normaltest(df["J searchers didn’t work"])
print('J searchers didn’t work Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('J searchers didn’t work looks Gaussian (fail to reject H0)')
else:
    print('J searchers didn’t work do not look Gaussian (reject H0)')
    
stat,p = stats.normaltest(df["Future starts"])
print('Future starts Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('Future starts looks Gaussian (fail to reject H0)')
else:
    print('Future starts do not look Gaussian (reject H0)')
    
stat,p = stats.normaltest(df["not worked last yr"])
print('not worked last yr Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('not worked last yr looks Gaussian (fail to reject H0)')
else:
    print('not worked last yr do not look Gaussian (reject H0)')

    
stat,p = stats.normaltest(df["Total unemployed"])
print('Total unemployed Statistics=%.3f, p=%.3f' % (stat,p))
alpha = 0.05
if p > alpha:
    print('Total unemployed looks Gaussian (fail to reject H0)')
else:
    print('Total unemployed do not look Gaussian (reject H0)')


# # Step 4: Vector Autoregression model

# In[30]:


### CHECKING FOR CAUSALITY USING GRANGER CAUSALITY TEST ###

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=5):    
    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. 

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df  


# In[31]:


grangers_causation_matrix(df, variables = ['Temp layoff', 'Perm layoff', 'Job leavers','J searchers worked',
                                           'J searchers didn’t work', 'Future starts', 'not worked last yr', 
                                           'Total unemployed']) 


# In[32]:


### COINTEGRATION TEST USING JOHANSON'S COINTEGRATION TEST ###

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df [['Temp layoff', 'Perm layoff', 'Job leavers','J searchers worked',
                                           'J searchers didn’t work', 'Future starts', 'not worked last yr', 
                                           'Total unemployed']]) 


# In[3]:


### SPLIT TRAIN TEST ###

train_date = date[:int(len(df)*0.8)]
train = df[:int(len(df)*0.8)].copy()

test_date = date[int(len(df)*0.8):]
test = df[int(len(df)*0.8):].copy()

print(train.shape, test.shape)


# In[4]:


### PLOTTING UTILITY FUNCTIONS ###

def plot_sensor(name):
    
    plt.figure(figsize=(16,4))

    plt.plot(train_date, train[name], label='train')
    plt.plot(test_date, test[name], label='test')
    plt.ylabel(name); plt.legend()
    plt.show()
    
def plot_autocor(name, df):
    
    plt.figure(figsize=(16,4))
    
    # pd.plotting.autocorrelation_plot(df[name])
    # plt.title(name)
    # plt.show()
    
    timeLags = np.arange(1,100*24)
    plt.plot([df[name].autocorr(dt) for dt in timeLags])
    plt.title(name); plt.ylabel('autocorr'); plt.xlabel('time lags')
    plt.show()


# In[5]:


### PLOT ORIGINAL SERIES ###
for col in df.columns:
    plot_sensor(col)


# In[6]:


### PLOT AUTOCORRELATION ###

for col in df.columns:
    plot_autocor(col, train)


# In[8]:


### OPERATE DIFFERENTIATION ###

period = 12

df_diff = df.diff(period).dropna()


# In[9]:


### SPLIT DIFFERENTIAL DATA IN TRAIN AND TEST ###

train_diff = df_diff.iloc[:len(train)-period,:].copy()
test_diff = df_diff.iloc[len(train)-period:,:].copy()

train_init = df.iloc[:len(train)-period,:].copy()
test_init = df.iloc[len(train)-period:-period,:].copy()

print(train_diff.shape, train_init.shape)
print(test_diff.shape, test_init.shape)


# In[13]:


### CHECK FOR STATIONARITY USING AUGMENTED DICKEY-FULLER (ADF) METHOD ###

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# In[14]:


### ADF TEST ON EACH COLUMN ###

for name, column in train_diff.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[15]:


### PLOT DIFFERENTIAL SERIES ###

for col in df.columns:
    plot_autocor(col, train_diff)


# In[16]:


### FIND BEST VAR ORDER ###

AIC = {}
best_aic, best_order = np.inf, 0

for i in range(1,10):
    model = VAR(endog=train_diff.values)
    model_result = model.fit(maxlags=i)
    AIC[i] = model_result.aic
    
    if AIC[i] < best_aic:
        best_aic = AIC[i]
        best_order = i
        
print('BEST ORDER', best_order, 'BEST AIC:', best_aic)


# In[17]:


### PLOT AICs ### 

plt.figure(figsize=(14,5))
plt.plot(range(len(AIC)), list(AIC.values()))
plt.plot([best_order-1], [best_aic], marker='o', markersize=8, color="red")
plt.xticks(range(len(AIC)), range(1,50))
plt.xlabel('lags'); plt.ylabel('AIC')
np.set_printoptions(False)


# In[50]:


## FIT FINAL VAR WITH LAG CORRESPONTING TO THE BEST AIC ###

var = VAR(endog=train_diff.values)
var_result = var.fit(maxlags=best_order)
var_result.aic


# In[21]:


var_result.summary()


# In[23]:


### CHECK FOR SERIAL CORRELATION OF RESIDUALS (ERRORS) USING DURBIN WATSON STATISTIC ###

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(var_result.resid)

for col, val in zip(df[['Temp layoff', 'Perm layoff', 'Job leavers','J searchers worked',
                                           'J searchers didn’t work', 'Future starts', 'not worked last yr', 
                                           'Total unemployed']], out):
    print((col), ':', round(val, 2))


# In[51]:


### UTILITY FUNCTION FOR RETRIVE VAR PREDICTIONS ###

def retrive_prediction(prior, prior_init, steps):
    
    pred = var_result.forecast(np.asarray(prior), steps=steps)
    init = prior_init.tail(period).values
    
    if steps > period:
        id_period = list(range(period))*(steps//period)
        id_period = id_period + list(range(steps-len(id_period)))
    else:
        id_period = list(range(steps))
    
    final_pred = np.zeros((steps, prior.shape[1]))
    for j, (i,p) in enumerate(zip(id_period, pred)):
        final_pred[j] = init[i]+p
        init[i] = init[i]+p    
        
    return final_pred


# In[52]:


### RETRIVE PREDICTION AND OBTAIN THE CORRESPONDING ACTUAL VALUES ###

date = '2017-01-31'
forward = 36
date_range = pd.date_range(date, periods = forward+1, freq='M', closed='right')

final_pred = retrive_prediction(df_diff.loc[:date], df.loc[:date], steps = forward)
final_true = df.loc[date_range]


# In[55]:


### PLOT ACTUAL vs PREDICTION ###

for i,col in enumerate(df.columns):

    plt.figure(figsize=(16,4))
    plt.plot(date_range, final_pred[:,i], c='green', label='prediction var')
    plt.plot(date_range, final_true[col].values, c='orange', label='true')
    plt.ylabel(col); plt.legend()
    plt.show()


# In[54]:


#ACCURACY METRICS

from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

accuracy_prod = forecast_accuracy(final_pred[:,i], final_true[col].values)

for k, v in accuracy_prod.items():
    print((k), ': ',round(v,4))


# # Step 5: Combine VAR & LSTM

# In[102]:


### UTILITY FUNCTIONS FOR NEURAL NETWORK TRAINING ###

def autocor_pred(real, pred, lag=1):
    return pearsonr(real[:-lag], pred[lag:])[0]


seq_length = 12

def get_model():
    
    opt = RMSprop(lr=0.002)
    
    inp = Input(shape=(seq_length, 8))
    
    x = LSTM(100)(inp)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    out = Dense(8)(x)
    
    model = Model(inp, out)
    model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape'])
    
    return model


# In[103]:


### GET TRAIN VALIDATION AND TEST DATA FOR NEURAL NETWORK ###

X = var_result.fittedvalues

y_train = train.iloc[period+best_order:].values
y_train_var = X + train_init.iloc[best_order:].values
X_train = np.concatenate([train_diff.iloc[best_order:].values], axis=1)
X_train_var = np.concatenate([X], axis=1)

y_val = y_train[int(len(X)*0.8):]
y_val_var = y_train_var[int(len(X)*0.8):]
X_val = X_train[int(len(X)*0.8):]
X_val_var = X_train_var[int(len(X)*0.8):]

y_train = y_train[:int(len(X)*0.8)]
y_train_var = y_train_var[:int(len(X)*0.8)]
X_train = X_train[:int(len(X)*0.8)]
X_train_var = X_train_var[:int(len(X)*0.8)]

y_test = test.values
X_test = np.concatenate([test_diff.values], axis=1)


# In[104]:


### SCALE DATA ###

scaler_y = StandardScaler()
scaler = StandardScaler()

y_train = scaler_y.fit_transform(y_train)
y_train_var = scaler_y.transform(y_train_var)
y_val = scaler_y.transform(y_val)
y_val_var = scaler_y.transform(y_val_var)
y_test = scaler_y.transform(y_test)

X_train = scaler.fit_transform(X_train)
X_train_var = scaler.transform(X_train_var)
X_val = scaler.transform(X_val)
X_val_var = scaler.transform(X_val_var)
X_test = scaler.transform(X_test)


# In[105]:


### BUILD DATA GENERATOR ###

generator_train = TimeseriesGenerator(X_train, y_train, length=seq_length, batch_size=100)
generator_train_var = TimeseriesGenerator(X_train_var, y_train_var, length=seq_length, batch_size=100)
generator_val = TimeseriesGenerator(X_val, y_val, length=seq_length, batch_size=100)
generator_val_var = TimeseriesGenerator(X_val_var, y_val_var, length=seq_length, batch_size=100)
generator_test = TimeseriesGenerator(X_test, y_test, length=seq_length, batch_size=100)


# In[106]:


### FIT NEURAL NETWORK WITH VAR FITTED VALUES AND RAW DATA ###

tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)

print('--------', 'train model with VAR fitted values', '--------')
model_var = get_model()
model_var.fit_generator(generator_train_var, steps_per_epoch= len(generator_train_var),
                        epochs=300, validation_data=generator_val_var, validation_steps = len(generator_val_var), 
                        callbacks=[es], verbose = 1)


print('--------', 'train model with raw data', '--------')
model_var.fit_generator(generator_train, steps_per_epoch= len(generator_train),
                        epochs=300, validation_data=generator_val, validation_steps = len(generator_val), 
                        callbacks=[es], verbose = 1)


# In[128]:


### OBTAIN PREDICTIONS AND RETRIVE ORIGINAL DATA ###

true = scaler_y.inverse_transform(y_test[seq_length:])

pred = model_var.predict_generator(generator_test)
pred = scaler_y.inverse_transform(pred)


# In[129]:


### FIT NEURAL NETWORK WITH ONLY ORIGINAL DATA ###

tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)

simple_model = get_model()
simple_model.fit_generator(generator_train, steps_per_epoch= len(generator_train),
                            epochs=300, validation_data=generator_val, validation_steps = len(generator_val), 
                            callbacks=[es], verbose = 1)


# In[130]:


### OBTAIN PREDICTIONS ###

pred_simple = simple_model.predict_generator(generator_test)
pred_simple = scaler_y.inverse_transform(pred_simple)


# In[110]:


### COMPUTE METRICS ON TEST DATA ###

diz_error_lstm, diz_ac_lstm = {}, {}
diz_error_var_lstm, diz_ac_var_lstm = {}, {}
diz_mse_lstm, diz_mse_var_lstm = {}, {} 

for i,col in enumerate(df.columns):
    
    mse = mean_squared_error(true[:,i], pred_simple[:,i])
    diz_mse_lstm[col] = mse
    
    mse = mean_squared_error(true[:,i], pred[:,i])
    diz_mse_var_lstm[col] = mse
    
    error = mean_absolute_error(true[:,i], pred_simple[:,i])
    diz_error_lstm[col] = error
    
    error = mean_absolute_error(true[:,i], pred[:,i])
    diz_error_var_lstm[col] = error
    
    ac = autocor_pred(true[:,i], pred_simple[:,i])
    diz_ac_lstm[col] = ac
    
    ac = autocor_pred(true[:,i], pred[:,i])
    diz_ac_var_lstm[col] = ac


# In[123]:


from math import sqrt
rmse = sqrt(mean_squared_error(true, pred))
print('Test RMSE: %.3f' % rmse)


# In[111]:


plt.figure(figsize=(14,5))
plt.bar(np.arange(len(diz_mse_lstm))-0.15, diz_mse_lstm.values(), alpha=0.5, width=0.3, label='lstm')
plt.bar(np.arange(len(diz_mse_var_lstm))+0.15, diz_mse_var_lstm.values(), alpha=0.5, width=0.3, label='var_lstm')
plt.xticks(range(len(diz_mse_lstm)), diz_mse_lstm.keys())
plt.ylabel('mse'); plt.legend()
np.set_printoptions(False)


# In[112]:


plt.figure(figsize=(14,5))
plt.bar(np.arange(len(diz_error_lstm))-0.15, diz_error_lstm.values(), alpha=0.5, width=0.3, label='lstm')
plt.bar(np.arange(len(diz_error_var_lstm))+0.15, diz_error_var_lstm.values(), alpha=0.5, width=0.3, label='var_lstm')
plt.xticks(range(len(diz_error_lstm)), diz_error_lstm.keys())
plt.ylabel('error'); plt.legend()
np.set_printoptions(False)


# In[113]:


plt.figure(figsize=(14,5))
plt.bar(np.arange(len(diz_ac_lstm))-0.15, diz_ac_lstm.values(), alpha=0.5, width=0.3, label='lstm')
plt.bar(np.arange(len(diz_ac_var_lstm))+0.15, diz_ac_var_lstm.values(), alpha=0.5, width=0.3, label='var_lstm')
plt.xticks(range(len(diz_ac_lstm)), diz_ac_lstm.keys())
plt.ylabel('correlation lag1'); plt.legend()
np.set_printoptions(False)

