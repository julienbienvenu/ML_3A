import pandas as pd
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import pydot
import datetime
import matplotlib.pyplot as plt
from datetime import datetime
import random
import joblib

from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AutoReg

#Initialisation

df = pd.read_csv('TimeSeriesForecasting.csv')

rows = df.columns[1:]

for row in rows:
    
    if df[row][0] == 0:
        df=df.drop(labels=[row], axis=1)
        
    elif math.isnan(df[row][0]):
        df=df.drop(labels=[row], axis=1)
        
df=df.dropna(axis = 0)
df.to_csv('new_data.csv') #save clean data

forecast_target = np.array(df['forecastedTagets'][5:]) #save predictive comparaison 

#On crée une colonne qui est la prédiction des 5 jours précédents 
df['avg'] = df['targets'].rolling(5).mean()
df = df.reset_index()
df.to_csv('new_data.csv') #save clean data

print('Save!')

list_date = []
for i in range(len(df.index)):
    list_date.append(df['time'][i][11:-6])

df['time'] = list_date
df=df[5:]

def random_forest(df):
    features =df.drop(labels=['forecastedTagets'], axis=1)    

    #describing dates with new columns
    features = pd.get_dummies(features, columns=['time'])

    # Labels are the values we want to predict
    labels = np.array(features['targets'])

    # Remove the labels from the features
    features= features.drop('targets', axis = 1)

    # Convert to numpy array
    features = np.array(features)
   
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42, shuffle=False)

    # Instantiate model 
    rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE ARIMA: %.3f' % rmse)

    return predictions.tolist()

def MA(train, test):

    predictions=[]
    a = len(train)
    for i in range(len(test)):
        avg_mov = (train[-1]+train[-2]+train[-3])/3
        train.append(avg_mov)
        predictions.append(avg_mov)

    predictions_arr = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions_arr))
    print('Test RMSE MVA: %.3f' % rmse)

    return predictions

    
def calculate_ema(targets_train, days, smoothing=2):
    ema = [sum(targets_train[:days]) / days]
    for target in targets_train[days:]:
        ema.append((target * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema

def exp_avg_mov(df):

    targets = df['targets'].to_list()
    x = [i for i in range(len(targets))]
    train = targets[:int(len(targets)*0.75)]
    test_labels = np.array(targets[int(len(targets)*0.75):])

    train_ema = calculate_ema(train, 10)

    for _ in range(len(test_labels)):
        train_ema.append((targe * (smoothing / (1 + days))) + train_ema[-1] * (1 - (smoothing / (1 + days))))

def arima(train, test):

    history = [x for x in train]
    predictions = list()
    
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE ARIMA: %.3f' % rmse)

    return predictions.tolist()

def ar(train, test):
    model = AutoReg(train, lags=29)
    model_fit = model.fit()
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE AR: %.3f' % rmse)
    return predictions.tolist()

def arma(train, test):
    model = ARMA(train)
    model_fit = model.fit()
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print(predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE ARMA: %.3f' % rmse)
    
    return predictions.tolist()

def graph():
    
    ## Ploting the dispersion
    plt.figure()
    plt.subplot(221)
    lag_plot(df['targets'])
    plt.title('Dispersion of targets')
    plt.subplot(222)
    lag_plot(df['forecastedTagets'], c='Red')
    plt.title('Dispersion of Forecast targets')

    rmse = np.sqrt(mean_squared_error(np.array(df['targets']), np.array(df['forecastedTagets'])))
    print('ForecastedTargets RMSE: %.3f' % rmse)

    #Running predictive models    
    #mva_pred = MA(train, test)
    ar_pred = ar(train, test)
    '''
    arima_pred = arima(train, test)
    rf_pred = random_forest(df) 
    '''   

    #df['MVA'] = train + mva_pred
    df['AR'] = train + ar_pred

    plt.subplot(223)    
    lag_plot(df['AR'][:len(train)], c='Green', label='Train')
    lag_plot(df['AR'][len(train):], c='Yellow', label='Test')
    plt.legend(loc='lower right')
    plt.title('Dispersion of AR targets')
    plt.savefig('Dispersion')
    plt.show()

    '''
    df['ARIMA'] = train + arima_pred
    df['RandomForest'] = train + rf_pred
    '''
    
    #Plotting the predictive models results
    plt.subplot(223)
    plt.title('AR method')
    plt.plot(df['AR'])
    plt.subplot(224)
    plt.title('ARIMA method')
    plt.plot(df['ARIMA'])
    plt.subplot(225)
    plt.title('Random Forest method')
    plt.plot(df['RandomForest'])
    plt.subplot(225)
    plt.title('MVA method')
    plt.plot(df['MVA'])
    plt.save('Results.png')
    plt.show()

    df.to_csv('Results.csv')

targets = df['targets'].to_list()
x = [i for i in range(len(targets))]
train = targets[:int(len(targets)*0.75)]
test = targets[int(len(targets)*0.75):]

graph()