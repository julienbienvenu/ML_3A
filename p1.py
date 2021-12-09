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

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

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

def ranfor(df):
    features =df.drop(labels=['forecastedTagets'], axis=1)    

    #describing dates with new columns
    features = pd.get_dummies(features, columns=['time'])

    # Labels are the values we want to predict
    labels = np.array(features['targets'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features= features.drop('targets', axis = 1)

    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    #Define accuracy of example
    delta = 100 * abs(forecast_target - labels)/labels
    acc_data = 100 - np.mean(delta)
    print('Acc_data :', acc_data)


    '''
    Training : 
    25 % de valeur de training
    shuffle -> prises en aléatoire dans le schéma
    '''

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # The baseline predictions are the targets
    baseline_preds = test_features[:, feature_list.index('avg')]

    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    #print('Average baseline error: ', round(np.mean(baseline_errors), 2), '€')

    # Instantiate model 
    rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    #Hyperparameters
    rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, min_samples_split = 2, min_samples_leaf = 1)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), '€')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    #print('Accuracy:', round(accuracy, 2), '%.')

    # Get numerical feature importances
    importances = list(rf.feature_importances_)# List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
    print([print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances])     


    #print('Training Features Shape:', train_features.shape)
    #print('Training Labels Shape:', train_labels.shape)
    #print('Testing Features Shape:', test_features.shape)
    #print('Testing Labels Shape:', test_labels.shape)    

    print('Acc_train', accuracy)

    joblib.dump('model.joblib')

def MA(df):

    targets = df['targets'].to_list()
    x = [i for i in range(len(targets))]
    train = targets[:int(len(targets)*0.75)]
    test_labels = np.array(targets[int(len(targets)*0.75):])
    
    predictions=[]
    a = len(train)
    for i in range(a, len(targets)):
        avg_mov = (train[-1]+train[-2]+train[-3])/3
        if i<a+10:
            print(avg_mov)
        train.append(avg_mov)
        predictions.append(avg_mov)

    accuracy = 100 - np.mean(100 * (abs(np.array(predictions) - test_labels) / test_labels))
    print('MA accuracy :', accuracy)

    plt.plot(x, train, 'r')
    #plt.plot(x, targets, 'b')
    plt.show()

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
    for t in range(5):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    rmse = np.sqrt(mean_squared_error(test[:5], predictions))
    print('Test RMSE: %.3f' % rmse)



targets = df['targets'].to_list()
x = [i for i in range(len(targets))]
train = targets[:int(len(targets)*0.75)]
test = targets[int(len(targets)*0.75):]

print(arima(train, test))