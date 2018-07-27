import datetime
import pandas as pd
from pandas_datareader import wb
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Flatten
from keras import optimizers
from keras import regularizers
import keras.backend as K
from sklearn.metrics import mean_squared_error
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.ensemble import ExtraTreesClassifier

style.use('ggplot')
import scipy.stats as st
from get_data import Get_data
#from get_Data_all import Get_data
from iiLSTM import IndexerLSTM

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore')


def model_uncertainity2(model, x_test, y_test, B, confidence):
    MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    learning_phase = True  # use dropout at test time

    MC_samples = [MC_output([x_test, learning_phase])[0] for _ in range(B)]
    MC_samples = np.array(MC_samples)

    eta1 = np.mean(MC_samples - (np.mean(MC_samples) ** 2))  # model misspecification and model uncertainity
    eta2 = mean_squared_error(model.predict(x_test), y_test)  # inherent noise
    model_uncer = np.sqrt(eta1 + eta2)

    Merror = (st.norm.ppf((1 + (confidence / 100)) / 2)) * model_uncer

    return Merror


def denormalize(s, max, min):
    return s * (max - min) + min


def get_results(model, is_xgb=False):
    if is_xgb:
        yhat = model.predict(xgbtest_X)
        ythat = model.predict(xgbtrain_X)
    else:
        yhat = model.predict(test_X)
        ythat = model.predict(train_X)

    yhat_d = denormalize(yhat, gdp_max, gdp_min)
    ythat_d = denormalize(ythat, gdp_max, gdp_min)

    results = {'actual': np.append(train_yd, test_yd),
               'predicted': np.append(ythat_d, yhat_d)}
    resultdf = pd.DataFrame(results, index=d.index)
    rmse = np.sqrt(mean_squared_error(test_yd, yhat_d))
    return rmse, resultdf


data_object = Get_data()

name = input('Enter the state for which you need data for : ')
d2,d = data_object.gea(state=str(name))

gdp_growth=np.array(d['All industry total'])
scaler = MinMaxScaler()
gdp_scaled = scaler.fit_transform(gdp_growth.reshape(-1, 1))

values = d2.values
scaled = scaler.fit_transform(values)

n_train_hours = int(values.shape[0] * 0.75)
train = scaled[:n_train_hours, :]
test = scaled[n_train_hours:, :]
train_X, train_y = train, gdp_scaled[:n_train_hours]

test_X, test_y = test, gdp_scaled[n_train_hours:]
ax = train_X
ay = train_y
bx = test_X
by = test_y

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
length = train_X.shape[1]
numAttr = train_X.shape[2]
neurons = [128, 128, 128, 128]
dropouts = [0.1, 0.1, 0.1]
activations = ['linear', 'linear', 'linear', 'linear']

lstm = IndexerLSTM(length, numAttr, neurons, dropouts, activations)

lstm.buildModel()
lstm.compileModel()
history0 = lstm.fitModel(train_X, train_y, epochs=1000, batchSize=128, validation_data=(test_X, test_y))

plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.plot(history0.history['loss'], label='train')
plt.plot(history0.history['val_loss'], label='test')
plt.legend()
plt.title('LSTM')

plt.show()

gdp_min = min(gdp_growth)
gdp_max = max(gdp_growth)

n_train_hours = int(values.shape[0] * 0.75)
train2 = values[:n_train_hours, :]
test2 = values[n_train_hours:, :]
train_yd = gdp_growth[:n_train_hours]
test_yd = gdp_growth[n_train_hours:]

rmse0, res0 = get_results(lstm)
plt.figure(figsize=(20, 6))
plt.title('GDP, UNITED STATES$, % change')
plt.plot(res0.index, res0['actual'], label='actual', c='red')
plt.plot(res0.index, res0['predicted'], label='LSTM predicted', c='blue')
axvline = int(res0.index[train_y.shape[0]])
plt.axvline(x=axvline, ls='--', lw=2, c='k', label='Train-Test Boundary')
plt.legend()
plt.show()
print("LSTM RMSE: {}".format(rmse0))
merror = model_uncertainity2(lstm.get_model(), test_X, test_y, 100, 95)
print(merror)
