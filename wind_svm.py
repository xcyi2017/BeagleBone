"""
Date: 2016-Mar-26

Solar forecasting using SVM

"""

import numpy as np
import neurolab as nl
import ems_functions as ef
from sklearn.svm import SVR
import time
import pylab as pl
import sqlite3 as sq
from sklearn.externals import joblib


# load data from database
tic = time.time()
train_start = "2010-01-01 00:01:00"
train_end = "2010-01-04 00:00:00"
test_start = "2010-01-04 12:01:00"
test_end = "2010-01-04 13:00:00"

wind, speed, train_len, test_len = ef.load_data_wind(train_start, train_end, test_start, test_end)

# normalization
norm_speed = nl.tool.Norm(speed)
radn = norm_speed(speed)
norm_power = nl.tool.Norm(wind)
windn = norm_power(wind)

# prepare data for SVR
horizon = 1             # 1-step forecasting
lag_wind = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
lag_wind = lag_wind + horizon - 1
lag_speed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

trainX, trainY, testX, testY = ef.prepare_data(lag_wind, lag_speed, windn, radn, train_len, test_len)

# SVR
trainY = np.ravel(trainY)
svr_rbf = SVR(kernel='rbf', tol=1e-5, gamma=0.16, epsilon=0.001, C=0.5)
TY = svr_rbf.fit(trainX, trainY).predict(testX)

TY = norm_power.renorm(TY)                                                # reverse to real scale
ybar = TY.reshape(len(testY))

testY = norm_power.renorm(testY)                                          # actual output
y = testY.reshape(len(testY))
toc = time.time()

# performance evaluation
maxPower = 2.0
ybar = ef.y_in_range(ybar, maxPower)

test_rmse = np.sqrt(np.mean((y - ybar)**2))/maxPower*100
print('Forecasting nRMSE is: %.2f' % test_rmse)
print('Running time is: %.5f ' % (toc - tic))


# save forecast model
# filename = './predict/wind_predict.pkl'
# joblib.dump(svr_rbf, filename, compress=9)

# save forecasted value to database
con = sq.connect('ems.sqlite')
cur = con.cursor()

cur.execute("select rowid from forecast where date = '2010-01-04 12:01:00'")
rowid = cur.fetchone()[0]
for item in ybar:
	cur.execute("update forecast set wind_f = ? where rowid = ?", (item, rowid))
	rowid = rowid + 1

con.commit()

# result display
pl.plot(y, '-', ybar, 'o', linewidth=2.0)
pl.legend(['actual power', 'forecasted power'], loc='lower right')
pl.xlabel('Minute')
pl.ylabel('Wind power (kW)')
pl.savefig("wind.png",dpi=500)
pl.show()