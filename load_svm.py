"""
Date: 2016-Mar-26

Load forecasting using SVM

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

eload, temp, train_len, test_len = ef.load_data_load(train_start, train_end, test_start, test_end)

# normalization
norm_temp = nl.tool.Norm(temp)
tempn = norm_temp(temp)
norm_power = nl.tool.Norm(eload)
loadn = norm_power(eload)

# joblib.dump(norm_power, 'norm_load.pkl', compress=9)
# joblib.dump(norm_temp, 'norm_temp.pkl', compress=9)

# prepare data for SVR
horizon = 1             # 1-step forecasting
lag_load = np.array([1, 2, 3, 23, 24, 25, 48, 72, 144, 168])
lag_load = lag_load + horizon - 1
lag_temp = []

trainX, trainY, testX, testY = ef.prepare_data_load(lag_load, lag_temp, loadn, tempn, train_len, test_len)

# SVR
trainY = np.ravel(trainY)
svr_rbf = SVR(kernel='rbf', tol=1e-5, gamma=3, epsilon=0.0001)
TY = svr_rbf.fit(trainX, trainY).predict(testX)

TY = norm_power.renorm(TY)                                                # reverse to real scale
ybar = TY.reshape(len(testY))

testY = norm_power.renorm(testY)                                          # actual output
y = testY.reshape(len(testY))
toc = time.time()

maxPower = 8.0
test_rmse = np.sqrt(np.mean((y - ybar)**2))/maxPower*100
print('Load forecasting completed')
print('Forecasting nRMSE is: %.2f' % test_rmse)
print('Running time is: %.5f ' % (toc - tic))

# save forecast model
# filename = './predict/load_predict.pkl'
# joblib.dump(svr_rbf, filename, compress=9)

# save forecasted value to database
con = sq.connect('ems.sqlite')
cur = con.cursor()

cur.execute("select rowid from forecast where date = '2010-01-04 12:01:00'")
rowid = cur.fetchone()[0]
for item in ybar:
	cur.execute("update forecast set load_f = ? where rowid = ?", (item, rowid))
	rowid = rowid + 1

con.commit()

# result display
pl.plot(y, '-', ybar, 'o', linewidth=2.0)
pl.legend(['actual power', 'forecasted power'], loc='upper left')
pl.xlabel('Minute')
pl.ylabel('Load (kW)')
pl.savefig("load_svm.png",dpi=500)
pl.show()