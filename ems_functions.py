"""
Date: 2016-Mar-26

User-defined functions for EMS

"""

import numpy as np
import sqlite3 as sq
from scipy.optimize import minimize
from sklearn.externals import joblib
import neurolab as nl
import time
import sys


def load_data_solar(train_start, train_end, test_start, test_end):
	"""

    Load data from database

    """

	con = sq.connect("ems.sqlite")
	cur = con.cursor()

	cur.execute("select rad, solar from forecast where date >= ? and date <= ?", (train_start, train_end))
	train_data = cur.fetchall()  # list
	train_len = len(train_data)  # no. of training data

	cur.execute("select rad, solar from forecast where date >= ? and date <= ?", (test_start, test_end))
	test_data = cur.fetchall()
	test_len = len(test_data)  # no. of testing data

	train_data0 = np.asarray(train_data)  # list to array
	test_data0 = np.asarray(test_data)  # list to array
	data = np.vstack((train_data0, test_data0))

	rad = data[:, 0].reshape(-1, 1)
	solar = data[:, 1].reshape(-1, 1)

	return solar, rad, train_len, test_len


def load_data_load(train_start, train_end, test_start, test_end):
	"""

    Load data from database

    """

	con = sq.connect("ems.sqlite")
	cur = con.cursor()

	cur.execute("select temp, load from forecast where date >= ? and date <= ?", (train_start, train_end))
	train_data = cur.fetchall()  # list
	train_len = len(train_data)  # no. of training data

	cur.execute("select temp, load from forecast where date >= ? and date <= ?", (test_start, test_end))
	test_data = cur.fetchall()
	test_len = len(test_data)  # no. of testing data

	train_data0 = np.asarray(train_data)  # list to array
	test_data0 = np.asarray(test_data)  # list to array
	data = np.vstack((train_data0, test_data0))

	temp = data[:, 0].reshape(-1, 1)
	eload = data[:, 1].reshape(-1, 1)

	return eload, temp, train_len, test_len


def load_data_wind(train_start, train_end, test_start, test_end):
	"""

    Load data from database

    """

	con = sq.connect("ems.sqlite")
	cur = con.cursor()

	cur.execute("select speed, wind from forecast where date >= ? and date <= ?", (train_start, train_end))
	train_data = cur.fetchall()  # list
	train_len = len(train_data)  # no. of training data

	cur.execute("select speed, wind from forecast where date >= ? and date <= ?", (test_start, test_end))
	test_data = cur.fetchall()
	test_len = len(test_data)  # no. of testing data

	train_data0 = np.asarray(train_data)  # list to array
	test_data0 = np.asarray(test_data)  # list to array
	data = np.vstack((train_data0, test_data0))

	speed = data[:, 0].reshape(-1, 1)
	wind = data[:, 1].reshape(-1, 1)

	return wind, speed, train_len, test_len


def prepare_data(lag_power, lag_rad, power, rad, train_len, test_len):
	"""

    Prepare data for SVR

    """

	# training set
	trainY = power[24:train_len]

	X1 = np.zeros((len(trainY), len(lag_power)))
	h = range(24, train_len)
	for i in range(0, len(lag_power)):
		X1[:, i] = power[h - lag_power[i]].reshape(len(trainY))

	X2 = np.zeros((len(trainY), len(lag_rad)))
	for j in range(0, len(lag_rad)):
		X2[:, j] = rad[h - lag_rad[j]].reshape(len(trainY))

	# testing set
	test_from = len(power) - test_len
	testY = power[test_from:]

	TX1 = np.zeros((len(testY), len(lag_power)))
	h = range(test_from, len(power))
	for i in range(0, len(lag_power)):
		TX1[:, i] = power[h - lag_power[i]].reshape(len(testY))

	TX2 = np.zeros((len(testY), len(lag_rad)))
	for j in range(0, len(lag_rad)):
		TX2[:, j] = rad[h - lag_rad[j]].reshape(len(testY))

	if lag_rad == []:
		trainX = X1
		testX = TX1
	elif lag_power == []:
		trainX = X2
		testX = TX2
	else:
		trainX = np.hstack((X1, X2))
		testX = np.hstack((TX1, TX2))

	return trainX, trainY, testX, testY


def prepare_data_load(lag_power, lag_rad, power, rad, train_len, test_len):
	"""

    Prepare data for SVR

    """

	# training set
	trainY = power[168:train_len]

	X1 = np.zeros((len(trainY), len(lag_power)))
	h = range(168, train_len)
	for i in range(0, len(lag_power)):
		X1[:, i] = power[h - lag_power[i]].reshape(len(trainY))

	X2 = np.zeros((len(trainY), len(lag_rad)))
	for j in range(0, len(lag_rad)):
		X2[:, j] = rad[h - lag_rad[j]].reshape(len(trainY))

	# testing set
	test_from = len(power) - test_len
	testY = power[test_from:]

	TX1 = np.zeros((len(testY), len(lag_power)))
	h = range(test_from, len(power))
	for i in range(0, len(lag_power)):
		TX1[:, i] = power[h - lag_power[i]].reshape(len(testY))

	TX2 = np.zeros((len(testY), len(lag_rad)))
	for j in range(0, len(lag_rad)):
		TX2[:, j] = rad[h - lag_rad[j]].reshape(len(testY))

	if lag_rad == []:
		trainX = X1
		testX = TX1
	elif lag_power == []:
		trainX = X2
		testX = TX2
	else:
		trainX = np.hstack((X1, X2))
		testX = np.hstack((TX1, TX2))

	return trainX, trainY, testX, testY


def y_in_range(y, maxPower):
	"""

    Force predictions in range, used for solar and wind

    """

	for i in range(0, len(y)):
		if y[i] > maxPower:
			y[i] = maxPower
		elif y[i] < 0:
			y[i] = 0

	return y


def dispatch(test_start, test_end):
	"""
    Economic dispatch

	--------------------------------
	Cost functions:

	f_diesel = 2.6975 + 1.1153 * p + 0.05 * p * p
	f_bat_discharge = 0.1154 + 0.7975 * p + 0.1409 * p * p
    --------------------------------
    Marginal cost functions:

	df_diesel = 1.1153 + 0.1 * p
	df_bat_discharge = 0.7975 + 0.2818 * p
	df_bat_charge = 1.381
	--------------------------------
	Remark:
	1. battery is cheaper than diesel
	2. Using battery only is cheaper than using the combination of battery and diesel

    """

	con = sq.connect('ems.sqlite')
	cur = con.cursor()
	cur.execute("select load_f, solar_f, wind_f from forecast "
	            "where date >= ? and date <= ?", (test_start, test_end))
	forecasted_data = np.asarray(cur.fetchall())  # a list

	eload = forecasted_data[:, 0]
	solar = forecasted_data[:, 1]
	wind = forecasted_data[:, 2]

	# net load for power dispatch
	netload = eload - solar - wind

	# diesel generator cost function
	# df_diesel = 1.1153 + 0.1*P
	maxDiesel = 5.0

	# battery charging: make money
	# df_bat_charge = 1.381
	maxBat = 2.0

	# battery discharging: cost money
	# df_bat_discharge = 0.7975 + 0.2818*P

	# objective: min cost
	len0 = len(eload)
	set_diesel = np.zeros(len0)
	set_bat = np.zeros(len0)

	bat_start = 180.0  # kW*min
	bat_max = 270.0
	bat_min = 30.0
	bat_now = bat_start  # instant SOC, should get it outside actually

	for i in range(0, len0):
		if netload[i] < 0:  # not happen in this case
			pass

		elif netload[i] == 0:  # not happen in this case
			pass

		else:  # solar + wind < load, diesel and battery supply
			if bat_now > bat_min:
				bat_gap = bat_now - bat_min
				bat_ub = min(maxBat, bat_gap)

				if netload[i] <= bat_ub:  # battery is cheaper
					set_bat[i] = netload[i]

				else:  # diesel is needed, call SLSQP to optimize

					def obj2(x):  # objective function 2
						return 2.6975 + 1.1153 * x[0] + 0.05 * x[0] * x[0] \
						       + 0.1154 + 0.7975 * x[1] + 0.1409 * x[1] * x[1]

					def obj2_deriv(x):  # derivative of obj2
						dfdx0 = 1.1153 + 0.05 * x[0] * 2
						dfdx1 = 0.7975 + 0.1409 * x[1] * 2
						return np.array([dfdx0, dfdx1])

					# add constraints and bounds
					cons = ({'type': 'eq',
					         'fun': lambda x: np.array([x[0] + x[1] - netload[i]]),
					         'jac': lambda x: np.array([1.0, 1.0])})
					bnds = ((0, maxDiesel), (0, bat_ub))

					# optimize
					res = minimize(obj2, [1.0, 1.0], jac=obj2_deriv, constraints=cons,
					               method='SLSQP', bounds=bnds)

					set_diesel[i] = res.x[0]
					set_bat[i] = res.x[1]  # discharge is positive

				# update battery status
				bat_now = bat_now - set_bat[i]

			else:  # battery at lowest energy, cannot discharge, diesel supplied solely
				set_diesel[i] = min(maxDiesel, netload[i])

	print('Unit commitment and economic dispatch completed\n')

	# save set points to database
	cur.execute("select rowid from dispatch where date = '2010-01-04 12:01:00'")
	rowid = cur.fetchone()[0]

	for i in range(0, len0):
		cur.execute("update dispatch set diesel = ?, battery = ? where rowid = ?", (set_diesel[i], set_bat[i], rowid))
		rowid = rowid + 1

	con.commit()

	return set_diesel, set_bat, netload


def load_svm(train_start, train_end, test_start, test_end):
	"""

    Use saved model to predict load

    """

	eload, temp, train_len, test_len = load_data_load(train_start, train_end, test_start, test_end)

	# normalization
	norm_temp = nl.tool.Norm(temp)
	tempn = norm_temp(temp)
	norm_power = nl.tool.Norm(eload)
	loadn = norm_power(eload)

	# prepare data for SVM
	horizon = 1
	lag_load = np.array([1, 2, 3, 23, 24, 25, 48, 72, 144, 168])
	lag_load = lag_load + horizon - 1
	lag_temp = []

	trainX, trainY, testX, testY = prepare_data_load(lag_load, lag_temp, loadn, tempn, train_len, test_len)

	# load saved model
	modelname = 'load_predict.pkl'
	model = joblib.load(modelname)
	TY = model.predict(testX)

	TY = norm_power.renorm(TY)  # reverse to real scale
	ybar = TY.reshape(len(testY))

	testY = norm_power.renorm(testY)  # actual output
	y = testY.reshape(len(testY))

	maxPower = 8.0
	test_rmse = np.sqrt(np.mean((y - ybar) ** 2)) / maxPower * 100

	print('Load forecasting completed\n')
	# print('Forecasting nRMSE is: %.2f\n' % test_rmse)

	# save forecasted value to database
	con = sq.connect('ems.sqlite')
	cur = con.cursor()

	cur.execute("select rowid from forecast where date = '2010-01-04 12:01:00'")
	rowid = cur.fetchone()[0]
	for item in ybar:
		cur.execute("update forecast set load_f = ? where rowid = ?", (item, rowid))
		rowid = rowid + 1

	con.commit()


def solar_svm(train_start, train_end, test_start, test_end):
	"""

    Use saved model to predict solar energy

    """

	solar, rad, train_len, test_len = load_data_solar(train_start, train_end, test_start, test_end)

	# normalization
	norm_rad = nl.tool.Norm(rad)
	radn = norm_rad(rad)
	norm_power = nl.tool.Norm(solar)
	solarn = norm_power(solar)

	# prepare data for SVR
	horizon = 1
	lag_solar = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	lag_solar = lag_solar + horizon - 1
	lag_rad = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	trainX, trainY, testX, testY = prepare_data(lag_solar, lag_rad, solarn, radn, train_len, test_len)

	# load saved model
	modelname = 'solar_predict.pkl'
	model = joblib.load(modelname)
	TY = model.predict(testX)

	TY = norm_power.renorm(TY)  # reverse to real scale
	ybar = TY.reshape(len(testY))

	testY = norm_power.renorm(testY)  # actual output
	y = testY.reshape(len(testY))

	maxPower = 2.0
	test_rmse = np.sqrt(np.mean((y - ybar) ** 2)) / maxPower * 100

	print('Solar forecasting completed\n')
	# print('Forecasting nRMSE is: %.2f\n' % test_rmse)

	# save forecasted value to database
	con = sq.connect('ems.sqlite')
	cur = con.cursor()

	cur.execute("select rowid from forecast where date = '2010-01-04 12:01:00'")
	rowid = cur.fetchone()[0]
	for item in ybar:
		cur.execute("update forecast set solar_f = ? where rowid = ?", (item, rowid))
		rowid = rowid + 1

	con.commit()


def wind_svm(train_start, train_end, test_start, test_end):
	"""

    Use saved model to predict wind energy

    """

	wind, speed, train_len, test_len = load_data_wind(train_start, train_end, test_start, test_end)

	# normalization
	norm_speed = nl.tool.Norm(speed)
	radn = norm_speed(speed)
	norm_power = nl.tool.Norm(wind)
	windn = norm_power(wind)

	# prepare data for SVR
	horizon = 1
	lag_wind = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	lag_wind = lag_wind + horizon - 1
	lag_speed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	trainX, trainY, testX, testY = prepare_data(lag_wind, lag_speed, windn, radn, train_len, test_len)

	# load saved model
	modelname = 'wind_predict.pkl'
	model = joblib.load(modelname)
	TY = model.predict(testX)

	TY = norm_power.renorm(TY)  # reverse to real scale
	ybar = TY.reshape(len(testY))

	testY = norm_power.renorm(testY)  # actual output
	y = testY.reshape(len(testY))

	maxPower = 2.0
	test_rmse = np.sqrt(np.mean((y - ybar) ** 2)) / maxPower * 100

	print('Wind forecasting completed\n')
	# print('Forecasting nRMSE is: %.2f\n' % test_rmse)

	# save forecasted value to database
	con = sq.connect('ems.sqlite')
	cur = con.cursor()

	cur.execute("select rowid from forecast where date = '2010-01-04 12:01:00'")
	rowid = cur.fetchone()[0]
	for item in ybar:
		cur.execute("update forecast set wind_f = ? where rowid = ?", (item, rowid))
		rowid = rowid + 1

	con.commit()


def showResult(test_start, test_end):
	con = sq.connect('ems.sqlite')
	cur = con.cursor()
	cur.execute("select load_f, solar_f, wind_f from forecast "
	            "where date >= ? and date <= ?", (test_start, test_end))
	forecasted_data = cur.fetchall()  # a list

	print('-------------------- Forecasting Results ---------------------------\n')
	print('%-20s%-20s%-20s%-20s' % ('Minute', 'Predicted Load', 'Predicted Solar', 'Predicted Wind'))
	minute = 1
	for row in forecasted_data:
		print('%.2d                  %.4f              %.4f              %.4f' % (minute, row[0], row[1], row[2]))
		minute += 1
	print('\n')

	cur.execute("select diesel, battery from dispatch "
	            "where date >= ? and date <= ?", (test_start, test_end))
	dispatch_data = cur.fetchall()  # a list

	print('-------------------- Unit Commitment Results ---------------------\n')
	print('%-20s%-20s%-20s%-20s%-20s' % ('Minute', 'Solar PV', 'Wind Turbine', 'Diesel Gen', 'Battery'))
	minute = 1
	for row in dispatch_data:
		diesel_status = int(row[0] > 1e-4)
		bat_status = int(row[1] > 1e-4)
		print('%.2d                  %d                   %d                   %d                   %d' % (minute, 1, 1, diesel_status, bat_status))
		minute += 1
	print('\n')
	print('-------------------- Economic Dispatch Results ---------------------\n')
	print('%-20s%-20s%-20s%-20s%-20s' % ('Minute', 'Solar PV', 'Wind Turbine', 'Diesel Gen', 'Battery'))
	minute = 1
	for row in dispatch_data:
		x = np.asarray(forecasted_data)
		print('%.2d                  %.4f              %.4f              %.4f              %.4f' % (minute, x[minute-1, 1], x[minute-1, 2], row[0], row[1]))
		minute += 1
	print('\n')


'''
Date: 2016-Apr-12

Forecast and dispatch step by step

'''

def load_svm_one(test_time):
	"""

	Use saved model to predict load point by point

	"""

	# read data from database
	# need data prior to test_start to setup the testX
	time_int = int(time.mktime(time.strptime(test_time, '%Y-%m-%d %H:%M:%S')))
	before_test = 170
	before_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_int - before_test * 60))

	con = sq.connect("ems.sqlite")
	cur = con.cursor()
	cur.execute("select temp, load from forecast where date >= ? and date <= ?", (before_start, test_time))
	selected_data = cur.fetchall()

	data = np.asarray(selected_data)
	temp = data[:, 0].reshape(-1, 1)
	eload = data[:, 1].reshape(-1, 1)

	# normalization
	# norm_load = joblib.load('norm_load.pkl')
	# norm_temp = joblib.load('norm_temp.pkl')
	norm_load = nl.tool.Norm(eload)
	norm_temp = nl.tool.Norm(temp)
	tempn = norm_temp(temp)
	loadn = norm_load(eload)

	# lagged values used as input variables
	horizon = 1
	lag_load = np.array([1, 2, 3, 23, 24, 25, 48, 72, 144, 168])
	lag_load = lag_load + horizon - 1
	lag_temp = []

	# form testX
	test_from = len(loadn) - 1
	TX1 = np.zeros((1, len(lag_load)))
	for i in range(0, len(lag_load)):
		TX1[0, i] = loadn[test_from - lag_load[i]]

	TX2 = np.zeros((1, len(lag_temp)))
	for j in range(0, len(lag_temp)):
		TX2[0, j] = tempn[test_from - lag_temp[j]]

	if lag_temp == []:
		testX = TX1

	elif lag_load == []:
		testX = TX2

	else:
		testX = np.hstack((TX1, TX2))

	# load saved model
	m = 'load_predict.pkl'
	model = joblib.load(m)
	TY = model.predict(testX)
	ybar = norm_load.renorm(TY)        # reverse to real scale

	# save forecasted value to database, only 1 data
	cur.execute("update forecast set load_f = ? where date = ?", (ybar[0, 0], test_time))
	con.commit()

	return ybar


def solar_svm_one(test_time):
	"""

	Use saved model to predict solar point by point

	"""

	# read data from database
	# need data prior to test_start to setup the testX
	time_int = int(time.mktime(time.strptime(test_time, '%Y-%m-%d %H:%M:%S')))
	before_test = 11
	before_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_int - before_test * 60))

	con = sq.connect("ems.sqlite")
	cur = con.cursor()
	cur.execute("select rad, solar from forecast where date >= ? and date <= ?", (before_start, test_time))
	selected_data = cur.fetchall()

	data = np.asarray(selected_data)
	rad = data[:, 0].reshape(-1, 1)
	solar = data[:, 1].reshape(-1, 1)

	# normalization
	norm_rad = nl.tool.Norm(rad)
	radn = norm_rad(rad)
	norm_solar = nl.tool.Norm(solar)
	solarn = norm_solar(solar)

	# lagged values used as input variables
	horizon = 1
	lag_solar = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	lag_solar = lag_solar + horizon - 1
	lag_rad = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	# form testX
	test_from = len(solarn) - 1
	TX1 = np.zeros((1, len(lag_solar)))
	for i in range(0, len(lag_solar)):
		TX1[0, i] = solarn[test_from - lag_solar[i]]

	TX2 = np.zeros((1, len(lag_rad)))
	for j in range(0, len(lag_rad)):
		TX2[0, j] = radn[test_from - lag_rad[j]]

	if lag_rad == []:
		testX = TX1

	elif lag_solar == []:
		testX = TX2

	else:
		testX = np.hstack((TX1, TX2))

	# load saved model
	m = 'solar_predict.pkl'
	model = joblib.load(m)
	TY = model.predict(testX)
	ybar = norm_solar.renorm(TY)        # reverse to real scale

	# save forecasted value to database, only 1 data
	cur.execute("update forecast set solar_f = ? where date = ?", (ybar[0, 0], test_time))
	con.commit()

	return ybar


def wind_svm_one(test_time):
	"""

	Use saved model to predict wind point by point

	"""

	# read data from database
	# need data prior to test_start to setup the testX
	time_int = int(time.mktime(time.strptime(test_time, '%Y-%m-%d %H:%M:%S')))
	before_test = 11
	before_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_int - before_test * 60))

	con = sq.connect("ems.sqlite")
	cur = con.cursor()
	cur.execute("select speed, wind from forecast where date >= ? and date <= ?", (before_start, test_time))
	selected_data = cur.fetchall()

	data = np.asarray(selected_data)
	speed = data[:, 0].reshape(-1, 1)
	wind = data[:, 1].reshape(-1, 1)

	# normalization
	norm_speed = nl.tool.Norm(speed)
	speedn = norm_speed(speed)
	norm_wind = nl.tool.Norm(wind)
	windn = norm_wind(wind)

	# lagged values used as input variables
	horizon = 1
	lag_wind = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	lag_wind = lag_wind + horizon - 1
	lag_speed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	# form testX
	test_from = len(windn) - 1
	TX1 = np.zeros((1, len(lag_wind)))
	for i in range(0, len(lag_wind)):
		TX1[0, i] = windn[test_from - lag_wind[i]]

	TX2 = np.zeros((1, len(lag_speed)))
	for j in range(0, len(lag_speed)):
		TX2[0, j] = speedn[test_from - lag_speed[j]]

	if lag_speed == []:
		testX = TX1

	elif lag_wind == []:
		testX = TX2

	else:
		testX = np.hstack((TX1, TX2))

	# load saved model
	m = 'wind_predict.pkl'
	model = joblib.load(m)
	TY = model.predict(testX)
	ybar = norm_wind.renorm(TY)        # reverse to real scale

	# save forecasted value to database, only 1 data
	cur.execute("update forecast set wind_f = ? where date = ?", (ybar[0, 0], test_time))
	con.commit()

	return ybar


def dispatch_one(test_time):
	"""
	dispatch step by step

	"""
	con = sq.connect('ems.sqlite')
	cur = con.cursor()
	cur.execute("select load_f, solar_f, wind_f from forecast "
	            "where date >= ? and date <=?", (test_time, test_time))
	forecasted_data = np.asarray(cur.fetchall())  # a list

	eload = forecasted_data[0, 0]
	solar = forecasted_data[0, 1]
	wind = forecasted_data[0, 2]

	# net load for power dispatch
	netload = eload - solar - wind

	# diesel generator cost function
	# df_diesel = 1.1153 + 0.1*P
	maxDiesel = 5.0

	# battery charging: make money
	# df_bat_charge = 1.381
	maxBat = 2.0

	# battery discharging: cost money
	# df_bat_discharge = 0.7975 + 0.2818*P

	# objective: min cost
	set_diesel = 0
	set_bat = 0

	bat_start = 180.0  # kW*min
	bat_max = 270.0
	bat_min = 30.0
	bat_now = bat_start  # instant SOC, should get it outside actually

	if netload < 0:  # not happen in this case
		pass

	elif netload == 0:  # not happen in this case
		pass

	else:  # solar + wind < load, diesel and battery supply
		if bat_now > bat_min:
			bat_gap = bat_now - bat_min
			bat_ub = min(maxBat, bat_gap)

			if netload <= bat_ub:  # battery is cheaper
				set_bat = netload

			else:  # diesel is needed, call SLSQP to optimize

				def obj2(x):  # objective function 2
					return 2.6975 + 1.1153 * x[0] + 0.05 * x[0] * x[0] \
					       + 0.1154 + 0.7975 * x[1] + 0.1409 * x[1] * x[1]

				def obj2_deriv(x):  # derivative of obj2
					dfdx0 = 1.1153 + 0.05 * x[0] * 2
					dfdx1 = 0.7975 + 0.1409 * x[1] * 2
					return np.array([dfdx0, dfdx1])

				# add constraints and bounds
				cons = ({'type': 'eq',
				         'fun': lambda x: np.array([x[0] + x[1] - netload]),
				         'jac': lambda x: np.array([1.0, 1.0])})
				bnds = ((0, maxDiesel), (0, bat_ub))

				# optimize
				res = minimize(obj2, [1.0, 1.0], jac=obj2_deriv, constraints=cons,
				               method='SLSQP', bounds=bnds)

				set_diesel = res.x[0]
				set_bat = res.x[1]  # discharge is positive

			# update battery status
			bat_now = bat_now - set_bat

		else:  # battery at lowest energy, cannot discharge, diesel supplied solely
			set_diesel = min(maxDiesel, netload)

	# save set points to database
	cur.execute("update dispatch set diesel = ?, battery = ? where date = ?", (set_diesel, set_bat, test_time))
	con.commit()

	return set_diesel, set_bat


def dispatch_one2(test_time, load_predict, solar_predict, wind_predict):
	"""
	dispatch step by step version 2

	"""
	# net load for power dispatch
	netload = load_predict - solar_predict - wind_predict

	# diesel generator cost function
	# df_diesel = 1.1153 + 0.1*P
	maxDiesel = 5.0

	# battery charging: make money
	# df_bat_charge = 1.381
	maxBat = 2.0

	# battery discharging: cost money
	# df_bat_discharge = 0.7975 + 0.2818*P

	# objective: min cost
	set_diesel = 0
	set_bat = 0

	bat_start = 180.0  # kW*min
	bat_max = 270.0
	bat_min = 30.0
	bat_now = bat_start  # instant SOC, should get it outside actually

	if netload < 0:  # not happen in this case
		pass

	elif netload == 0:  # not happen in this case
		pass

	else:  # solar + wind < load, diesel and battery supply
		if bat_now > bat_min:
			bat_gap = bat_now - bat_min
			bat_ub = min(maxBat, bat_gap)

			if netload <= bat_ub:  # battery is cheaper
				set_bat = netload

			else:  # diesel is needed, call SLSQP to optimize

				def obj2(x):  # objective function 2
					return 2.6975 + 1.1153 * x[0] + 0.05 * x[0] * x[0] \
					       + 0.1154 + 0.7975 * x[1] + 0.1409 * x[1] * x[1]

				def obj2_deriv(x):  # derivative of obj2
					dfdx0 = 1.1153 + 0.05 * x[0] * 2
					dfdx1 = 0.7975 + 0.1409 * x[1] * 2
					return np.array([dfdx0, dfdx1])

				# add constraints and bounds
				cons = ({'type': 'eq',
				         'fun': lambda x: np.array([x[0] + x[1] - netload]),
				         'jac': lambda x: np.array([1.0, 1.0])})
				bnds = ((0, maxDiesel), (0, bat_ub))

				# optimize
				res = minimize(obj2, [1.0, 1.0], jac=obj2_deriv, constraints=cons,
				               method='SLSQP', bounds=bnds)

				set_diesel = res.x[0]
				set_bat = res.x[1]  # discharge is positive

			# update battery status
			bat_now = bat_now - set_bat

		else:  # battery at lowest energy, cannot discharge, diesel supplied solely
			set_diesel = min(maxDiesel, netload)

	# save set points to database
	con = sq.connect('ems.sqlite')
	cur = con.cursor()
	cur.execute("update dispatch set diesel = ?, battery = ? where date = ?", (set_diesel, set_bat, test_time))
	con.commit()

	return set_diesel, set_bat



# testing code

# if __name__ == '__main__':
# 	train_start = "2010-01-01 00:01:00"
# 	train_end = "2010-01-04 00:00:00"
# 	test_start = "2010-01-04 12:01:00"
# 	test_end = "2010-01-04 13:00:00"
#
# 	tic = time.time()
# 	load_svm(train_start, train_end, test_start, test_end)
# 	wind_svm(train_start, train_end, test_start, test_end)
# 	solar_svm(train_start, train_end, test_start, test_end)
# 	set_diesel, set_bat, netload = dispatch(test_start, test_end)
# 	toc = time.time()
# 	print('Total running time is: %.4f s\n' % (toc-tic))
#
# 	showResult(test_start, test_end)

	# if sys.platform == 'win32':
	# 	import pylab as pl
	# 	pl.plot(set_diesel, label='diesel generator', linewidth=2.0)
	# 	pl.plot(set_bat, label='battery', linewidth=2.0)
	# 	pl.plot(netload, label='netload', linewidth=2.0)
	# 	pl.legend(loc='upper right')
	# 	# pl.savefig('dispatch.png', dpi=500)
	# 	# pl.show()


if __name__ == '__main__':
	test_start = "2010-01-04 12:01:00"
	test_end = "2010-01-04 13:00:00"

	con = sq.connect("ems.sqlite")
	cur = con.cursor()
	cur.execute("select date from forecast where date >= ? and date <= ?", (test_start, test_end))
	selected_dates = cur.fetchall()

	print('\n')
	print('---------------------- Forecasting and Economic Dispatch Results ----------------------\n')
	print('%-20s%-20s%-20s%-20s%-20s%-20s' % ('Minute', 'Load', 'Solar PV', 'Wind T', 'Diesel', 'Battery'))
	for i in range(0, len(selected_dates)):
		test_time = ''.join(selected_dates[i])
		load_f = load_svm_one(test_time)
		solar_f = solar_svm_one(test_time)
		wind_f = wind_svm_one(test_time)
		set_diesel, set_bat = dispatch_one(test_time)
		# minute = test_time.encode('ascii','ignore')[11:16]
		# print('%s               %.4f              %.4f              %.4f              %.4f              %.4f'
		#       % (minute, load_f, solar_f, wind_f, set_diesel, set_bat))
		print('%2d                  %.4f              %.4f              %.4f              %.4f              %.4f'
		      % (i + 1, load_f, solar_f, wind_f, set_diesel, set_bat))
		# time.sleep(0.5)