"""
Date: 2016-Apr-07

Use Mixed Integer Programming to perform economic dispatch

Remark: SUCCESSFUL with GUROBI Optimization

"""

from gurobipy import *
import numpy as np
import sqlite3 as sq

# read forecasted load, wind, solar from database
test_start = "2010-01-04 12:01:00"
test_end = "2010-01-04 13:00:00"

con = sq.connect('ems.sqlite')
cur = con.cursor()
cur.execute("select load_f, solar_f, wind_f from forecast "
            "where date >= ? and date <= ?", (test_start, test_end))
forecasted_data = np.asarray(cur.fetchall())  # list

eload = forecasted_data[:, 0]
solar = forecasted_data[:, 1]
wind = forecasted_data[:, 2]

# net load for power dispatch
netload = eload - solar - wind

# objective: f_diesel + f_bat
len0 = len(eload)
set_diesel = np.zeros(len0)
set_bat = np.zeros(len0)
set_int1 = np.zeros(len0)
set_int2 = np.zeros(len0)

bat_start = 180.0  # kW*min
bat_max = 270.0
bat_min = 30.0
bat_now = bat_start  # instant SOC, should get it outside actually

maxDiesel = 5.0
maxBat = 2.0

m = Model("mip")

for i in range(0, len0):
	if netload[i] <= 0:  # not happen in this case
		pass

	else:  # solar + wind < load, diesel and battery supply
		if bat_now > bat_min:
			bat_gap = bat_now - bat_min
			bat_ub = min(maxBat, bat_gap)  # update bat bound every step

			# create variables
			diesel = m.addVar(lb=0, ub=maxDiesel, name='diesel')
			bat = m.addVar(lb=0, ub=bat_ub, name='bat')
			int1 = m.addVar(vtype=GRB.BINARY, name='int1')
			int2 = m.addVar(vtype=GRB.BINARY, name='int2')

			m.update()

			# objective function
			obj = 2.6975 * int1 + 1.1153 * diesel + 0.05 * diesel * diesel \
			      + 0.1154 * int2 + 0.7975 * bat + 0.1409 * bat * bat

			m.setObjective(obj)

			# add constraints
			m.addConstr(diesel * int1 + bat * int2, GRB.EQUAL, netload[i], "c0")
			m.addConstr(diesel <= maxDiesel * int1, "c1")
			m.addConstr(bat <= bat_ub * int2, "c2")

			m.optimize()

			set_diesel[i] = diesel.x
			set_bat[i] = bat.x
			set_int1[i] = int1.x
			set_int2[i] = int2.x

			# update battery status
			bat_now = bat_now - set_bat[i]

		else:  # battery at lowest energy, cannot discharge, diesel supplied solely
			set_diesel[i] = min(maxDiesel, netload[i])

# Cost calculation
f_diesel = np.zeros(len0)
f_bat_disc = np.zeros(len0)

for i in range(0, len0):
	if set_diesel[i] > 1e-4:
		f_diesel[i] = 2.6975 + 1.1153 * set_diesel[i] + 0.05 * set_diesel[i] * set_diesel[i]
		# print('%.4f' % f_diesel[i])

print ("--------------------------------------")
for i in range(0, len0):
	if set_bat[i] > 1e-4:
		f_bat_disc[i] = 0.1154 + 0.7975 * set_bat[i] + 0.1409 * set_bat[i] * set_bat[i]
		# print('%.4f' % (f_diesel[i]+f_bat_disc[i]))

print ("--------------------------------------")
print (np.sum(f_diesel) + np.sum(f_bat_disc))
