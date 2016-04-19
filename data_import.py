# 2016-Mar-24
# Import data from excel to txt and then to database

import sqlite3 as sq
import time
import numpy as np

# create date index
start = '2010-01-01 00:01:00'
end = '2010-01-11 00:01:00'

a1 = int(time.mktime(time.strptime(start, '%Y-%m-%d %H:%M:%S')))
b1 = int(time.mktime(time.strptime(end, '%Y-%m-%d %H:%M:%S')))

# import data from txt to database
data = np.loadtxt('forecast.txt')
rad = data[:, 0]
solar = data[:, 1]
speed = data[:, 2]
wind = data[:, 3]
temp = data[:, 4]
eload = data[:, 5]

con = sq.connect('ems.sqlite')
cur = con.cursor()

# table 1: forecasted data
cur.execute("drop table if exists forecast")
cur.execute("create table forecast (date DATETIME, temp REAL, load REAL, load_f REAL, rad REAL, "
            "solar REAL, solar_f REAL, speed REAL, wind REAL, wind_f REAL)")

# table 2: set points
cur.execute("drop table if exists dispatch")
cur.execute("create table dispatch (date DATETIME, diesel REAL, battery REAL, buy REAL, sell REAL)")

for i in range(0, 14400):
	x = time.localtime(a1 + 60 * i)
	y = time.strftime('%Y-%m-%d %H:%M:%S', x)
	cur.execute("insert into forecast values (?,?,?,?,?,?,?,?,?,?)", (y, temp[i], eload[i], 0,
	            rad[i], solar[i], 0, speed[i], wind[i], 0))
	cur.execute("insert into dispatch values (?,?,?,?,?)", (y, 0, 0, 0, 0))

con.commit()