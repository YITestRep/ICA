#coding: utf-8
import numpy as np
import scipy as sp
import pandas as pd
import pylab as pl
from sklearn.decomposition import FastICA


import csv
"""
data = csv.reader(open('data/under.txt', 'rb'), delimiter=' ', quotechar='|')
print len(data)
data_np = np.array(data)
print data_np.shape
"""

def zscore(data):

	data = data - data.mean()
	data = data/data.std()
	return data
def paa_time(data, sampling_time, start_time, stop_time):
	stop_time = stop_time + sampling_time
	complement_time = np.arange(start_time, stop_time, sampling_time)
	#print complement_time
	bin_names = complement_time[0:-1]
	print complement_time
	data["Complement"] = pd.cut(data.index, complement_time, labels=bin_names)
	print data.describe()
	c_data = data.groupby("Complement").apply(lambda x:x.mean())
	return c_data.drop("Complement", axis=1)

def getdata(source):
	contents = csv.reader(open(source, 'rb'), delimiter=' ')
	lines = []
	for line in contents:
		lines.append(line)
	data = np.array(lines).astype(np.float)
	df = pd.DataFrame(data)
	df.columns = ['time', 'x', 'y', 'z']
	df = df.set_index('time')
	
	return df


if __name__ == "__main__":
	acce_data = {}
	under = getdata('data/under.txt')
	left= getdata('data/left.txt')
	right= getdata('data/right.txt')

	z_under = zscore(under)
	z_left = zscore(left)
	z_right = zscore(right)
	start_time = 100
	stop_time = 200
	paa_under = paa_time(z_under, 0.2, start_time =start_time, stop_time = stop_time)#z_under.index[-1])
	paa_left = paa_time(z_left, 0.2, start_time = start_time, stop_time = stop_time)#z_left.index[-1])
	paa_right = paa_time(z_right, 0.2, start_time = start_time, stop_time = stop_time)#z_right.index[-1])

	
	fig, axes = pl.subplots(nrows=3, ncols=3, sharey=True)
	paa_under.x.plot(ax=axes[0,0]); #axes[0,0].set_title('Under X')
	paa_under.y.plot(ax=axes[0,1]); #axes[0,1].set_title('Under Y')
	paa_under.z.plot(ax=axes[0,2]); #axes[0,2].set_title('Under Z')

	paa_left.x.plot(ax=axes[1,0]); #axes[1,0].set_title('Left X')
	paa_left.y.plot(ax=axes[1,1]); #axes[1,1].set_title('Left Y')
	paa_left.z.plot(ax=axes[1,2]); #axes[1,2].set_title('Left Z')

	paa_right.x.plot(ax=axes[2,0]); #axes[2,0].set_title('Right X')
	paa_right.y.plot(ax=axes[2,1]); #axes[2,1].set_title('Right Y')
	paa_right.z.plot(ax=axes[2,2]); #axes[2,2].set_title('Right Z')

	
	
	#under.plot(subplots=True, style=list('rbg'), sharey=True)
	#pl.show()
	
	# Gen X Matrix
	x1 = paa_under.values
	x2 = paa_left.values
	x3 = paa_right.values

	
	X = np.hstack((x1, x2, x3))
	
	#ica
	pl.figure(1)
	ica = FastICA(max_iter=1000)
	S_ = ica.fit(X).transform(X)
	df_S_ =pd.DataFrame(S_)
	df_S_.plot(subplots=True, sharey=True)
	pl.show()

	"""
	fig, axes = pl.subplots(nrows=3, ncols=3, sharey=True)
	under.x.plot(ax=axes[0,0]); #axes[0,0].set_title('Under X')
	under.y.plot(ax=axes[0,1]); #axes[0,1].set_title('Under Y')
	under.z.plot(ax=axes[0,2]); #axes[0,2].set_title('Under Z')

	left.x.plot(ax=axes[1,0]); #axes[1,0].set_title('Left X')
	left.y.plot(ax=axes[1,1]); #axes[1,1].set_title('Left Y')
	left.z.plot(ax=axes[1,2]); #axes[1,2].set_title('Left Z')

	right.x.plot(ax=axes[2,0]); #axes[2,0].set_title('Right X')
	right.y.plot(ax=axes[2,1]); #axes[2,1].set_title('Right Y')
	right.z.plot(ax=axes[2,2]); #axes[2,2].set_title('Right Z')

	pl.show()

	"""
"""
pl.subplot(3, 3, 1)
pl.plot(acce_data["under"][:,0], acce_data["under"][:,1], 'r')
pl.subplot(3, 3, 2)
pl.plot(acce_data["under"][:,0], acce_data["under"][:,2], 'b')
pl.subplot(3, 3, 3)
pl.plot(acce_data["under"][:,0], acce_data["under"][:,3], 'g')

pl.subplot(3, 3, 4)
pl.plot(acce_data["left"][:,0], acce_data["left"][:,1], 'r')
pl.subplot(3, 3, 5)
pl.plot(acce_data["left"][:,0], acce_data["left"][:,2], 'b')
pl.subplot(3, 3, 6)
pl.plot(acce_data["left"][:,0], acce_data["left"][:,3], 'g')

pl.subplot(3, 3, 7)
pl.plot(acce_data["right"][:,0], acce_data["right"][:,1], 'r')
pl.subplot(3, 3, 8)
pl.plot(acce_data["right"][:,0], acce_data["right"][:,2], 'b')
pl.subplot(3, 3, 9)
pl.plot(acce_data["right"][:,0], acce_data["right"][:,3], 'g')

"""

#取りあえず前1000個（50Hz×20秒くらい）

