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

def getdata(source):
	contents = csv.reader(open(source, 'rb'), delimiter=' ')
	lines = []
	for line in contents:
		lines.append(line)
	data = np.array(lines).astype(np.float)
	return data

""
acce_data = {}
acce_data["under"] = getdata('data/under.txt')
acce_data["left"] = getdata('data/left.txt')
acce_data["right"] = getdata('data/right.txt')


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
contents = csv.reader(open('data/under.txt', 'rb'), delimiter=' ')
lines = []
for line in contents:
	lines.append(float(line)

data = np.array(lines)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

pl.plot(t, x, 'r', t, y, 'b', t, z, 'g')
pl.show()

df = pd.DataFrame({'t':data[:,0], 'x': data[:,1], 'y': data[:,2], 'z': data[:,3]})
"""

#取りあえず前1000個（50Hz×20秒くらい）




"""
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 10, n_samples)
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = np.random.normal(size=time.shape)
S = np.c_[s1, s2, s3]
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data

# Mix data
A = np.array([[1, 0.2, 0.2], [0.5, 0.1, 0.2], [0.5, 0.5, 0.5]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA()
S_ = ica.fit(X).transform(X)  # Get the estimated sources
A_ = ica.mixing_  # Get estimated mixing matrix
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

###############################################################################
# Plot results
pl.subplot(2, 1, 1)
pl.plot(X)
pl.title('Observations (mixed signal)')
pl.subplot(2, 1, 2)
pl.plot(S_)
pl.title('ICA estimated sources')
pl.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
pl.savefig('test_3gen.png')
pl.show()
"""