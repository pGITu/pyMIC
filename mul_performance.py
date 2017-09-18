import pymic as mic
import numpy as np
import time
import sys

if len(sys.argv) > 1 :
	sz = int(sys.argv[1])
	m, n, k = sz, sz, sz
else :
	sz = 4096
	m, n, k = sz, sz, sz
np.random.seed(10)
a = np.random.random(m * k).reshape((m, k))
b = np.random.random(k * n).reshape((k, n))
c = np.zeros((m, n))


device = mic.devices[0]
stream = device.get_default_stream()
offl_a = stream.bind(a)
offl_b = stream.bind(b)
offl_c = stream.bind(c)

pymic_dot_start = time.time()
offl_c = offl_a * offl_b
offl_c.update_host()
stream.sync()
pymic_dot_end = time.time()

print("pymic dot :")
print("--------------------------------------")
print(offl_c)
print("checksum:", np.sum(offl_c.array))
print("Run time:", pymic_dot_end - pymic_dot_start)
print()
print("--------------------------------------")