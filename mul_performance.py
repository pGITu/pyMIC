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
stream.sync()
pymic_dot_end = time.time()

print("pymic dot :")
print("--------------------------------------")
offl_c.update_host()
print(offl_c)
print("checksum:", np.sum(offl_c.array))
print("Run time:", pymic_dot_end - pymic_dot_start)
print()
print("--------------------------------------")

c[:] = 0.0
offl_c.update_device()
alpha = 1.0
beta = 0.0
library = device.load_library("libdgemm.so")
pymic_dgemm_start = time.time()
stream.invoke(library.dgemm_kernel,
              offl_a, offl_b, offl_c,
              m, n, k, alpha, beta)
stream.sync()
pymic_dgemm_end = time.time()

print("pymic dgemm :")
print("--------------------------------------")
offl_c.update_host()
print(offl_c)
print("checksum:", np.sum(offl_c.array))
print("Run time:", pymic_dgemm_end - pymic_dgemm_start)
print()
print("--------------------------------------")