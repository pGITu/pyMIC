import pymic as mic
import numpy as np
import time
import sys

def sum(operand1, operand2, axis=None, dtype=None) :
	#if operand1.dtype != operand1.astype(np.float32) or operand2.dtype != operand2.astype(np.float32):
	#	return -1
	device = mic.devices[0]
	dt = map_data_types(operand1.dtype)
	n = operand1.size
	library = device.load_library("libmic.so")
	stream = device.get_default_stream()
	result = np.zeros((m, n))
	offl_o1 = stream.bind(operand1)
	offl_o2 = stream.bind(operand2)	
	offl_r = stream.bind(result)
	stream_invoke (library.pymic_offload_array_sum, dt, n, offl_o1, offl_r)
	offl_r.update_host()
	stream.sync()
	print (result)
	return result

sz = 50
m, n, k = sz, sz, sz
a = np.random.random(m * k).reshape((m, k))
b = np.random.random(k * n).reshape((k, n))
test = sum(a,b)


	











		




































































































































