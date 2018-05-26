import pickle
import torch
import numpy as np
import os

def tanh_apx(input_d,file_name):
	file = open(os.path.join('process_data', file_name + '.pkl'), 'rb')
	x_linspace = pickle.load(file)
	values = pickle.load(file)
	num = len(values)
	output = torch.zeros_like(input_d)
	temp_1 = x_linspace[-1]
	output_1 = torch.where(torch.lt(input_d, temp_1), torch.zeros_like(input_d), torch.ones_like(input_d))
	alpha = 10 * output_1
	input_d = torch.add(input_d, alpha)
	output = torch.add(output, output_1)
	temp_1 = x_linspace[0]
	output_1 = torch.where(torch.lt(input_d, temp_1), torch.ones_like(input_d), torch.zeros_like(input_d))
	alpha = 10 * output_1
	input_d = torch.add(input_d, alpha)
	output = torch.add(output, -1 * output_1)
	for t in range(1, num):
		temp_1 = x_linspace[t]
		temp_2 = values[t-1]
		output_1 = torch.where(torch.lt(input_d, temp_1), torch.ones_like(input_d),
                               torch.zeros_like(input_d))
		alpha = 10 * output_1
		input_d = torch.add(input_d, alpha)
		output = torch.add(temp_2 * output_1, output)
	return output

def selu_apx(input_d,file_name):
	def selu(x,name = "selu"):
		scale = 1.0507009873554804934193349852946
		alpha = 1.6732632423543772848170429916717
		return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)

	file = open(os.path.join('process_data', file_name + '.pkl'), 'rb')
	scale = 1.0507009873554804934193349852946
	alpha_1 = 1.6732632423543772848170429916717
	x_linspace = pickle.load(file)
	values = pickle.load(file)
	num = len(values)
	output = torch.zeros_like(input_d)
	temp_1 = 0
	output_1 = torch.where(torch.lt(input_d, temp_1), torch.zeros_like(input_d), input_d)
	output_temp = torch.where(torch.lt(input_d, temp_1), torch.zeros_like(input_d),
                           torch.ones_like(input_d))
	alpha = 10 * output_temp
	input_d = torch.add(input_d, alpha)
	output = torch.add(output, scale*output_1)
	temp_1 = x_linspace[0]
	output_1 = torch.where(torch.lt(input_d, temp_1), torch.ones_like(input_d), torch.zeros_like(input_d))
	alpha = 10 * output_1
	input_d = torch.add(input_d, alpha)
	output = torch.add(output, -scale*alpha_1* output_1)
	for t in range(1,num):
		temp_1 = x_linspace[t]
		temp_2 = values[t-1]
		output_1 = torch.where(torch.lt(input_d, temp_1), torch.ones_like(input_d),
                            torch.zeros_like(input_d))
		alpha = 10 * output_1
		input_d = torch.add(input_d, alpha)
		output = torch.add(temp_2 * output_1, output)
	return output