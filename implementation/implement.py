import platform
import numpy as np
import matplotlib.pyplot as plt
import pickle
import NN_models.MNIST_tanh as M_tanh
import NN_models.MNIST_selu as M_selu
import NN_models.MNIST_self_define as M_self_define
import NN_models.IMG_NET_tanh as I_tanh
import NN_models.IMG_NET_selu as I_selu
import NN_models.IMG_NET_self_define as I_self_define
import NN_models.CIFAR10_tanh as C_tanh
import NN_models.CIFAR10_selu as C_selu
import NN_models.CIFAR10_self_define as C_self_define
import os
if(platform.system() == 'Windows'):
    from pywinauto.application import Application
#here to add your own AF

def self_define(x):
    x = np.array(x)
    return x


def selu(x, name="selu"):
    x = np.array(x)
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * np.where(x > 0.0, x, alpha * np.exp(x) - alpha)

def trans_value(value,int_bits,float_bits):
    value_new=0.0
    if(value>0):
        sign = 1
    else:
        sign = -1
    value = abs(value)
    int_part = int(value)
    float_part = value - int_part
    bin_list = []
    for i in range(int_bits):
        int_part,bin = divmod(int_part,2)
        if (bin == 1):
            bin_list.append(1)
            value_new = value_new + 2 ** (i)
        else:
            bin_list.append(0)
    for i in range(float_bits):
        float_part = float_part * 2
        if(float_part >= 1):
            bin_list.append(1)
            float_part = float_part - 1
            value_new = value_new + 2 ** (-i - 1)
        else:
            bin_list.append(0)

    return bin_list,sign*value_new

def aproxi_AF(AF,rang,int_bits,float_bits,i_bits):
    bit_map = []
    value_list = []
    l,r = rang
    x_linspace = [l+k*((r-l)/(2**i_bits)) for k in range(2**i_bits)]
    values = AF(x_linspace)
    for value in values:
        bit_list,t_value = trans_value(value,int_bits,float_bits)
        bit_map.append(bit_list)
        value_list.append(abs(t_value))
    bit_map = np.array(bit_map)
    return bit_map,value_list


# Run a target application

def get_expression(min,bits,a):
    a['Edit4'].TypeKeys(str(bits-1))
    for i,trem in enumerate(min):
        a['Edit3'].TypeKeys(trem)
        if(i != len(min)-1):
            a['Edit3'].type_keys('+=')
    a['as number'].Click()
    a['Confirm'].Click()
    express = a['Edit2'].WindowText()
    a['Empty'].Click()
    return express

def get_expressions(bit_map):
    eps = []
    app = Application().start(os.path.join('implementation','Logic Functions Minimization.exe'))
    a = app.window()
    size,bits = bit_map.shape
    i_bits = np.log2(size)
    for i in range(bits):
        if(np.all(bit_map[:,i]==0)==False):
            min = np.where(bit_map[:,i]==1)[0].tolist()
            ep = get_expression(min,i_bits,a)
            eps.append(ep)
        else:
            ep = 'the %d line is all 0'%i
            eps.append(ep)
    a['Quit'].Click()
    return eps
def save_eps(eps,file_name):
    with open(os.path.join('process_data',file_name+'.txt'),'w+') as f:
        for ep in eps:
            f.write(ep)
            f.write('\n')
def read_eps(file_name):
    eps = []
    with open(os.path.join('process_data',file_name+'.txt'),'r') as f:
        for line in f.readlines():
            eps.append(line)
    return eps

def simulate_eps(ep,x_bin):
    dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
    x_map = np.array([[int(s) for s in bin_s] for bin_s in x_bin])
    or_bins = []
    # print(x_map)
    for it in ep:
        bins = []
        for i in it:
            if(i == '\''):
                temp=1-temp
                bins.pop()
                bins.append(temp)
            else:
                temp = x_map[:,dict[i]]
                bins.append(temp)
        it_v = [1 for o in range(len(x_bin))]
        for bin in bins:
            it_v = it_v&bin
        or_bins.append(it_v)
    or_v = [0 for o in range(len(x_bin))]
    for bin in or_bins:
        or_v = or_v|bin
    return or_v

def simulate(eps,rang,int_bits,float_bits,i_bits):
    l, r = rang
    x_linspace = [l+k*((r-l)/(2**i_bits)) for k in range(2**i_bits)]
    x_s = [k for k in range(2**i_bits)]
    x_bin = ['0'*(i_bits - len(bin(x)[2:]))+bin(x)[2:] for x in x_s]
    y_bin = []
    for i,ep in enumerate(eps):
        if(ep != 'the %d line is all 0\n'%i):
            ep = ep.replace('\n','')
            ep = ep.split('+')
            bit_list = simulate_eps(ep,x_bin)
            y_bin.append(bit_list)
    v = np.zeros(len(x_bin))
    for i,y in enumerate(y_bin):
        v = v + y*2 ** (-i - 1)
    plt.figure(1)
    plt.plot(x_linspace,v)
    plt.show()
def generate_ep(ep):
    dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
    mep = ''
    for j,it in enumerate(ep):
        mep = mep + '('
        items = []
        for i in it:
            if (i == '\''):
                temp = items.pop()
                items.append('~' + temp)
            else:
                items.append('x['+str(dict[i])+']')
        for k,item in enumerate(items):
            mep = mep + item
            if(k!=len(items)-1):
                mep = mep + '&'
        mep = mep + ')'
        if (j != len(ep) - 1):
            mep = mep + '|'
    return mep



def generate_verilog(eps,file_name,int_bits,float_bits,i_bits):
    with open(os.path.join('verilog_file',file_name+'.v'),'w+') as f:
        f.write('module '+ file_name+'(x,y);\n')
        f.write('input wire['+str(i_bits-1)+':0]x;\n')
        f.write('output wire['+str(float_bits) +':0]y;\n')
        for i,ep in enumerate(eps):
            if (ep != 'the %d line is all 0\n' % i):
                ep = ep.replace('\n', '')
                ep = ep.split('+')
                ep_v = generate_ep(ep)
                f.write('assign '+'y['+str(i)+']=')
                f.write(ep_v + ';\n')
            else:
                f.write('assign ' + 'y[' + str(i) + ']=')
                f.write('0' + ';\n')
        f.write('endmodule')

def generate_coe(AF,rang,i_bits,int_bits,float_bits,file_name):
    l,r = rang
    step = 2**(-i_bits)*2
    bits = int_bits + float_bits + 1
    x_linspace = np.arange(l, r - step, step)
    if(AF == 'selu'):
        y = selu(x_linspace)
    elif(AF == 'tanh'):
        y = np.tanh(x_linspace)
    else:
        y = self_define(x_linspace)
    k = []
    b = []
    for i in range(len(x_linspace)-1):
        bit_list,value = trans_value((y[i+1] - y[i])/(x_linspace[i+1] - x_linspace[i]), int(bits/2), int(bits/2))
        bit_str = ''
        bit_str = bit_str.join(str(k) for k in bit_list)
        k += [bit_str]
        bit_list, value = trans_value((x_linspace[i+1]*y[i] - x_linspace[i]*y[i+1])/(x_linspace[i+1] - x_linspace[i]), int(bits/2), int(bits/2))
        bit_str = ''
        bit_str = bit_str.join(str(k) for k in bit_list)
        b += [bit_str]
    y_v = []
    for v in y:
        bit_list, value = trans_value(v,int_bits,float_bits)
        bit_str = ''
        bit_str = bit_str.join(str(k) for k in bit_list)
        y_v += [bit_str]
    if(os.path.exists(os.path.join('coe_file',file_name)) == False):
        os.makedirs(os.path.join('coe_file',file_name))
    with open(os.path.join('coe_file',file_name,'y.coe'),'w+') as f:
        f.write('memory_initialization_radix = 2;\n')
        f.write('memory_initialization_vector = ')
        for i,it in enumerate(y_v):
            if (i%3 == 0):
                f.write('\n')
            f.write(str(it))
            if (i == len(y_v) - 1):
                f.write(';')
            else:
                f.write(',')
    with open(os.path.join('coe_file',file_name,'k.coe'),'w+') as f:
        f.write('memory_initialization_radix = 2;\n')
        f.write('memory_initialization_vector = ')
        for i,it in enumerate(k):
            if (i%3 == 0):
                f.write('\n')
            f.write(str(it))
            if (i == len(k) - 1):
                f.write(';')
            else:
                f.write(',')
    with open(os.path.join('coe_file',file_name,'b.coe'),'w+') as f:
        f.write('memory_initialization_radix = 2;\n')
        f.write('memory_initialization_vector = ')
        for i,it in enumerate(b):
            if (i%3 == 0):
                f.write('\n')
            f.write(str(it))
            if(i == len(b)-1):
                f.write(';')
            else:
                f.write(',')

def MNIST_retrain(args):
    file_name = args.AF + '_' + str(args.int_bits) + '_' + str(args.float_bits) + '_' + str(args.i_bits)
    if(args.AF == 'tanh'):
        M_tanh.train_test(True,file_name)
    if(args.AF == 'selu'):
        M_selu.train_test(True,file_name)
    if (args.AF == 'self_define'):
        M_self_define.train_test(True,self_define,file_name)
def CIFAR_retrain(args):
    file_name = args.AF + '_' + str(args.int_bits) + '_' + str(args.float_bits) + '_' + str(args.i_bits)
    if (args.AF == 'tanh'):
        C_tanh.train_test(True, file_name)
    if (args.AF == 'selu'):
        C_selu.train_test(True, file_name)
    if (args.AF == 'self_define'):
        C_self_define.train_test(True, file_name)
def IMGNET_retrain(args):
    file_name = args.AF + '_' + str(args.int_bits) + '_' + str(args.float_bits) + '_' + str(args.i_bits)
    if (args.AF == 'tanh'):
        I_tanh.train_test(True,file_name)
    if (args.AF == 'selu'):
        I_selu.train_test(True,file_name)
    if (args.AF == 'self_define'):
        I_self_define.train_test(True,file_name)
def test_on_datates(args):
    file_name = args.AF + '_' + str(int_bits) + '_' + str(float_bits) + '_' + str(i_bits)
    if (args.AF == 'tanh'):
        M_tanh.train_test(False,file_name)
        # C_tanh.train_test(False,file_name)
        # I_tanh.train_test(False,file_name)
    if (args.AF == 'selu'):
        M_selu.train_test(False,file_name)
        # C_selu.train_test(False,file_name)
        # I_selu.train_test(False,file_name)
    if (args.AF == 'self_define'):
        M_self_define.train_test(False,file_name)
        C_self_define.train_test(False,file_name)
        I_self_define.train_test(False,file_name)
def implemet_AF(args):
    i_bits = args.i_bits
    int_bits = args.int_bits
    float_bits = args.float_bits
    AF = args.AF
    rang = (args.rang_l,args.rang_r)
    file_name = AF + '_' + str(int_bits) + '_' + str(float_bits) + '_' + str(i_bits)
    l, r = rang
    if(args.MNIST_retrain == True):
        MNIST_retrain(args)
    elif(args.CIFAR_retrain == True):
        CIFAR_retrain(args)
    elif(args.IMGNET_retrain == True):
        IMGNET_retrain(args)
    else:
        if (platform.system() == 'Windows'):
            if (AF == 'selu'):
                bit_map, valus = aproxi_AF(selu, rang, int_bits, float_bits, i_bits)
            if (AF == 'tanh'):
                bit_map, valus = aproxi_AF(np.tanh, rang, int_bits, float_bits, i_bits)
            if (AF == 'self_define'):
                bit_map, valus = aproxi_AF(self_define, rang, int_bits, float_bits, i_bits)
            x_linspace = [l + k * ((r - l) / (2 ** i_bits)) for k in range(2 ** i_bits)]
            eps = get_expressions(bit_map)
            save_eps(eps, file_name)
            if(AF == 'selu'):
                copy = [-1 * x for x in reversed(x_linspace)]
                x_linspace = x_linspace + [-1 * x for x in reversed(x_linspace)]
                valus = [-1 * x for x in valus] + copy
            if(AF == 'tanh'):
                x_linspace = [-1 * x for x in reversed(x_linspace)] + x_linspace
                valus = [-1 * x for x in reversed(valus)] + valus
            if (AF == 'self_define'):
                x_linspace = [-1 * x for x in reversed(x_linspace)] + x_linspace
                valus = [-1 * x for x in reversed(valus)] + valus
            output = open(os.path.join('process_data',file_name+'.pkl'), 'wb')
            pickle.dump(x_linspace, output)
            pickle.dump(valus, output)
            output.close()
            if(args.plot_AF == True):
                plt.figure(1)
                plt.plot(x_linspace, valus)
                plt.show()
            eps = read_eps(file_name)
            if(args.generate_verilog == True):
                generate_verilog(eps,file_name,int_bits,float_bits,i_bits)
            if(args.generate_coe_file == True):
                generate_coe(AF, rang, i_bits, int_bits, float_bits,file_name)
            if(args.simulate == True):
                simulate(eps,rang,int_bits,float_bits,i_bits)
            if(args.Test_on_Datasets == True):
                test_on_datates(args)
        else:
            if (args.Test_on_Datasets == True):
                test_on_datates(args)
            print('The linux system is only for training neural networks')