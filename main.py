from implementation import implement
import argparse


parser = argparse.ArgumentParser()


parser.add_argument("AF",choices=['tanh','selu','self_define'], help="The activation function you want to implemet(tanh,selu,self_define)",type=str)
parser.add_argument("rang_l", help="The range of the AF you want to implement(left end)",type=float)
parser.add_argument("rang_r", help="The range of the AF you want to implement(right end)",type=float)
parser.add_argument("int_bits", help="The number of bits you want for the integer part of the output",type=int)
parser.add_argument("float_bits", help="The number of bits you want for the decimal part of the output",type=int)
parser.add_argument("i_bits", help="The number of bits you want for the input",type=int)
parser.add_argument("-plot_AF", help="Plot the implemet AF or not",type=bool,default=False)
parser.add_argument("-generate_verilog", help="Generate the verilog file or not",type=bool,default=True)
parser.add_argument("-generate_coe_file", help="Generate the coes file or not(For ROM and kx+b)",type=bool,default=True)
parser.add_argument("-simulate", help="Simulate the implemented AF or not",type=bool,default=False)
parser.add_argument("-MNIST_retrain", help="Retrain MNIST with the AF or not",type=bool,default=False)
parser.add_argument("-CIFAR_retrain", help="Retrain CIFAR with the AF or not",type=bool,default=False)
parser.add_argument("-IMGNET_retrain", help="Retrain IMGNET with the AF or not",type=bool,default=False)
parser.add_argument("-Test_on_Datasets",help="Test the implemented AF on MNIST,CIFAR,IMGNET or not",type=bool,default=False)

if(__name__=='__main__'):
    args = parser.parse_args()
    implement.implemet_AF(args)