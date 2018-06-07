# Activation Function Demo

The "Activation Function Demo" is a demo for implementing activation function with the mathod, propsed in paper:[A](http://pytorch.org/), and evaluating the performance of it with different precision on diffierent datasets. And here is an example that we implemeted:
![](https://bbojia.bn.files.1drv.com/y4mCLg2iVujqTxYebloAkMe-yBDgGTcfmBrotdGRWZnFfitbOnjpIe_zM5b3IJYdPaxrl9KWJAxiTCem2FOlkoE9JdpfeUOXro4gXIAC0R5Tbr9sWsHWjV2gswxNFj_Uect5wUF2FAk9yJ7kqRvhqkg7nbjC3XpO6o-MPY-I3wOja2C0DXedq1EhDZ3bCeXmQiIhqrIgl2yVJW_EMoQsCgMng/af.png?psid=1)

Until now the activation function we supported:

- tanh
- selu
- self_define

And the evaluation datasets:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](http://www.imagenet.stanford.edu/challenges/LSVRC/2012/nonpub-downloads)
### License

Activation Function Demo is released under the MIT License (refer to the LICENSE file for details).

# Requirements
For implementation

* Windows

For training and test

* Linux
* Windows
# Install
- pytorch (>=0.4.0) and torchvision from [official website](http://pytorch.org/), for example, cuda8.0 for python3.5
    - `pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl`
    - `pip install torchvision`
- numpy
    - `pip install numpy`
- pywinauto(For windows)
    - `pip install pywinauto`
# Usage
The Arguements

    usage: main.py [-h] [-plot_AF PLOT_AF] [-generate_verilog GENERATE_VERILOG]
               [-generate_coe_file GENERATE_COE_FILE] [-simulate SIMULATE]
               [-MNIST_retrain MNIST_RETRAIN] [-CIFAR_retrain CIFAR_RETRAIN]
               [-IMGNET_retrain IMGNET_RETRAIN]
               [-Test_on_Datasets TEST_ON_DATASETS]
               {tanh,selu,self_define} rang_l rang_r int_bits float_bits
               i_bits

	positional arguments:
	{tanh,selu,self_define} The activation function you want to implement(tanh,selu,self_define)
	rang_l                The range of the AF you want to implement(left endpoint)
	rang_r                The range of the AF you want to implement(right endpoint)
	int_bits              The number of bits you want for the integer part of
                        the output
	float_bits            The number of bits you want for the decimal part of
                        the output
	i_bits                The number of bits you want for the input
	optional arguments:
		-h, --help            show this help message and exit
		-plot_AF PLOT_AF      Plot the implemet AF or not
		-generate_verilog GENERATE_VERILOG
                        Generate the verilog file or not
		-generate_coe_file GENERATE_COE_FILE
                        Generate the coes file or not(For ROM and kx+b)
		-simulate SIMULATE    Simulate the implemented AF or not
		-MNIST_retrain MNIST_RETRAIN
                        Retrain ANN with the AF on MNIST  or not
		-CIFAR_retrain CIFAR_RETRAIN
                        Retrain ANN with the AF on CIFAR-10  or not
		-IMGNET_retrain IMGNET_RETRAIN
                        Retrain ANN with the AF on IMGNET  or not
		-Test_on_Datasets TEST_ON_DATASETS
                        Test the implemented AF on MNIST,CIFAR,IMGNET or not
# Examples:
## Implemetation
The default mode of it can implement the activation function of a specific range with different precisions and generate a verilog file of our method and coe files of the methods we compared in the paper.

Implement tanh in [0,2], output: 1 bit for integer part, 6 bits for the decimal part, input: 4 bits

	python main.py tanh 0 2 1 6 4

Simulate the activation function in software and plot the figure of it:

	python main.py tanh 0 2 1 6 4 -Simulate=True

Implement selu in [-3.875,0], output: 1 bit for integer part, 6 bits for the decimal part, input: 4 bits

	python main.py tanh -3.875 0 1 6 4

After Implementation, you can find the verilog file in path:AF\_implementation\\verilog_file and the name rule is:

AF\_(integer bit width of outputs)\_(decimal bit width of outputs)\_(input bit width).v

Example: tanh\_1\_4\_4.v



Here is an verilog file example:
![](https://bbojia.bn.files.1drv.com/y4mRU5CS-7-kBB9xfljALBGy4s9oe6X5obp3EjI2NFqt9RKMtYUpFD6RhTWHMDg2MajJ57EAXmOdfj60QHNiTOxWiCB3DY9Ve-YVEgUjuhzzSWZm_PCh7YmB-T8PhkcO_lM_Hm8Nio9cSj_Ukyrxsm9qrm6WUCDDBOrots0lZj35nZPRbbg_PCK2WX5PThtVLb382eQ4eepcv1boPH5-dcc-g/verilog.png?psid=1)
And also you will get three coe files at AF\_implementation\\coe_file\\tanh\_1\_6\_4:y.coe, b.coe, k.coe, Here is an examples of coe files:
![](https://bbojia.bn.files.1drv.com/y4mDkmbSBtQIW6lV1BfCATs7BI5Hx0IqBZEsM9bSj7GuY6N8r0GlyGgmz1c53UUcxXBZBJGOyt33ErH9zLnS-UtsKhHN3365KdoXAnxcYw06taUEr_viOU0W5tvyEALYdanz2nQy9o5KCM3vPXpyeWeTZyHGVP0PNmyG03tsgwYplgSnyzd_ESVIM_h4pF-Tkz61vyR-njnqxaplr7W9Uyraw/coe.png?psid=1)
 
## Evaluation
To evaluation the activation function we implement we gernerate a simulate version on activation function in pytorch to simulate the effect on the neural network. Before start we need to download the parameters we trained:

* [MNIST](#mnist)
* [CIFAR-10](#cifar-10)
* [ImageNet](#imagenet):

Evaluate the tanh we implement above:

	python main.py tanh 0 2 1 6 4 -Test_on_Datasets=True
Attention:You need to edit the dataset path in the NN_models/IMG\_NET\_tanh.py(IMG\_NET\_selu.py,here IMG\_NET\_tanh.py is an example, each of them need to be edited), so that it can find the evaluation datasets.
## Retrian
To evaluate a self define activation function we need to retrain the neural networks. To improve the accuracy we still need to retrain the neural networks. So we also provide the function of retraining.

Here we take evaluating the tanh_apx on the ImageNet as an exmaple:

	python main.py tanh 0 2 1 6 4 -IMGNET_retrain=True
Attention:You need to edit the dataset path in files: NN\_models/IMG\_NET\_tanh.py(IMG\_NET\_selu.py,here IMG\_NET\_tanh.py is an example, each of them need to be edited), so that it can find the training datasets.
# Accuracy
We evaluate the performance of tanh and selu of different input/output precision on popular dataset and models.


We trained a LeNet-5 on MNIST of  which  the AFs were all replaced with tanh/ReLU,and trained a vgg-16 on CIFAR-10 of which  the AFs were all replaced with tanh/ReLU, Then an alexnet was trained on ImageNet of which  the AFs were all replaced with tanh/ReLU. Then we get the following origin accuracy:
<table>
   <tr>
      <td>origin</td>
      <td>MNIST</td>
      <td>CIFAR</td>
      <td>ImageNet</td>
   </tr>
   <tr>
      <td>tanh</td>
      <td>96.15</td>
      <td>87.17</td>
      <td>42.392/67.614</td>
   </tr>
   <tr>
      <td>SeLU</td>
      <td>97.67</td>
      <td>86.79</td>
      <td>39.260/63.342</td>
   </tr>
</table>


Then we replaced the AF of these models with the AF we implemented, validate the models on the test set, and get the following accuracies, but we find that accuracy loss was huge nearly destroied the models ability  on Imagenet. This is due to error accumulatting, thus, we use some training tricks to increase it. We retraining the models with the AF we implemented, in this way, tanh get a enormous increase, but Selu still very low, then we add BNs (batch norm) in the models, in this way, we can reduce the accuracy loss.
<table>
   <tr>
      <td>AF</td>
      <td>MNIST</td>
      <td>CIFAR</td>
      <td>ImageNet</td>
      <td colspan="3" align="center">ImageNet(retrain)</td>
   </tr>
   <tr>
      <td>tanh_1_4_4</td>
      <td>95.95</td>
      <td>82.1</td>
      <td>8.038/23.662</td>
      <td colspan="3" align="center">35.160/59.608</td>
   </tr>
   <tr>
      <td>tanh_1_4_5</td>
      <td>96.02</td>
      <td>58.47</td>
      <td>7.668/22.882</td>
      <td colspan="3" align="center">35.040/59.332</td>
   </tr>
   <tr>
      <td>tanh_1_4_6</td>
      <td>96.06</td>
      <td>70.09</td>
      <td>7.388/22.400</td>
      <td colspan="3" align="center">34.792/59.096</td>
   </tr>
   <tr>
      <td>tanh_1_6_4</td>
      <td>96.1</td>
      <td>85.21</td>
      <td>6.882/21.330 </td>
      <td colspan="3" align="center">34.562/59.196</td>
   </tr>
   <tr>
      <td>tanh_1_6_5</td>
      <td>96.17</td>
      <td>86.64</td>
      <td>6.562/20.432</td>
      <td colspan="3" align="center">34.294/58.756</td>   
   </tr>
   <tr>
      <td>tanh_1_6_6</td>
      <td>96.19</td>
      <td>86.88</td>
      <td>6.296/19.96</td>
      <td colspan="3" align="center">34.238/58.672</td>  
   </tr>
   <tr>
      <td>tanh_1_8_4</td>
      <td>96.13</td>
      <td>84.92</td>
      <td>6.712/20.792</td>
      <td colspan="3" align="center">34.442/58.956</td>
   </tr>
   <tr>
      <td>tanh_1_8_5</td>
      <td>96.2</td>
      <td>86.99</td>
      <td>6.278/19.778</td>
      <td colspan="3" align="center">34.300/58.658</td>
   </tr>
   <tr>
      <td>tanh_1_8_6</td>
      <td>96.23</td>
      <td>87.15</td>
      <td>5.940/19.128</td>
      <td colspan="3" align="center">34.132/58.522</td>
   </tr>
   <tr>
      <td>SeLU_1_4_4</td>
      <td>97.71</td>
      <td>71.71</td>
      <td>0.242/0.882</td>
      <td><font color=red>39.138/63.800</font></td>
      <td>16.500/33.840</td>
      <td>37.910/62.904</td>
   </tr>
   <tr>
      <td>SeLU_1_4_5</td>
      <td>97.69</td>
      <td>82.64</td>
      <td>0.284/0.916</td>
      <td>2.622/6.938</td>
      <td><font color=red>46.312/70.560</font></td>
      <td>40.466/65.316</td>
   </tr>
   <tr>
      <td>SeLU_1_4_6</td>
      <td>97.65</td>
      <td>86.1</td>
      <td>0.302/0.944</td>
      <td>0.608/2.218</td>
      <td>30.226/53.308</td>
      <td><font color=red>40.774/65.448</font></td>
   </tr>
   <tr>
      <td>SeLU_1_6_4</td>
      <td>97.68</td>
      <td>71.98</td>
      <td>0.242/0.878</td>
      <td><font color=red>39.256/64.208</font></td>
      <td>15.344/32.028</td>
      <td>37.460/62.534</td>
   </tr>
   <tr>
      <td>SeLU_1_6_5</td>
      <td>97.68</td>
      <td>82.87</td>
      <td>0.284/0.87</td>
      <td>2.652/7.142</td>
      <td><font color=red>46.074/70.466</font></td>
      <td>40.098/65.070</td>
   </tr>
   <tr>
      <td>SeLU_1_6_6</td>
      <td>97.65</td>
      <td>85.51</td>
      <td>0.308/0.892</td>
      <td>0.858/2.6</td>
      <td>35.492/59.952</td>
      <td><font color=red>40.762/65.506</font></td>
   </tr>
   <tr>
      <td>SeLU_1_8_4</td>
      <td>97.66</td>
      <td>71.99</td>
      <td>0.250/0.878</td>
      <td><font color=red>39.628/64.62</font></td>
      <td>15.338/31.980</td>
      <td>37.462/62.440</td>
   </tr>
   <tr>
      <td>SeLU_1_8_5</td>
      <td>97.66</td>
      <td>82.32</td>
      <td>0.296/0.9</td>
      <td>2.948/7.368</td>
      <td><font color=red>46.290/70.534</font></td>
      <td>39.984/65.036</td>
   </tr>
   <tr>
      <td>SeLU_1_8_6</td>
      <td>97.67</td>
      <td>85.11</td>
      <td>0.328/0.844</td>
      <td>0.846/2.644</td>
      <td>36.922/61.404</td>
      <td><font color=red>40.786/65.452</font></td>
   </tr>
</table>
The highest accuracy was noted by red.

The AF name rule above is:

AF\_(integer bit width of outputs)\_(decimal bit width of outputs)\_(input bit width)
# Download Parameters

## MNIST:

Please download the following files and put them in:AF\_implementation\\NN\_models\\MNIST\_data before evaluating your activation functions

* tanh:<https://1drv.ms/u/s!AhWdKGJb0BiJcJzuhsP_RIB9zPA>
* SeLU:<https://1drv.ms/u/s!AhWdKGJb0BiJb0VJkTLqin7ICvY>

## CIFAR-10:

Please download the following files and put them in:AF\_implementation\\NN\_models\\CIFAR\_data before evaluating your activation functions

* tanh:<https://1drv.ms/u/s!AhWdKGJb0BiJcdxTBqxubJDc_Gw>
* SeLu:<https://1drv.ms/u/s!AhWdKGJb0BiJdaKh7s-e3mmtVUc>

## ImageNet:

Please download the following files and put them in:AF\_implementation\\NN\_models\\IMGNET\_data before evaluating your activation functions

Unretrained:

code:<https://1drv.ms/u/s!AhWdKGJb0BiJd1inN2l2nnAh8z8>

The default code in the demo is for retrained mode, so this file need to be extracted to path:NN\_models/

* tanh:<https://1drv.ms/u/s!AhWdKGJb0BiJdhLxPNtwoWaIHMw>
* SeLU:<https://1drv.ms/u/s!AhWdKGJb0BiJbiU19WakQ4tHyrk>

This has an enormous accuracy loss, thus we retrain the AlexNet with the AF\_apx(approximate AFs)

Retrained:

* tanh:<https://1drv.ms/u/s!AhWdKGJb0BiJbYP_z8mfUOecyr4>
* SeLU_4:<https://1drv.ms/u/s!AhWdKGJb0BiJfPoaIL2VCqCSlvo>
* SeLU_5:<https://1drv.ms/u/s!AhWdKGJb0BiJeuh1tcRBqAvWtBU>
* SeLU_6:<https://1drv.ms/u/s!AhWdKGJb0BiJe5KHi1bFRzFFWww>

Where, SeLU_4,SeLU_5,SeLU_6 is the parameters file of retraining AlexNet with SeLU\_1\_4\_4, SeLU\_1\_4\_5,SeLU\_1\_4\_6.
# Reference
1. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification
with Deep Convolutional Neural Networks,” in Advances in Neural
Information Processing Systems 25, F. Pereira, C. J. C. Burges, L. Bottou,
and K. Q. Weinberger, Eds. Curran Associates, Inc., 2012, p.
1097?1105.
2. G. Hinton, L. Deng, D. Yu, G. Dahl, A. rahman Mohamed, N. Jaitly,
A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury,
“Deep neural networks for acoustic modeling in speech recognition,”
Signal Processing Magazine, 2012.
3. M. Luong, H. Pham, and C. D. Manning, “Effective approaches to
attention-based neural machine translation,” CoRR, vol. abs/1508.04025,
2015.
