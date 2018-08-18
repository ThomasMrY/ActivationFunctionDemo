# Activation Function Demo

The "Activation Function Demo" is a demo for implementing activation function with the mathod, propsed in paper:[A](http://pytorch.org/), and evaluating the performance of it with different precision on diffierent datasets. And here is an example that we implemeted:
![](http://thyrsi.com/t6/359/1534598671x-1376440150.png)

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
![](http://thyrsi.com/t6/359/1534598719x-1404775491.png)
And also you will get three coe files at AF\_implementation\\coe_file\\tanh\_1\_6\_4:y.coe, b.coe, k.coe, Here is an examples of coe files:
![](http://thyrsi.com/t6/359/1534598750x-1404775491.png)
 
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
  <td></td>
  <td>MNIST</td>
  <td>CIFAR-10</td>
  <td>ImageNet(top1/top5)</td>
 </tr>
 <tr>
  <td>Tanh(Original)</td>
  <td>96.15%</td>
  <td>87.17%</td>
  <td>42.39%/67.61%</td>
 </tr>
 <tr>
  <td>Tanh_5_4</td>
  <td>-0.2%</td>
  <td>-5.07%</td>
  <td>-8.16%/-8.94%</td>
 </tr>
 <tr>
  <td>Tanh_7_4</td>
  <td>-0.05%</td>
  <td>-1.96%</td>
  <td>-7.83%/-8.42%</td>
 </tr>
 <tr>
  <td>Tanh_7_6</td>
  <td>+0.04%</td>
  <td>-0.29%</td>
  <td>-7.23%/-8.0%</td>
 </tr>
 <tr>
  <td>SeLU(Original)</td>
  <td>97.67%</td>
  <td>86.79%</td>
  <td>39.260%/63.342%</td>
 </tr>
 <tr>
  <td>SeLU_5_4</td>
  <td>+0.04%</td>
  <td>-4.15%</td>
  <td>-0.122%/+0.458%</td>
 </tr>
 <tr>
  <td>SeLU_7_4</td>
  <td>+0.01%</td>
  <td>-4.47%</td>
  <td>-0.004%/+0.866%</td>
 </tr>
 <tr>
  <td>SeLU_8_5</td>
  <td>+0.37%</td>
  <td>-0.69%</td>
  <td>+0.368%/+1.278%</td>
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
attention-based neural machine translation,” CoRR, vol. abs/1508.04025,2015.
