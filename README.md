# Activation Function Demo

The "Activation Function Demo" is a application for implement activation function in the mathod propsed in paper:[A](http://pytorch.org/), and eveluat the performance  of it on diffierent datasets.

Untill now the activation function we support as exmaples:

- tanh
- selu
- self_define

And the evaluating datasets:

- MNIST
- CIFAR10
- ImageNet

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
- pywinauto
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
	{tanh,selu,self_define} The activation function you want to implemet(tanh,selu,self_define)
	rang_l                The range of the AF you want to implement(left end)
	rang_r                The range of the AF you want to implement(right end)
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
                        Retrain MNIST with the AF or not
		-CIFAR_retrain CIFAR_RETRAIN
                        Retrain CIFAR with the AF or not
		-IMGNET_retrain IMGNET_RETRAIN
                        Retrain IMGNET with the AF or not
		-Test_on_Datasets TEST_ON_DATASETS
                        Test the implemented AF on MNIST,CIFAR,IMGNET or not
# Examples:
## Implemetation
The default mode will implement the activation function of a specific range with different precisions and gernarate a verilog file of our mathod and coe files for the mathod we compared in the paper.

Implement tanh in [0,2], output: 1 bit for integer part, 6 bits for the decimal part, input: 4 bits

	python main.py tanh 0 2 1 6 4

Simualte the activation function in softwara:

	python main.py tanh 0 2 1 6 4 -Simulate=True

Implement selu in [-3.875,0], output: 1 bit for integer part, 6 bits for the decimal part, input: 4 bits

	python main.py tanh -3.875 0 1 6 4
 
## Evaluation
To evaluation the activation function we implement we gernerate a activation function in pytorch to simulate the effect on the neural network.
Before start we need to download the parameters we trained:

* MNIST:
* CIFAR10:
* ImageNet:

Evaluate the tanh we implement above:

	python main.py tanh 0 2 1 6 4 -Test_on_Datasets=True
## Retrian
To evaluate a self_define actiation function we need to retrain the neural networks. To improve the accuracy we need to retrained the neural networks.

Here we take evaluating the tanh_apx on the ImageNet as an exmaple:

	python main.py tanh 0 2 1 6 4 -IMGNET_retrain=True
# Accuracy
We evaluate the performance of popular dataset and models with tanh and selu of different input/output precision.

The origin accuracy:
<table>
   <tr>
      <td>Datasets</td>
      <td>tanh</td>
      <td>selu</td>
   </tr>
   <tr>
      <td>MNIST</td>
      <td>98.42</td>
      <td>98.42</td>
   </tr>
   <tr>
      <td>CIFAR10</td>
      <td>93.78</td>
      <td>93.78</td>
   </tr>
   <tr>
      <td>ImageNet(top1)</td>
      <td>93.78</td>
      <td>93.78</td>
   </tr>
   <tr>
      <td>ImageNet(top5)</td>
      <td>93.78</td>
      <td>93.78</td>
   </tr>
</table>


After using the approximate AF:
<table>
   <tr>
      <td>AF</td>
      <td>MNIST</td>
      <td>CIFAR10</td>
      <td>ImageNet</td>
   </tr>
   <tr>
      <td>tanh_1_4_4</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.42</td>
   </tr>
   <tr>
      <td>tanh_1_4_5</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.43</td>
   </tr>
   <tr>
      <td>tanh_1_4_6</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.44</td>
   </tr>
   <tr>
      <td>tanh_1_6_4</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.45</td>
   </tr>
   <tr>
      <td>tanh_1_6_5</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.46</td>
   </tr>
   <tr>
      <td>tanh_1_6_6</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.47</td>
   </tr>
   <tr>
      <td>tanh_1_8_4</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.48</td>
   </tr>
   <tr>
      <td>tanh_1_8_5</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.49</td>
   </tr>
   <tr>
      <td>tanh_1_8_6</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.50</td>
   </tr>
   <tr>
      <td>selu_1_4_4</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.42</td>
   </tr>
   <tr>
      <td>selu_1_4_5</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.43</td>
   </tr>
   <tr>
      <td>selu_1_4_6</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.44</td>
   </tr>
   <tr>
      <td>selu_1_6_4</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.45</td>
   </tr>
   <tr>
      <td>selu_1_6_5</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.46</td>
   </tr>
   <tr>
      <td>selu_1_6_6</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.47</td>
   </tr>
   <tr>
      <td>selu_1_8_4</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.48</td>
   </tr>
   <tr>
      <td>selu_1_8_5</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.49</td>
   </tr>
   <tr>
      <td>selu_1_8_6</td>
      <td>98.42</td>
      <td>98.42</td>
      <td>98.42/98.50</td>
   </tr>
</table>

