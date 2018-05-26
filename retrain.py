import os
import time
# os.popen("python main.py tanh 0 2 1 4 4",'r')
# time.sleep(20)
# os.popen("python main.py tanh 0 2 1 4 5",'r')
# time.sleep(20)
# os.popen("python main.py tanh 0 2 1 4 6",'r')
# time.sleep(20)
#
# os.popen("python main.py tanh 0 2 1 6 4",'r')
# time.sleep(20)
# os.popen("python main.py tanh 0 2 1 6 5",'r')
# time.sleep(20)
# os.popen("python main.py tanh 0 2 1 6 6",'r')
# time.sleep(60)
#
# os.popen("python main.py tanh 0 2 1 8 4",'r')
# time.sleep(60)
# os.popen("python main.py tanh 0 2 1 8 5",'r')
# time.sleep(60)
# os.popen("python main.py tanh 0 2 1 8 6",'r')
# time.sleep(60)

# os.popen("python main.py selu -3.875 0 1 4 4",'r')
# time.sleep(20)
# os.popen("python main.py selu -3.875 0 1 4 5",'r')
# time.sleep(30)
# os.popen("python main.py selu -3.875 0 1 4 6",'r')
# time.sleep(30)
#
# os.popen("python main.py selu -3.875 0 1 6 4",'r')
# time.sleep(30)
# os.popen("python main.py selu -3.875 0 1 6 5",'r')
# time.sleep(30)
# os.popen("python main.py selu -3.875 0 1 6 6",'r')
# time.sleep(60)
#
# os.popen("python main.py selu -3.875 0 1 8 4",'r')
# time.sleep(60)
# os.popen("python main.py selu -3.875 0 1 8 5",'r')
# time.sleep(60)
# os.popen("python main.py selu -3.875 0 1 8 6",'r')
# time.sleep(60)

os.system("python main.py tanh 0 2 1 4 4 -MNIST_retrain=True")

os.system("python main.py selu -3.875 0 1 4 4 -MNIST_retrain=True")
