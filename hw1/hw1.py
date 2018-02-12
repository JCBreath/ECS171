import numpy as np
import math, random

# N = 10,000 and d = 780
N = 10000
d = 780

nreps = 171

X_train=[]
y_train=[]
X_test=[]
y_test=[]


def get_accuracy(W):
	# get test file lines
	test_count = np.size(X_test) / d

	correct_count = 0

	for i in range(0,test_count):
		if np.sign(np.dot(W, X_test[i])) == y_test[i]:
			correct_count += 1

	accuracy = float(correct_count) / test_count

	print("Prediction accuracy: " + str(accuracy))

def GradientDescent(step_size):
	W = np.zeros(d)

	for i in range(1,nreps):
		gradient = np.zeros(d)

		for n in range(0, N):
			gradient += y_train[n]*X_train[n] / (1 + np.exp(y_train[n]*np.dot(W, X_train[n])))

		gradient = gradient / N

		W+=step_size*gradient


	# f(w)
	f_w = 0

	for n in range(0, N):
		f_w += np.log(1 + np.exp(-1*y_train[n]*np.dot(W,X_train[n])))

	f_w = f_w/N
	print("f(w)=" + str(f_w))
	
	print("(b)")
	get_accuracy(W)

def Stochastic(step_size):
	W = np.zeros(d)

	for i in range(1,nreps):
		gradient = np.zeros(d)

		batch_size = random.randint(1,N)

		for j in range(0,batch_size):
			n = random.randint(0,N - 1)
			gradient += y_train[n]*X_train[n] / (1 + np.exp(y_train[n]*np.dot(W, X_train[n])))
		
		gradient = gradient / batch_size
		W+=step_size*gradient

	get_accuracy(W)

def StochasticDecay(step_size, decay_rate, decay_value):
	W = np.zeros(d)

	for i in range(1,nreps):
		gradient = np.zeros(d)

		batch_size = random.randint(1,N)

		for j in range(0,batch_size):
			n = random.randint(0,N - 1)
			gradient += y_train[n]*X_train[n] / (1 + np.exp(y_train[n]*np.dot(W, X_train[n])))

		gradient = gradient / batch_size
		
		if i % decay_rate == 0:
			step_size -= step_size * decay_value
		
		W+=step_size*gradient

	get_accuracy(W)

def Problem4(step_size):
	W = np.zeros(d)

	for i in range(1,nreps):
		gradient = np.zeros(d)

		batch_size = random.randint(1,N)
		begin = random.randint(0,N - 1)
		end = min(begin+batch_size, N)
		for n in range(begin,end):
		
			gradient_change = 1 - y_train[n]*np.dot(W, X_train[n])
			if gradient_change > 0:
				gradient += y_train[n]*X_train[n]

		gradient = gradient / batch_size
		
		W+=step_size*gradient

	# f(w)
	f_w = 0

	for n in range(0, N):
		f_w += np.log(1 + np.exp(-1*y_train[n]*np.dot(W,X_train[n])))

	f_w = f_w/N
	print("f(w)=" + str(f_w))

	get_accuracy(W)

FILE="./mnist_2_vs_7/mnist_X_train.dat"
with open(FILE) as X_train_file:
	for line in X_train_file:
		X_train.append([int(i) for i in line.split(' ')])

FILE="./mnist_2_vs_7/mnist_y_train.dat"
with open(FILE) as y_train_file:
	for line in y_train_file:
		y_train.append([int(i) for i in line.split(' ')])

FILE="./mnist_2_vs_7/mnist_X_test.dat"
with open(FILE) as X_test_file:
	for line in X_test_file:
		X_test.append([int(i) for i in line.split(' ')])

FILE="./mnist_2_vs_7/mnist_y_test.dat"
with open(FILE) as y_test_file:
	for line in y_test_file:
		y_test.append([int(i) for i in line.split(' ')])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)



s_size = 5.2e-05



# The gradient descent algorithm
print("(a)")
print("Step size: " + str(s_size))
GradientDescent(s_size)

# Stochastic Gradient Descent
print("(c)")
print("Step size: " + str(s_size))
Stochastic(s_size)
print("Step size: " + str(s_size*10))
Stochastic(s_size*1.1)
print("Step size: " + str(s_size/10))
Stochastic(s_size/1.1)

# Decay
s_size = 8.2e-05
d_rate = 100
d_value = 0.1
print("(d)")
print("Step size: " + str(s_size) + " Decay rate: " + str(d_rate) + " Decay value: " + str(d_value))
StochasticDecay(s_size, d_rate, d_value)
print("Step size: " + str(s_size) + " Decay rate: " + str(d_rate) + " Decay value: " + str(d_value/10))
StochasticDecay(s_size, d_rate, d_value)
print("Step size: " + str(s_size) + " Decay rate: " + str(d_rate/10) + " Decay value: " + str(d_value))
StochasticDecay(s_size, d_rate, d_value)

# Problem 4
print("Problem 4 (b)")
print("Step size: 6.5e-6")
Problem4(6.5e-6)
