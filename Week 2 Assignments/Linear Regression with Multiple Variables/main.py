import tensorflow as tf
from plotData import *

##Reading data from file
with open("./mlclass-ex1/ex1data1.txt","r") as file:
	data = file.read()
data = data.split('\n')
input = [float(row.split(',')[0]) for row in data]
output = [float(row.split(',')[1]) for row in data]
m = len(output)

########################### Part 2: Plotting ##############################
# uncomment this part to see the graph of input datas
# from plotData import *
# plotData(x, y,'Population of City in 10,000s', 'Profit in $10,000s', 2, 5)

##################### Part 3: Gradient descent ############################
def computeCost(X, y, theta, m):
	with tf.Session() as sess:
		cost = tf.matmul(X, theta)
		summation = tf.reduce_sum(tf.squared_difference(cost, y))
		init = tf.global_variables_initializer()
		sess.run(init)
		summation = sess.run(summation)
	return summation/(2*m)

#Computing the Cost of the the linear regression
theta = tf.zeros([2, 1], dtype=tf.float32) #initializing theta to 0
tmpX = tf.reshape(tf.concat([tf.ones_like(input), input], 0), [m,2]) #adding extra one vector of ones and reshaping it's dimension
X = tf.Variable(tmpX, name="X", dtype=tf.float32)
y = tf.expand_dims(tf.Variable(output, name="y", dtype=tf.float32), 1) 

print(computeCost(X, y,theta, m)) #output is ~32.07 for the first iteration with theta values zero

alpha = 0.01
iteration = 100
J_history = []
theta = tf.Variable(tf.zeros([2, 1], dtype=tf.float32))
alpha = tf.Variable(alpha)
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	h = tf.matmul(X, theta)
	transpose = tf.transpose(X);
	for iter in range(iteration):
		tt = theta.assign_sub((alpha/m) * tf.matmul(transpose,tf.subtract(tf.matmul(X, theta), y)))
		theta_calculated = sess.run(tt)
		cost = computeCost(X, y, theta_calculated, m)
		J_history.append(cost)

hypothesisY = [(theta_calculated[0] + theta_calculated[1]*item) for item in input] #computes predicted hypothesis
plotData(input, output,'Population of City in 10,000s', 'Profit in $10,000s', 2, 5, hypothesisY)
