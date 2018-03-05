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

'''
Hence we need to find theta values with given input/x values, 
| x0 x1 |       |theta0 |    | y1 |
| x0 x1 |   *   |theta1 | =  | y2 |
| x0 x1 |

we need to convert out x values or input values add extra ones for |x0|
                                                                   |x0|
                                                                   |x0|
To compute matrix multiplicaiton in tensorflow we need to convert one dimensional vectors to mX1 dimensional matrices
'''


theta = tf.zeros([2, 1], dtype=tf.float32) #initializing theta to all zeros. 
#Here if we create with dimention [2] only two it will be vector with length 2, 
#but in order to implement matrix multiplication we converted this vector to 1 dimentional matrix by giving dim [2, 1]

tmpX = tf.transpose(tf.reshape(tf.concat([tf.ones_like(input), input], 0), [2, m])) #adding extra one vector of ones and 
#reshaping it's dimension to [2, m] and getting the transpose of it

INPUT = tf.Variable(tmpX, name="X", dtype=tf.float32) 
y = tf.Variable(output, name="output", dtype=tf.float32)
y = tf.expand_dims(y, 1) # converting vector to maxtix for matrix multiplication

print(computeCost(INPUT, y, theta, m))

alpha = 0.01
iteration = 200
J_history = []
theta = tf.Variable(tf.zeros([2, 1], dtype=tf.float32))
alpha = tf.Variable(alpha)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	h = tf.matmul(INPUT, theta)
	transpose = tf.transpose(INPUT);
	for iter in range(iteration):
		tt = theta.assign_sub((alpha/m) * tf.matmul(transpose,tf.subtract(tf.matmul(INPUT, theta), y)))
		theta_calculated = sess.run(tt)
		cost = computeCost(INPUT, y, theta_calculated, m)
		J_history.append(cost)

hypothesisY = [(theta_calculated[0] + theta_calculated[1]*item) for item in input] #computes predicted hypothesis
plotData(input, output,'Population of City in 10,000s', 'Profit in $10,000s', 2, 5, hypothesisY)


