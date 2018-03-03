import tensorflow as tf

with open("./mlclass-ex1/ex1data1.txt","r") as file:
	data = file.read()
data = data.split('\n')
x = [float(row.split(',')[0]) for row in data]
y = [float(row.split(',')[1]) for row in data]
m = len(y)

########################### Part 2: Plotting ##############################
# uncomment this part to see the graph of input data
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
tmpX = tf.reshape(tf.concat([tf.ones_like(x), x], 0), [m,2]) #adding extra one vector of ones and reshaping it's dimension
X = tf.Variable(tmpX, name="X", dtype=tf.float32)
y = tf.expand_dims(tf.Variable(y, name="y", dtype=tf.float32), 1) 
alpha = 0.01
print(computeCost(X, y,theta, m)) #output is ~32.07 for the first iteration with theta values zero
