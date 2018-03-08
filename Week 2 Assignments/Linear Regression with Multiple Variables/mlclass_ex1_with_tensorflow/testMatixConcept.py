import tensorflow as tf

with open('testMatrix.txt', 'r') as file:
	data = file.read()

data = data.split('\n')
x = [float(item.split(',')[0]) for item in data if item !='']
y = [float(item.split(',')[1]) for item in data if item !='']

m = len(x)

with tf.Session() as sess:
	tmp = tf.Variable(tf.transpose(tf.reshape(tf.concat([tf.ones_like(x), x], 0), [2, m])), name="tmp", dtype=tf.float32)
	mul = tf.Variable(tf.ones([2, 1]), name="mul", dtype=tf.float32)
	sess.run(tf.global_variables_initializer())
	# print(tmp.eval())
	# print(mul.eval())
	result = sess.run(tf.matmul(tmp, mul))
	print(result)



print('sdfsadfsdfsd')
import matplotlib.pyplot as plt

plt.plot(result, result)
plt.show()

