import tensorflow as tf

filename = './mlclass-ex1/ex1data1.txt'

def readData(filename):
	reader = tf.TextLineReader()
	_, csv_rows = reader.read(filename)
	#record_defaults = record_default
	population, profit = tf.decode_csv(csv_rows, record_defaults=[[0], [0]])
	#features = tf.pack([population, profit])
	return population, profit


with tf.Session() as sess:
	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
	features = readData(filename_queue)
	print(features)
