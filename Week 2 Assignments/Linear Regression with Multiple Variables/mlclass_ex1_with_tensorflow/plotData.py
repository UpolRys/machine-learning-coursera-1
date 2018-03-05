import matplotlib.pyplot as plt
import numpy as np

def plotData(x, y, xLabel, yLabel, xInterval, yInterval,J_history):
	def findRoundedBounds(arr, interval):
		minimum = min(arr)
		maximum = max(arr)
		minimum = minimum - (minimum % interval)
		maximum = maximum + (maximum % interval)
		return minimum, maximum

	_xRanges = findRoundedBounds(x, xInterval)
	_yRanges = findRoundedBounds(y, yInterval)
	plt.xticks(np.arange(_xRanges[0], _xRanges[1], xInterval))
	plt.yticks(np.arange(_yRanges[0], _yRanges[1], yInterval))

	plt.ylabel(yLabel, fontsize=8)
	plt.xlabel(xLabel, fontsize=8)
	plt.scatter(x, y, c='red', marker='x', alpha=0.5)
	plt.plot(x,J_history, c='blue')
	plt.show()