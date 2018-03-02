from plotData import *

with open("./mlclass-ex1/ex1data1.txt","r") as file:
	data = file.read()

data = data.split('\n')
x = [float(row.split(',')[0]) for row in data]
y = [float(row.split(',')[1]) for row in data]

plotData(x, y,'Population of City in 10,000s', 'Profit in $10,000s', 2, 5)