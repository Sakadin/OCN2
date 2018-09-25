import Orange
from Orange.data import Table
import numpy as np
data = Table('irisTestValues.tab')
lista = []

text_file = open("irisValues.txt", "r")

values = text_file.read().split(',')

#print(data.__len__())


for i in range(data.__len__()):
    temp = abs(values.index(data[i, 'iris']) - values.index(data[i, 'Laplace']))
    print(temp)
    lista.append(temp)
    #print("value:", values.index(data[i, 'y']), " prediction:", values.index(data[i, 'Laplace']))

print(np.mean(lista))
#print(values.index(data[1, 'Laplace']))

#print(data[1, 'Laplace'])
#print(data[1, 'y'])

