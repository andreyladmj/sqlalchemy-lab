import numpy as np

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', '<f8')})

np.dtype({'names':('name', 'age', 'weight'), 'formats':((np.str_, 10), int, np.float32)})
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])

print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
print(data['name'])

print(data[data['age'] < 30]['name'])
