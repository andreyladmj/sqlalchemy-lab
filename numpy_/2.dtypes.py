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


coords = [
    (15, 15),
    (20, 0),
    (0, 18),
    (20, 30),
    (30, 20),
]

import matplotlib.pyplot as plt

index = 3
current_point = (20, 25)

for p in coords:
    if p != current_point:
        plt.plot([current_point[0], p[0]], [current_point[1], p[1]])

plt.title('Index: {}, Point: {}'.format(index, current_point))
plt.show()

# delta_x = touch_x - center_x
# delta_y = touch_y - center_y
# theta_radians = atan2(delta_y, delta_x)

current_ponit = np.array(current_point)
other_points = np.array(coords)

diff = current_ponit - other_points
np.arctan2(diff[:, 0], diff[:, 1]) * 180 / np.pi

np.int
