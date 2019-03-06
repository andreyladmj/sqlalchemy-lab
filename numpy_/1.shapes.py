import numpy as np

grid = np.arange(2, 30).reshape((2,14))

x = np.array([1,2,4])
x.reshape((1,3))
x[:, np.newaxis]

x = np.arange(5, 10)
y = np.array([2,5,7,9,0,12,123,123,123,123,12,31,23])
np.concatenate((x,y)).reshape((5,2))


x = np.arange(1, 10).reshape((3,3))
y3x3 = np.arange(11, 20).reshape((3,3))
y3x4 = np.arange(11, 23).reshape((3,4))
np.concatenate((x,y3x3))
np.concatenate((x,y3x4), axis=1)

np.vstack((x,y3x4))
np.hstack((x,y3x4))

x = np.arange(6,29)
x1, x2, x3 = np.split(x, [6, 20])

x = np.arange(50).reshape((5,10))
l = np.vsplit(x, [2, 3])
l[0]
l[1]
l[2]


x = np.linspace(0, np.pi, 3)
y = np.arange(0, 9, np.pi)
z = np.zeros((3))
np.multiply(x,y,out=z)

x = np.linspace(0, np.pi, 10)
y = np.zeros((20))
np.power(3,x,out=y[::2])

x = np.arange(0,8)
np.add.reduce(x)
np.add.accumulate(x)

np.set_printoptions(linewidth=500)
x = np.arange(1, 26)
np.power.outer(x,2)
np.multiply.outer(x,x)




L = np.random.random(100)
sum(L)
np.sum(L) # more quikly


M = np.random.random((3, 4))
M.max(axis=1)

heights = np.array([189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193, 178, 173,
                    174, 183, 183, 168, 170, 178, 182, 180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183,
                    177, 185, 188, 188, 182, 185])

print("Mean height:", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height: ", heights.min())
print("Maximum height:", heights.max())
print('25th percentile:', np.percentile(heights, 25))
print('Median:', np.median(heights))
print('75th percentile', np.percentile(heights, 75))

import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.hist(heights)
plt.title('Height Distribution of US Presidents')

plt.xlabel('height (cm)')
plt.ylabel('number')
plt.show()


a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a + b


x = np.random.random((10,3))
Xmean = x.mean(0)
nx = x - Xmean
nx.mean(0)