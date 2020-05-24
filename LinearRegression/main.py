from datetime import datetime, time

from LinearRegression.SimpleLinearRegression import *
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# reg1 = SimpleLinerRegression1()
reg1 = SimpleLinerRegression2()
reg1.fit(x, y)

x_predict = 6

print(reg1.predict(np.array([x_predict])))

print(reg1.a_)
print(reg1.b_)

y_hat1 = reg1.predict(x)

plt.scatter(x, y)
plt.plot(x, y_hat1, color='r')
plt.axis([0, 6, 0, 6])
plt.show()

# 性能测试
m = 1000000
big_x = np.random.random(size=m)
big_y = big_x * 2.0 + 3.0 + np.random.normal(size=m)

reg1 = SimpleLinerRegression1()
reg2 = SimpleLinerRegression2()

print(datetime.now())
reg1.fit(big_x, big_y)
print(datetime.now())

print(datetime.now())
reg2.fit(big_x, big_y)
print(datetime.now())