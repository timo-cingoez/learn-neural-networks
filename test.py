import math
import numpy as np
import matplotlib.pyplot as pyplot

def f(x):
    return 3*x**2 - 4*x + 5

n = f(3.0)
print(n)

xs = np.arange(-5, 5, 0.25)
print(xs)
ys = f(xs)
print(ys)
pyplot.plot(xs, ys)
pyplot.show()