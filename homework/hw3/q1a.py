import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-7, 7.5, 0.5)

delta = 3.0
huber = np.piecewise(x,
                    [abs(x) <= delta, abs(x) > delta],
                    [lambda x: 1/2*x**2, lambda x: delta*(abs(x) - delta/2)])
plt.plot(x, huber, label="Huber Loss (δ=2)")
plt.plot(x, (1/2)*x**2, label="Squared Error Loss")
plt.xlabel("(y-t)")
plt.ylabel("L(y,t)")
plt.legend()
plt.savefig("q1a_d3.0.png")
plt.show()
