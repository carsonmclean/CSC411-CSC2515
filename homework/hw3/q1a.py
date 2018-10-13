import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-7, 7.5, 0.5)

deltas = [0.5, 1.0, 3.0]
for delta in deltas:
    huber = np.piecewise(x,
                        [abs(x) <= delta, abs(x) > delta],
                        [lambda x: 1/2*x**2, lambda x: delta*(abs(x) - delta/2)])
    label = "Huber Loss (δ=" + str(delta) + ")"
    plt.plot(x, huber, label=label)
    plt.plot(x, (1/2)*x**2, label="Squared Error Loss")
    plt.xlabel("(y-t)")
    plt.ylabel("L(y,t)")
    plt.legend()
    filename = "q1a_d" + str(delta) + ".png"
    plt.savefig(filename)
    # plt.show()
    plt.clf()
