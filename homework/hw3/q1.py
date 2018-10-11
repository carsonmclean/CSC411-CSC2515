from sklearn.datasets import load_boston
import numpy as np

def huber_loss(x, delta):
    return np.piecewise(x,
                        [abs(x) <= delta, abs(x) > delta],
                        [lambda x: 1/2*x**2, lambda x: delta*(abs(x) - delta/2)])

def thing(X, y):
    w = np.zeros(X.shape[1])
    b = 0.0 # scalar
    delta=1

    for i in range(100000):
        pred = w.dot(X.T) + b
        print(np.sum(huber_loss(pred-y, delta)))
        # using derivates found in 1b for Huber Loss
        a = pred-y
        dL_dy= np.piecewise(a,
                             [abs(a) <= delta, a > delta, a < -delta],
                             [lambda a: a, delta, -delta])
        # dL_dw = dL_dy * dy_dw
        dL_dw = dL_dy.dot(X) # why not X.T?

        w -= (dL_dw*0.00000001)

        # bias?



def main():
    boston = load_boston()
    x = boston['data']
    N = x.shape[0]
    d = x.shape[1]
    y = boston['target']
    thing(x, y)

main()