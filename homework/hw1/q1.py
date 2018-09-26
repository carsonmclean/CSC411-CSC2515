import numpy as np

def getZ():
    X = np.random.uniform()
    Y = np.random.uniform()
    Z = (X-Y)**2

    return Z

results = []
iterations = 1000
i = 0
d = 10000
while i < iterations:
    print(iterations - i)
    i += 1
    # if np.random.uniform() > 0.9999:
    #     print("% complete: " + str(i/iterations*100))

    R = 0
    j = 0
    while j < d:
        R += getZ()
        j += 1

    results.append(R)

print("Iterations: " + str(iterations))
print("Expected Value Results:")
print("Paper: " + str(d/6))
print(sum(results)/iterations)

print("\nVariance Results:")
print("Paper: " + str(7/180*d))
print(np.var(results))