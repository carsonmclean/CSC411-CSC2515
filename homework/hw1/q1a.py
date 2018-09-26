import numpy as np

results = []
iterations = 100000000
i = 0
while i < iterations:
    i += 1
    # if np.random.uniform() > 0.9999:
    #     print("% complete: " + str(i/iterations*100))
    X = np.random.uniform()
    Y = np.random.uniform()
    Z = (X-Y)**2
    results.append(Z)

print("Iterations: " + str(iterations))
print("Expected Value Results:")
print("Paper: " + str(1/6))
print(sum(results)/iterations)

print("Variance Results:")
print("Paper: " + str(7/180))
print(np.var(results))