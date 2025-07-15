import random
import numpy as np

comp_freq = [random.uniform(0,10) for _ in range(5)]
modes = [[random.uniform(0, 1.0),
          random.uniform(1, 2.0),
          random.uniform(2, 3.0),
          random.uniform(3, 4.0),
          random.uniform(4, 5.0)
          ] for _ in range(5)]

data = {"freq":comp_freq, "modes":modes}
#data = None

if isinstance(data, dict):
    keys = list(data.keys())
    print(keys)
    print(data[keys[0]])
    print(len(data))
else:
    print("bla")

data0 = np.asarray(data[keys[0]])
data1 = np.asarray(data[keys[1]])

print(data0.ndim)
print(data1.ndim)

data1 = np.asarray(data[keys[1]])

print(data1.ndim)
print(len(data1))



