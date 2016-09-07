import numpy as np
import matplotlib.pyplot as plt

#valtest, testval, glsvm
x = [0,0.1,0.2,0.5,1.0];

#valval, testtest, glsvm
y_30=[0.515, 0.620, 0.565, 0.270, 0.155,]
y_40=[0.645, 0.685, 0.635, 0.655,0.660,]
y_50=[0.640, 0.660, 0.640, 0.700, 0.690,]
y_60=[0.595, 0.635, 0.630, 0.670, 0.630,]
y_70=[0.665,0.68,0.68,0.69,0.68]
y_80=[0.690, 0.685, 0.705, 0.705, 0.710,]
y_90=[0.690, 0.695, 0.705, 0.71, 0.65,]
y_100 = [0.68,0.68,0.68,0.68,0.68]

plt.figure(figsize=(8,6))

plt.xlabel("tradeoff")
plt.ylabel("acc")
plt.title("acc, scale=80")
plt.plot(x, y_80)
plt.show()