import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_heat_map(x, gamma = 1.0, cmap=plt.cm.Reds):
    if len(x.shape) == 1:
        length = int(np.sqrt(x.shape[0]))
        assert x.shape[0] == length ** 2
        x = x.reshape(length, length)
    x = x ** gamma
    x = x / x.sum()
    x = (x * 255).astype(int)
    plt.imshow(x, cmap=cmap, norm=matplotlib.colors.NoNorm())
    plt.xticks(range(x.shape[0]))
    plt.yticks(range(x.shape[1]))
    plt.imshow(x, cmap=cmap, norm=matplotlib.colors.NoNorm())
    plt.show()

x = np.random.rand(100).reshape(10,10)
plt.imshow(x, cmap=plt.cm.hot)
plt.xticks(range(10))
plt.xticks(range(10))
plt.colorbar()
plt.show()
