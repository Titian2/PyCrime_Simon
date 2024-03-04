import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create a plot
plt.plot(x, y)
plt.title('Sin Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()
