import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(data):
    return (data ** 5) + (0.3 * np.sin(data * 6 * np.pi)) + (0.3 * np.cos(data * 6 * np.pi))

# Generate data points
data = np.linspace(-1, 1, 400)
y = f(data)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(data, y, label=r'$(x^5 + 0.3 \cdot \sin(6\pi x) + 0.3 \cdot \cos(6\pi x))$', color='b')
plt.title("Plot of the function $f(x) = x^5 + 0.3 \sin(6\pi x) + 0.3 \cos(6\pi x)$")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

