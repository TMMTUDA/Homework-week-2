import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

# Define the function F(x, y) and its complex conjugate F*(x, y)
def F(x, y):
    i = complex(0, 1)
    return np.exp(i*y) + 2*np.exp(-i*y/2)*np.cos((np.sqrt(3)/2)*x)

def F_star(x, y):
    return np.conj(F(x, y))

# Define the matrix M
def create_M(x, y):
    return np.array([[0, F(x, y)], [F_star(x, y), 0]])

# Define range for x and y
x_range = np.linspace(-np.pi, np.pi, 100)
y_range = np.linspace(-np.pi, np.pi, 100)

# Create grid of (x, y) values
X, Y = np.meshgrid(x_range, y_range)

# Initialize arrays to store eigenvalues
eigenvalues1 = np.zeros_like(X, dtype=np.complex128)
eigenvalues2 = np.zeros_like(X, dtype=np.complex128)

# Calculate eigenvalues for each (x, y) point
for i in range(len(x_range)):
    for j in range(len(y_range)):
        M = create_M(x_range[i], y_range[j])
        eigenvalues, _ = np.linalg.eig(M)
        eigenvalues1[i, j] = eigenvalues[0]
        eigenvalues2[i, j] = eigenvalues[1]

# Plot eigenvalues as surface plots
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, eigenvalues1.real, cmap='viridis')
ax1.set_title('Eigenvalue 1 (Real)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Eigenvalue 1 (Real)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, eigenvalues2.real, cmap='viridis')
ax2.set_title('Eigenvalue 2 (Real)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Eigenvalue 2 (Real)')

plt.tight_layout()
plt.show()