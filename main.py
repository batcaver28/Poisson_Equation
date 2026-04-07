import numpy as np
import matplotlib.pyplot as plt

n = 19
L = 10
h = L / (n - 1)

def crear_matriz(n):
    matriz = np.zeros((n**3, n**3))
    for i in range(n**3):
        z = i % n
        y = (i // n) % n
        x = (i // n**2) % n
        if x == 0 or x == n-1 or y == 0 or y == n-1 or z == 0 or z == n-1:
            matriz[i, i] = 1
        else:
            matriz[i, i] = -6
            if z + 1 < n:
                matriz[i, (z + 1) + n * y + n**2 * x] = 1
            if 0 <= z - 1:
                matriz[i, (z - 1) + n * y + n**2 * x] = 1
            if y + 1 < n:
                matriz[i, z + n * (y + 1) + n**2 * x] = 1
            if 0 <= y - 1:
                matriz[i, z + n * (y - 1) + n**2 * x] = 1
            if x + 1 < n:
                matriz[i, z + n * y + n**2 * (x + 1)] = 1
            if 0 <= x - 1:
                matriz[i, z + n * y + n**2 * (x - 1)] = 1
    return matriz

def crear_densidad_puntual(n, h):
    rho = np.zeros(n**3)
    p = n // 2
    rho[p + p * n + p * n**2] = 1
    return rho


A = crear_matriz(n)
b = -h**2 * crear_densidad_puntual(n, h)

phi = np.linalg.solve(A, b)
phi = phi.reshape((n, n, n))






z = 4  # corte central

X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

plt.figure()

sc = plt.scatter(
    X.flatten(),
    Y.flatten(),
    c=phi[:, :, z].flatten(),
    cmap='viridis'
)

plt.colorbar(sc)
plt.title(f'Distribución en corte z = {z}')
plt.xlabel('x')
plt.ylabel('y')

plt.gca().set_aspect('equal')

plt.show()