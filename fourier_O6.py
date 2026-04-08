import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


n = 5
L = 10
h = L / (n - 1)

def crear_densidad_puntual(n, h):
    rho = np.zeros((n, n, n))
    p = n // 2
    rho[p, p, p] = 1 / h**3
    #rho[2*p, 2*p, 2*p] = 1 / h**3
    return rho

def rho_hat_pqr(p, q, r, rho, n):
    suma = 0
    for i in range(1, n):
        for j in range(1, n):
            for k in range(1, n):
                rho_ijk = rho[i, j, k]
                sin_i = np.sin(np.pi * i * p / n)
                sin_j = np.sin(np.pi * j * q / n)
                sin_k = np.sin(np.pi * k * r / n)
                suma += rho_ijk * sin_i * sin_j * sin_k
    return suma

def phi_hat_pqr(p, q, r, rho_hat, n, h):
    top = -h**2 * rho_hat[p, q, r]
    bottom = 2 * (np.cos(np.pi * p / n) + np.cos(np.pi * q / n) + np.cos(np.pi * r / n) - 3)
    return top / bottom

def phi_ijk(i, j, k, phi_hat, n):
    suma = 0
    for p in range(1, n):
        for q in range(1, n):
            for r in range(1, n):
                phi_hat_pqr = phi_hat[p, q, r]
                sin_i = np.sin(np.pi * i * p / n)
                sin_j = np.sin(np.pi * j * q / n)
                sin_k = np.sin(np.pi * k * r / n)
                suma += (8 / n**3) * (phi_hat_pqr * sin_i * sin_j * sin_k)
    return suma

rho = crear_densidad_puntual(n, h)
rho_hat = np.zeros((n, n, n))
phi_hat = np.zeros((n, n, n))
phi = np.zeros((n, n, n))

for i in range(1, n-1):
    for j in range(1, n-1):
        for k in range(1, n-1):
            rho_hat[i, j, k] = rho_hat_pqr(i, j, k, rho, n)

for i in range(1, n-1):
    for j in range(1, n-1):
        for k in range(1, n-1):
            phi_hat[i, j, k] = phi_hat_pqr(i, j, k, rho_hat, n, h)

for i in range(1, n-1):
    for j in range(1, n-1):
        for k in range(1, n-1):
            phi[i, j, k] = phi_ijk(i, j, k, phi_hat, n)


print(phi)

'''

z = n//2  # corte central

X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

plt.figure()

sc = plt.scatter(
    X.flatten(),
    Y.flatten(),
    c=phi[:, :, z].flatten(),
    cmap='plasma',
    marker='o'
)

plt.colorbar(sc)
plt.title(f'Distribución en corte z = {z}')
plt.xlabel('x')
plt.ylabel('y')

plt.gca().set_aspect('equal')

plt.show()



# -------------------------
# Gráfico de curvas de nivel 3D
# -------------------------
niveles = [0.02, 0.05, 0.1]  # ajusta según tu phi

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for nivel in niveles:
    verts, faces, normals, values = measure.marching_cubes(phi, level=nivel)
    
    # Escalamos vértices a coordenadas físicas
    verts = verts * h
    
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    mesh.set_facecolor(plt.cm.plasma(nivel / max(niveles)))
    ax.add_collection3d(mesh)

# Límites y proporciones reales
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
ax.set_box_aspect([1,1,1])  # fuerza proporción cúbica

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Curvas de nivel 3D del potencial')
plt.show()


'''