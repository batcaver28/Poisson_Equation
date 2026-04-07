import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

n = 30
L = 10
h = L / (n - 1)


def crear_matriz(n):
    size = n**3
    # Usamos LIL porque es eficiente para ir rellenando valores uno a uno
    matriz = lil_matrix((size, size)) 
    
    for i in range(size):
        z = i % n
        y = (i // n) % n
        x = (i // n**2) % n
        
        if x == 0 or x == n-1 or y == 0 or y == n-1 or z == 0 or z == n-1:
            matriz[i, i] = 1
        else:
            matriz[i, i] = -6
            # Vecinos
            matriz[i, i + 1] = 1          # z + 1
            matriz[i, i - 1] = 1          # z - 1
            matriz[i, i + n] = 1          # y + 1
            matriz[i, i - n] = 1          # y - 1
            matriz[i, i + n**2] = 1       # x + 1
            matriz[i, i - n**2] = 1       # x - 1
            
    return matriz.tocsr() # Convertimos a CSR para que el solver vuele

def crear_densidad_puntual(n, h):
    rho = np.zeros(n**3)
    p = n // 3
    rho[p * (1 + n + n**2)] = 1 / h**3
    rho[2 * p * (1 + n + n**2)] = 1 / h**3
    return rho


A = crear_matriz(n)
b = -h**2 * crear_densidad_puntual(n, h)

phi = spsolve(A, b)
phi = phi.reshape((n, n, n))




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