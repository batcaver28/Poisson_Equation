import numpy as np
from scipy.fftpack import dst, idst
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

n = 301
L = 10
h = L / (n - 1)

def crear_densidad_puntual(n, h):
    rho = np.zeros((n, n, n))
    p = n // 3
    rho[p, p, p] = 1 / h**3
    rho[2*p, 2*p, 2*p] = 1 / h**3
    return rho

def solve_poisson_dirichlet(rho, n, h):
    
    # DST tipo-I en cada dimensión
    rho_hat = dst(dst(dst(rho, type=1, axis=0), type=1, axis=1), type=1, axis=2)
    
    # Precomputar denominadores
    p = np.arange(1, n+1)[:,None,None]
    q = np.arange(1, n+1)[None,:,None]
    r = np.arange(1, n+1)[None,None,:]
    denom = 2*(np.cos(np.pi*p/(n+1)) + np.cos(np.pi*q/(n+1)) + np.cos(np.pi*r/(n+1)) - 3)
    
    phi_hat = -h**2 * rho_hat / denom
    
    # Transformada inversa
    phi = idst(idst(idst(phi_hat, type=1, axis=0), type=1, axis=1), type=1, axis=2)
    
    # Normalización DST tipo-I
    phi /= 8 * (n + 1)**3
    
    return phi

rho = crear_densidad_puntual(n, h)

phi = solve_poisson_dirichlet(rho, n, h)



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


'''
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