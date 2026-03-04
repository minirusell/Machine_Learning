import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def calcular_energia(W,y):
    # Energia = 0
    # for i in range(len(y)):
    #     for j in range(len(y)):
    #         if i != j:
    #             Energia = Energia +W[i,j]*y[i]*y[j]
    # Energia = -1/2*Energia
    return -0.5 * y @ W @ y
resultado = 1000
epsilon = 1

uno = cv2.imread('uno.jpg', cv2.IMREAD_GRAYSCALE)
dos = cv2.imread('dos.jpg', cv2.IMREAD_GRAYSCALE)
dos = cv2.resize(dos, (64,64))
uno = cv2.resize(uno, (64,64))
_,uno = cv2.threshold(uno,127,1,cv2.THRESH_BINARY)#pasa de escala de grises a blanco y negro, donde lo que super 127 = 1
_,dos = cv2.threshold(dos,127,1,cv2.THRESH_BINARY)
uno = uno.astype(np.int16)
dos = dos.astype(np.int16)
plt.imshow(dos)
plt.axis('off') # Opcional: oculta los ejes
plt.show()
uno[uno == 0] = -1
dos[dos == 0] = -1
vector_uno = uno.ravel()#pasa de matriz a vector
vector_dos = dos.ravel()
W = 1/2*(vector_uno.reshape(-1,1)*vector_uno+vector_dos.reshape(-1,1)*vector_dos) - np.identity(64*64)

y = vector_dos
index = np.random.choice(len(vector_uno), size = 300, replace=False)
y[index] *= -1
N = 64*64
max_sweeps = 100

energia_old = calcular_energia(W, y)

for sweep in range(max_sweeps):

    # Un sweep = N actualizaciones de neuronas aleatorias
    for _ in range(N):

        neurona_index = random.randrange(N)

        h = W[neurona_index, :] @ y

        if h > 0:
            new_state = 1
        elif h < 0:
            new_state = -1
        else:
            new_state = y[neurona_index]

        # actualizar solo si cambia
        if new_state != y[neurona_index]:
            y[neurona_index] = new_state

    # calcular energía tras el sweep
    energia_new = calcular_energia(W, y)

    print("Sweep:", sweep, "Energia:", energia_new)

    # criterio de parada basado en energía
    if abs(energia_old - energia_new) < 1e-6:
        print("Convergencia alcanzada")
        break

    energia_old = energia_new
    

y[y == -1] = 0
y_matriz = y.reshape(64,64)
plt.imshow(y_matriz)
plt.axis('off') # Opcional: oculta los ejes
plt.show()

