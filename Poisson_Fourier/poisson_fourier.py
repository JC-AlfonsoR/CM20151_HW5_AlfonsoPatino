'''
Toca optimizar el codigo bastante porque asi como esta demoraria horas encontrando que puntos influyen en cada celda.
Pase parte del codigo a python para probarlo y solo con una una matriz de 100x100x100 se demora muchisimo en recorrer
solo el primer punto en x. Si quiere ejecute el codigo en la consola y vera. Agregue una pequenia optimizacion en la funcion
vecinos pero aun asi no es suficiente.
'''
# python 2.7
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


coordenadas = np.loadtxt('Serena-Venus.txt')
m = 1.0
x_puntos = coordenadas[:,1]
y_puntos = coordenadas[:,2]
z_puntos = coordenadas[:,3]


l_x = max(x_puntos) - min(x_puntos)
l_y = max(y_puntos) - min(y_puntos)
l_z = max(z_puntos) - min(z_puntos)

Puntos = np.zeros((len(x_puntos), 3))

for i in range(len(x_puntos)):
    linea = np.zeros(3)
    linea[0] = x_puntos[i]
    linea[1] = y_puntos[i]
    linea[2] = z_puntos[i]
    Puntos[i] = linea

#Numero de casilla por eje
tamano_grid = 10

#Creacion de grid
dx,dy,dz = l_x/tamano_grid, l_y/tamano_grid, l_z/tamano_grid #NUEVAS DEFINICIONES DE dx, dy y dz y x, y y z
x = np.arange(min(x_puntos),max(x_puntos),dx)
y = np.arange(min(y_puntos),max(y_puntos),dy)
z = np.arange(min(z_puntos),max(z_puntos),dz)
g = np.meshgrid(x,y,z)
X_g = g[0].ravel()
Y_g = g[1].ravel()
Z_g = g[2].ravel()

def superponen(x_g, x_p, dx):
    """Analiza si el dominio x_p +- dx/2 esta dentro del rango x_g + dx . Devuelve True o False
    Se modela que el cuadrado de la grilla esta determinado por el punto en su esquina inferior izquierda, mientras que el
    cuadro de la particula tiene su centro en la posicion del punto de la particula
        Inputs
    x_g       double    Posicion del punto en la grilla. El dominio de la grilla se define como x_g + dx
    x_p       double    Posicion de la particula
    dx        double    Longitud del dominio para la particula y para la grilla
    
        Outputs
    output    boolean
    """
    # Dos casos: 
    # x_p puede ser mayor que x_g
    # x_p puede ser menor que x_g
    
    if x_p == x_g: # Misma posicion
        return True
    
    elif x_p > x_g: # Particula a la derecha del punto en la grilla
        if x_p - x_g <= 3.0/2.0*dx: # La maxima distancia entre xp y xg es 3/2 dx
            return True
            
        else:
            return False
            
    else: # Particula a la izquierda del punto en la grilla
        if x_p + dx/2 -x_g >= 0:
            return True
            
        else:
            return False

def vecinos(x_g, y_g, z_g, dx, dy, dz, Puntos):
    """Encuentra las particulas cuyo dominio definido por xp+-dx/2 yp+-dy/2 esta incluido 
        en el dominio de la malla definido por x_g+dx y_g+dx

        Inputs
    x_g     double      Posicion x del punto en la grilla
    y_g     double      Posicion y del punto en la grilla
    z_g     double      Posicion z del punto en la grilla
    dx,dy,dz   double      Dimensiones del cubo en la grilla
    Puntos array nx3   Posiciones x,y,z de las n particulas muestreadas

        Outputs
    I       list [m x 1]  Lista con los 'm' indices que representan las particulas cuyo dominio 
                            esta contenido en el dominio del cuadro de grilla especificado

        Uso
    I = vecinos(x_g, y_g, dx, dy, Puntos)
    """
    n = len(Puntos)
    I = [] # Indices de las particulas vecinas al punto de la grilla
    
    for i in range(n):
        xp,yp,zp = Puntos[i,:]
        if (superponen(x_g,xp,dx) and superponen(y_g, yp, dy)) and superponen(z_g, zp, dz):
            I.append(i)
            
    return I

def W_peso(x, Dx):
    """ Funcion W(x) definida en el enunciado de la tarea
    """
    if -Dx<x:
        return(1+x/Dx)
    elif x<Dx:
        return(1-x/Dx)
    else:
        return(0)

Vecinos = []
#Econtrar las particulas que aportan masa a cada cuadrado de la grilla
j = 0
for xg,yg,zg in np.vstack(map(np.ravel, g)).T:
    j = j + 1
    print(xg, j)
    Vecinos.append(vecinos(xg, yg, zg, dx, dy, dz, Puntos))
#Vecinos
Rho = np.zeros(np.shape(X_g))
n_g = len(X_g) # Numero de elementos en la malla
N = np.zeros(np.shape(X_g)) 



# Iteracion sobre cada elemento de la malla
for i in range(n_g):
    s = 0;    
    # Iteracion en las particulas que afectan al cuadrado i del grid
    for j in Vecinos[i]:
        #j es la identidad de cada particula
        s = s + (W_peso(X_g[i] - Puntos[j,0],dx)*
                 W_peso(Y_g[i] - Puntos[j,1],dy)*
                 W_peso(Z_g[i] - Puntos[j,2],dz))
        
    Rho[i] = m/(dx*dy*dz)*s
    #N[i] = len(Vecinos[i])
R = np.reshape(Rho, newshape=np.shape(g[0]))

#Se realiza la transformada de Fourier para encontrar la densidad en el espacio de Fourier
rho_fourier = np.fft.fftn(R)


#Se haya  el potencial en el espacio de Fourier
phi_fourier = -rho_fourier

#Se hace la transformada inversa de Fourier para encontrar el potencial real.
phi = np.fft.ifftn(phi_fourier)

#Esta linea devuelve el valor de lal fuerza de gravedad en las componentes fx, fy y fz
fx,fy, fz = np.asarray(np.gradient(phi))

#Como la transformada da como resultado valores complejos, se saca la norma de cada valor en la matriz.
fx = np.absolute(fx)
fy = np.absolute(fy)
fz = np.absolute(fz)

F_x = fx.ravel()
F_y = fy.ravel()
F_z = fz.ravel()

# Creacion del array para los valores de la gravedad
Gravedad = np.zeros(np.shape(X_g))
 
#Se calculan los valores de la fuerza gravitacional sumando las tres componentes que se calcularon previamente usando la funcion gradiente
Gravedad = np.sqrt((F_x**2)+(F_y**2)+(F_z**2))  
 
#Matriz de valores de gravedad
Gravedad_m = np.reshape(Gravedad, newshape=np.shape(g[0]))

'''
Las funciones busqueda_maximos y busqueda_minimos buscan picos y valles en los valores de la fuerza de gravedad.
Para esto se busca un valor que, en el caso de un pico, sea mayor a todoo los valores vecinos o en su "vecindario".
La funcion retorna un arreglo con los indices en los que hay ya sea maximos o minimos.
'''



def busqueda_maximos(array):
    
    #Se define un vecindario que se le exigira a un maximo ser mayor a todos los valores que esten en este vecindario
    vecindario = generate_binary_structure(len(array.shape),2)

    #Para encontrar los maximos locales se usa la funcion maximum_filter del paquete scipy. En local_max, debido a la funcion filtro se incluye tambien un valor de background
    local_max = (maximum_filter(array, footprint = vecindario)==array)

    #El background se establece como el valor minimo que puede tomar la gravedad que en este caso es cero
    background = (array == 0)

    #Se usa la funcion binary_erosion para remover correctamente el background y obtener correctamente los indices de los picos.
    eroded_background = binary_erosion(background, structure = vecindario, border_value = 1)

    #Se resta el background a los valores de local_max para obtener los indices correctos.
    maximos_detectados = local_max - eroded_background

    return np.where(maximos_detectados)

def busqueda_minimos(array):
     
    vecindario = generate_binary_structure(len(array.shape),2)

    local_min = (minimum_filter(array, footprint = vecindario)==array)

    background = (array == 0)

    eroded_background = binary_erosion(background, structure = vecindario, border_value = 1)

    minimos_detectados = local_min - eroded_background

    return np.where(minimos_detectados)


ind_maximos = busqueda_maximos(Gravedad_m)

ind_minimos = busqueda_minimos(Gravedad_m)

print ("Los picos de la fuerza gravitatoria son", Gravedad_m[ind_maximos])
print ("Los valles de la fuerza gravitatoria son", Gravedad_m[ind_minimos])

#Calculo de maximos y minimos globales
maximo = Gravedad_m.max()
minimo = Gravedad_m.min()

print ("El maximo de la fuerza gravitatoria es", maximo)
print ("El minimo de la fuerza gravitatoria es", minimo)

#Creacion de la grafica

#Valores de x y y en el plano

x_plano, y_plano = np.meshgrid(x,y)





#Se escogen el valor de z para el plano para el que se quiere hacer la grafica. Z debe estar en el intervalo de divisiones de la grilla
#Usando este valor se seleccionan los valores de la gravedad en ese plano
z_plano = 0
valores_gravedad = Gravedad_m[:][:][z_plano]

#Guardar datos en csv
x_data = x_plano.ravel()
np.savetxt('x_data.csv',x_data,delimiter=",")
y_data = y_plano.ravel()
np.savetxt('y_data.csv',y_data,delimiter=",")
g_data = valores_gravedad.ravel()
np.savetxt('g_data.csv',g_data,delimiter=",")

#Creacion de grafica
titulo = 'Valores de gravedad para z=', z_plano
plt.pcolor(x_plano, y_plano, valores_gravedad, cmap='RdBu')
plt.title(titulo)
plt.plot(x_puntos,y_puntos, 'bo',alpha = 0.1)
CP1 = plt.contour(x_plano, y_plano, valores_gravedad)
# replace this
plt.clabel(CP1, inline=True, fontsize=10)
plt.colorbar()
plt.show()
plt.savefig('Contornos.png')







 
