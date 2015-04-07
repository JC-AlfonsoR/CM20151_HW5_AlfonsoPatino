'''
Toca optimizar el codigo bastante porque asi como esta demoraria horas encontrando que puntos influyen en cada celda.
Pase parte del codigo a python para probarlo y solo con una una matriz de 100x100x100 se demora muchisimo en recorrer
solo el primer punto en x. Si quiere ejecute el codigo en la consola y vera. Agregue una pequenia optimizacion en la funcion
vecinos pero aun asi no es suficiente.
'''
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

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

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
        ans = []
        for i, arr in enumerate(arrs):
            slc = [1]*dim
            slc[i] = lens[i]
            arr2 = np.asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
            ans.append(arr2)
        return tuple(ans)

dx,dy,dz = l_x/5, l_y/5, l_z/5 #NUEVAS DEFINICIONES DE dx, dy y dz y x, y y z
x = np.arange(min(x_puntos),max(x_puntos),dx)
y = np.arange(min(y_puntos),max(y_puntos),dy)
z = np.arange(min(z_puntos),max(z_puntos),dz)
g = meshgrid2(x,y,z)
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
    print xg, j
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

#Como la transformada da como resultado valores complejos, se necesita sacar el valor absoluto
fx = np.absolute(fx)
fy = np.absolute(fy)
fz = np.absolute(fz)

F_x = fx.ravel()
F_y = fy.ravel()
F_z = fz.ravel()

# Iteracion sobre cada elemento de la malla
Gravedad = np.zeros(np.shape(X_g))
#for i in range(n_g): 
    #Calculo de gravedad para cada elemento en la malla
#    Gravedad[i] = np.sqrt((F_x[i]**2)+(F_y[i]**2)+(F_z[i]**2))
 
Gravedad = np.sqrt((F_x**2)+(F_y**2)+(F_z**2))  
 
#Matriz de valores de gravedad
Gravedad_m = np.reshape(Gravedad, newshape=np.shape(g[0]))

#print Gravedad
#print Gravedad_m

def busqueda_maximos(array):
    
    vecindario = generate_binary_structure(len(array.shape),2)

    local_max = (maximum_filter(array, footprint = vecindario)==array)

    background = (array == 0)

    eroded_background = binary_erosion(background, structure = vecindario, border_value = 1)

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

print Gravedad_m[ind_maximos]


 
