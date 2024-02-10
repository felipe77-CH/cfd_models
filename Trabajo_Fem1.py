from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
#Generacion de malla, en este caso seria de 18x18 divisiones (elementos) en un cuadrado unitario
mesh = UnitSquareMesh(3,3)
#Definir el conjunto del espacion
#V es el espacio de funciones asociado con elementos Galerkin continuos (CG) de grado 1.
# Esto significa que las funciones definidas en este espacio serán lineales por partes sobre cada elemento de la malla.
V = FunctionSpace(mesh, "CG", 1)
#############################################Condiciones de borde######################################################
# Definir la condición de borde en funcion de la geometria 
def boundary_b(x):
    return  x[1] < DOLFIN_EPS

def boundary_u(x):
    return x[1] > 1.0 - DOLFIN_EPS

def boundary_r(x):
    return x[0] > 1.0 - DOLFIN_EPS

def boundary_l(x):
    return x[0] < DOLFIN_EPS
# Expresiones de las condiciones de borde 
u_D_left = Expression("(x[1])", degree=1)
u_D_right = Expression("(x[1]*x[1]+1)", degree=2)
u_D_bottom = Expression("(x[0])", degree=1)
u_D_top = Expression("(x[0]*x[0]+1)", degree=2)
# Definir las condiciones de Dirichlet en los bordes usando las expresiones
bc_left = DirichletBC(V, u_D_left, boundary_l)
bc_right = DirichletBC(V, u_D_right, boundary_r)
bc_bottom = DirichletBC(V, u_D_bottom, boundary_b)
bc_top = DirichletBC(V, u_D_top, boundary_u)
# Aplicar las condiciones de Dirichlet al problema
bcs = [bc_left, bc_right, bc_bottom, bc_top]
#############################################Condiciones de borde######################################################
f = Expression("0", degree=1)
g = Expression("0", degree=1)
#Procesamiento de la solucion-armando la formulacion variacional
u = TrialFunction(V)
w = TestFunction(V)
a = inner(grad(u), grad(w))*dx
L = f*w*dx + g*w*ds
u = Function(V)
#Resolucion del problema
solve(a == L, u, bcs)
#Post-Procesamiento
file = File("Fem_2D_Def_.png.pvd")
file << u;
plot(u)
plt.savefig('Fem_2D_Def_.png')