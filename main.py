import numpy as np
import matplotlib.pyplot as plt

import method

# Função a ser interpolada
def fx(x):
  return 1 / (2+25*(x**2))

## Segunda derivada da função
def fd2x(x):
  return - (50*(1-75*(x**2))) / ((1 + 25*(x**2))**3)


## Cálculo de ek para Lagrange
(ks, eks) = method.get_eks_lagrange(fx) 
## Cálculo de ek para Langrange com Nós de Chebyshev
(ks_1, eks_1) = method.get_eks_lagrange_chebyschev(fx) 
## Cálculo de ek para Splines Cúbicas Naturais
(ks2, eks2) = method.get_eks_spline(fx) 
## Cálculo de ek para Splines Cúbicas Derivadas Conhecidas
(ks3, eks3) = method.get_eks_spline_known_derivative(fx, fd2x) 


plt.figure(figsize=(12,8))

## Gráfico log(ek) por k para todos os métodos #########################
ax1 = plt.subplot(2, 2, 1)
ax1.set_ylabel('log(ek)')
ax1.set_xlabel('k')
ax1.semilogy(ks, eks, label="Lagrange")
ax1.semilogy(ks_1, eks_1, label="Lagrange Chebyschev")
ax1.semilogy(ks2, eks2, label="Splines cúbicas naturais")
ax1.semilogy(ks2, eks3, label="Splines cúbicas derivadas conhecidas")
ax1.legend(loc="best")
ax1.grid()
#######################################################################

## Gráfico log(ek) por log(k) para os dois métodos de Splines Cúbicas
ax2 = plt.subplot(2, 2, 2)
ax2.set_ylabel('log(ek)')
ax2.set_xlabel('log(k)')
ax2.loglog(ks2, eks2, label="Splines cúbicas naturais")
ax2.loglog(ks2, eks3, label="Splines cúbicas derivadas conhecidas")
ax2.legend(loc='best')
ax2.grid()
####################################################################

## Interpolação por mínimos quadrados da primeira Splines ##########
k=2
func = method.least_squares(np.log(ks2), np.log(eks2), k)
x_plot = np.array(np.log(ks2))
b = func(0) ## calculo do coeficiente linear
a = (func(x_plot[1])-b)/x_plot[1] ## calculo do coeficiente angular
print("Splines cúbicas naturais -> a: %f, b: %f"%(a, b))
####################################################################

## Gráfico log(ek) por log(ek) da primeira Splines com a Interpolação por Mínimos Quadrados
ax3 = plt.subplot(2, 2, 3)
ax3.set_ylabel('log(ek)')
ax3.set_xlabel('log(k)')
ax3.plot(np.log(ks2), np.log(eks2), label="Splines Cúbicas Naturais")
ax3.plot(x_plot, func(x_plot), label="Reta interpolada por Quadrados Mínimos")
ax3.legend(loc='best')
ax3.grid()
##########################################################################################

## Interpolação por mínimos quadrados da segunda Splines ###########
func = method.least_squares(np.log(ks3), np.log(eks3), k)
x_plot = np.array(np.log(ks3))
b = func(0) ## calculo do coeficiente linear
a = (func(x_plot[1])-b)/x_plot[1] ## calculo do coeficiente angular
print("Splines cúbicas derivadas conhecidas -> a: %f, b: %f"%(a, b))
####################################################################

## Gráfico log(ek) por log(ek) da segunda Splines com a Interpolação por Mínimos Quadrados
ax4 = plt.subplot(2, 2, 4)
ax4.set_ylabel('log(ek)')
ax4.set_xlabel('log(k)')
ax4.plot(np.log(ks3), np.log(eks3), label="Splines Cúbicas Derivadas Conhecidas")
ax4.plot(x_plot, func(x_plot), label="Reta interpolada por Quadrados Mínimos")
ax4.legend(loc='best')
ax4.grid()
##########################################################################################


plt.show()
