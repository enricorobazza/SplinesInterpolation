import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# Author: https://github.com/melpomene
# Repository: https://gist.github.com/melpomene/2482930
def lagrange(x_nodes, y_nodes):
  def P(x):
    total = 0
    n = len(x_nodes)
    for i in range(n):
      xi = x_nodes[i]
      yi = y_nodes[i]

      def g(i, n):
        tot_mul = 1
        for j in range(n):
          if i == j:
            continue
          xj = x_nodes[j]
          yj = y_nodes[j]
          tot_mul *= (x - xj) / float(xi - xj)
        return tot_mul
  
      total += yi * g(i, n)
    return total
  return P
##########

def ft(t, x):
  soma = 0
  for i in range(len(x)):
    soma += x[i] * t ** i
  return soma

def least_squares(x_nodes, y_nodes, k):
  A = np.empty((len(y_nodes), k))
  for i in range(len(y_nodes)):
    A[i] = [x_nodes[i] ** j for j in range(k)]
  AA = np.dot(A.T, A)
  Ab = np.dot(A.T, y_nodes)
  x = np.linalg.solve(AA, Ab)
  return lambda t: ft(t, x)
  
def get_x_nodes_for_k(k):
  x_nodes = [-1+2*i/k for i in range(k+1)]
  return np.array(x_nodes)

def get_x_nodes_chebyschev(n, a, b):
  x_nodes = []
  for k in range(n):
    x_nodes.append((a+b)/2 - (b-a)/2 * np.cos(k/n*np.pi))
  return np.array(x_nodes)

def get_eks_lagrange(fx):
  eks = []
  ks = []
  for k in range(2, 100):
    x_nodes = get_x_nodes_for_k(k)
    y_nodes = fx(x_nodes)
    pk = lagrange(x_nodes, y_nodes)
    x_plot = np.linspace(-1,1,200)
    ek = max(np.absolute(fx(x_plot)-pk(x_plot)))
    ks.append(k)
    eks.append(ek)
  return (ks, eks)

def get_eks_lagrange_chebyschev(fx):
  eks = []
  ks = []
  for k in range(2, 100):
    x_nodes = get_x_nodes_chebyschev(k, -1, 1)
    y_nodes = fx(x_nodes)
    pk = lagrange(x_nodes, y_nodes)
    x_plot = np.linspace(-1,1,200)
    ek = max(np.absolute(fx(x_plot)-pk(x_plot)))
    ks.append(k)
    eks.append(ek)
  return (ks, eks)

def get_eks_spline(fx):
  eks = []
  ks = []
  for k in range(2, 100):
    x_nodes = get_x_nodes_for_k(k)
    y_nodes = fx(x_nodes)
    sk = CubicSpline(x_nodes, y_nodes, bc_type='natural')
    x_plot = np.linspace(-1,1,200)
    ek = max(np.absolute(fx(x_plot)-sk(x_plot)))
    ks.append(k)
    eks.append(ek)
  return (ks, eks)

def get_eks_spline_known_derivative(fx, fd2x):
  eks = []
  ks = []
  for k in range(2, 100):
    x_nodes = get_x_nodes_for_k(k)
    y_nodes = fx(x_nodes)
    dx1 = fd2x(x_nodes[0])
    dxn = fd2x(x_nodes[-1])
    sk = CubicSpline(x_nodes, y_nodes, bc_type=((2, dx1), (2, dxn)))
    x_plot = np.linspace(-1,1,200)
    ek = max(np.absolute(fx(x_plot)-sk(x_plot)))
    ks.append(k)
    eks.append(ek)
  return (ks, eks)