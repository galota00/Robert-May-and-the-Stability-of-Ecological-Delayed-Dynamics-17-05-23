import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp

S = 1000  # size of system
C = 0.1  # connectance
d = -1.1  # value of diagonal for A
d2 = -0.9 # value of diagonal for B

sigma = 0.1  # standard deviation

tao = 1

matnum = 1  # number of random matrices made

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues
tao = 0

for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values

    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d+d2 # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0

    w, v = np.linalg.eig(M) # gets the eigenvalues of the matrix
    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0.8,0.8), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues

tao=1
for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)

    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0
    w1, v1 = np.linalg.eig(M) # gets the eigenvalues of the matrix B
    w2, v2 = np.linalg.eig(M2) # gets eigenvalues of matrix A

    w=[]
    for i in range(0,S):
        lam = w2[i] + sp.special.lambertw(w1[i]*tao*math.exp(-w2[i]*tao))/tao
        w.append(lam)

    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0.6,0.6), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues
tao = 1.3

for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)

    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0
    w1, v1 = np.linalg.eig(M) # gets the eigenvalues of the matrix B
    w2, v2 = np.linalg.eig(M2) # gets eigenvalues of matrix A

    w=[]
    for i in range(0,S):
        lam = w2[i] + sp.special.lambertw(w1[i]*tao*math.exp(-w2[i]*tao))/tao
        w.append(lam)

    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0.4,0.4), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot
# grpah stuffs

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues
tao = 2

for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)

    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0
    w1, v1 = np.linalg.eig(M) # gets the eigenvalues of the matrix B
    w2, v2 = np.linalg.eig(M2) # gets eigenvalues of matrix A

    w=[]
    for i in range(0,S):
        lam = w2[i] + sp.special.lambertw(w1[i]*tao*math.exp(-w2[i]*tao))/tao
        w.append(lam)

    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0.2,0.2), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues
tao = 3

for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)

    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0
    w1, v1 = np.linalg.eig(M) # gets the eigenvalues of the matrix B
    w2, v2 = np.linalg.eig(M2) # gets eigenvalues of matrix A

    w=[]
    for i in range(0,S):
        lam = w2[i] + sp.special.lambertw(w1[i]*tao*math.exp(-w2[i]*tao))/tao
        w.append(lam)

    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0,0), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot

# grpah stuffs
theta = np.linspace(0, 2 * np.pi, 150) # plots the circle
radius = sigma * math.sqrt(S*C)
a = radius * np.cos(theta) + d + d2
b = radius * np.sin(theta)
plt.plot(a, b,color=(1,0.5,0.5))

delta = 0.025
d =-d #a
d2 = -d2 #b

tao=3
x = np.arange(-1,0.5, delta)
y = np.arange(-0.8,0.8, delta)
p, q = np.meshgrid(x, y)
# define some function f(n,x,y)
f = lambda n, x, y: (d2**2)+2*d2*np.exp(tao*x)*((d+x)*np.cos(tao*y)-y*np.sin(tao*y))+np.exp(2*tao*x)*(((d+x)**2)+y**2)
z=f(1, p,q)
# plot contour line of f(1,x,y)==0
plt.contour(p, q, z , [1], colors=['blue'])

plt.xlabel('Λ')
plt.ylabel("ω")
plt.xlim([-3.5, 0.5])
plt.ylim([-2.25, 2.25])
plt.axvline(x=0, color='black',zorder=0)
plt.show() # shows the plot
