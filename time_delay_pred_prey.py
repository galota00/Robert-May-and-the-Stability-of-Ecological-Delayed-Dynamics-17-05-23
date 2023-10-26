import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp

S = 1000  # size of system
C = 0.1  # connectance
d = -4.5  # value of diagonal for A
d2 = -0.9 # value of diagonal for B

sigma = 0.1  # standard deviation

tao = 1

matnum = 1  # number of random matrices made

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues
tao = 0

for m in range(0, matnum): # goes matnum number of matrices
    for m in range(0, matnum): # goes matnum number of matrices
        #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
        #g = np.random.uniform(0,0.6)
        #b = np.random.uniform(0,0.1)
        diag = 0 # value to help compute along diagonal of matrix
        M = np.zeros((S,S)) # matrix full of 0's
        for i in range(0, S): # goes through each entry in the matrix
            for j in range(diag, S):
                if i == j: # if it is on the diagonal
                    M[i][j] = d+d2 # set it to d
                else:
                    if np.random.random() < C: # sets the matrix values to |X| and -|X| with chance C
                        M[i][j] = np.random.normal(0, sigma)
                        if M[i][j] > 0:
                            M[j][i] = -np.absolute(np.random.normal(0, sigma))
                        if M[i][j] < 0:
                            M[j][i] = np.absolute(np.random.normal(0, sigma))
            diag = diag + 1 # to compute along diagonal
        w, v = np.linalg.eig(M) # gets the eigenvalues of the matrix
        X += [x.real for x in w] # adds the real values to the array
        Y += [x.imag for x in w] # adds the imaginary values to the array
        #print(M)
    plt.scatter(X, Y, color=(1,0.8,0.8), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot


X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues

tao=1
for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)
    diag = 0
    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() < C: # sets the matrix values to |X| and -|X| with chance C
                    M[i][j] = np.random.normal(0, sigma)
                    if M[i][j] > 0:
                        M[j][i] = -np.absolute(np.random.normal(0, sigma))
                    if M[i][j] < 0:
                        M[j][i] = np.absolute(np.random.normal(0, sigma))
        diag = diag + 1 # to compute along diagonal
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

tao=1.3
for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)
    diag = 0
    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() < C: # sets the matrix values to |X| and -|X| with chance C
                    M[i][j] = np.random.normal(0, sigma)
                    if M[i][j] > 0:
                        M[j][i] = -np.absolute(np.random.normal(0, sigma))
                    if M[i][j] < 0:
                        M[j][i] = np.absolute(np.random.normal(0, sigma))
        diag = diag + 1 # to compute along diagonal
    w1, v1 = np.linalg.eig(M) # gets the eigenvalues of the matrix B
    w2, v2 = np.linalg.eig(M2) # gets eigenvalues of matrix A

    w=[]
    for i in range(0,S):
        lam = w2[i] + sp.special.lambertw(w1[i]*tao*math.exp(-w2[i]*tao))/tao
        w.append(lam)

    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0.4,0.4), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues

tao=2
for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)
    diag = 0
    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() < C: # sets the matrix values to |X| and -|X| with chance C
                    M[i][j] = np.random.normal(0, sigma)
                    if M[i][j] > 0:
                        M[j][i] = -np.absolute(np.random.normal(0, sigma))
                    if M[i][j] < 0:
                        M[j][i] = np.absolute(np.random.normal(0, sigma))
        diag = diag + 1 # to compute along diagonal
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

tao=3
for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    M2 = d*np.identity(S)
    diag = 0
    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d2 # set it to d
            else:
                if np.random.random() < C: # sets the matrix values to |X| and -|X| with chance C
                    M[i][j] = np.random.normal(0, sigma)
                    if M[i][j] > 0:
                        M[j][i] = -np.absolute(np.random.normal(0, sigma))
                    if M[i][j] < 0:
                        M[j][i] = np.absolute(np.random.normal(0, sigma))
        diag = diag + 1 # to compute along diagonal
    w1, v1 = np.linalg.eig(M) # gets the eigenvalues of the matrix B
    w2, v2 = np.linalg.eig(M2) # gets eigenvalues of matrix A

    w=[]
    for i in range(0,S):
        lam = w2[i] + sp.special.lambertw(w1[i]*tao*math.exp(-w2[i]*tao))/tao
        w.append(lam)

    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color=(1,0,0), marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot


t = np.linspace(0, 2*np.pi, 100) # plots the elipse
radius = sigma * math.sqrt(S*C)
plt.plot(d+d2+(0.3634*radius*np.cos(t)) , (1.64*radius*np.sin(t)),color=(1,0.5,0.5) )

plt.title('Predator-prey')
plt.xlabel('Λ')
plt.ylabel('ω')
plt.xlim([-2, 1])
plt.ylim([-2.25, 2.25])
plt.axvline(x=0, color='black',zorder=0)
plt.show() # shows the plot
