import numpy as np
import matplotlib.pyplot as plt
import math

S = 250  # size of system
C = 0.5  # connectance
d = -1  # value of diagonal

sigma = 1  # standard deviation

matnum = 10  # number of random matrices made

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues



for m in range(0, matnum): # goes matnum number of matrices
    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,1)
    #b = np.random.uniform(0,1)
    diag = 0 # value to help compute along diagonal of matrix
    M = np.zeros((S,S)) # matrix full of 0's
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(diag, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d # set it to d
            else:
                if np.random.random() < C: # sets the matrix values to |X| and -|X| with chance C
                    M[i][j] = np.random.normal(0, sigma)
                    M[j][i] = np.random.normal(0, sigma)
        diag = diag + 1 # to compute along diagonal
    w, v = np.linalg.eig(M) # gets the eigenvalues of the matrix
    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
    #print(M)

    plt.scatter(X, Y, color='red', marker=",",s=1,alpha=0.05) # plots the eigenvalues on the plot
X=[]
Y=[]

theta = np.linspace(0, 2 * np.pi, 150) # plots the circle
radius = sigma * math.sqrt(S*C)
a = radius * np.cos(theta) + d
b = radius * np.sin(theta)
plt.plot(a, b, color='black')


plt.xlabel('Real')
plt.ylabel("Imaginary")
plt.xlim([-20, 20])
plt.ylim([-20, 20])
plt.axvline(x=d, linestyle = 'dashed')
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.plot(d, 0, marker="o", markersize=5, color='black')
plt.text(d,-0.5, str(d),horizontalalignment='right',verticalalignment='top',fontsize=12)
plt.title('Random')
plt.show() # shows the plot
