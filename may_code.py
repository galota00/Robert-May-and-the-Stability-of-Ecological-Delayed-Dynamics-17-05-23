import numpy as np
import matplotlib.pyplot as plt
import math

S = 1000  # size of system
C = 0.25  # connectance
d = -10  # value of diagonal

sigma = 1  # standard deviation

matnum = 1  # number of random matrices made

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues

for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    #r = np.random.uniform(0,1) # was trying to randomise the colours but it looked awful
    #g = np.random.uniform(0,0.6)
    #b = np.random.uniform(0,0.1)
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0
    w, v = np.linalg.eig(M) # gets the eigenvalues of the matrix
    X += [x.real for x in w] # adds the real values to the array
    Y += [x.imag for x in w] # adds the imaginary values to the array
plt.scatter(X, Y, color='red', marker=",",s=1, alpha=0.4) # plots the eigenvalues on the plot
X=[]
Y=[]

theta = np.linspace(0, 2 * np.pi, 150) # plots the circle
radius = sigma * math.sqrt(S*C)
a = radius * np.cos(theta) + d
b = radius * np.sin(theta)
plt.plot(a, b,color='orange')


# grpah stuffs
plt.xlabel('Real')
plt.ylabel("Imaginary")
plt.xlim([-30, 10])
plt.ylim([-20, 20])
plt.axvline(x=d, linestyle = 'dashed')
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.plot(d, 0, marker="o", markersize=5, color='black')
plt.text(d,-0.5, str(d),horizontalalignment='right',verticalalignment='top',fontsize=12)
#plt.title('Random')
plt.show() # shows the plot
