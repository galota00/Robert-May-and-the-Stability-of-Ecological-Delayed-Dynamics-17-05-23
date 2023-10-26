import numpy as np
import matplotlib.pyplot as plt
import math

S = 35 # size of system
C = 0.25  # connectance
d = 0  # value of diagonal

sigma = 50  # standard deviation

matnum = 10  # number of random matrices made

IntM = np.zeros((matnum, S)) # generates matrix that will store summed interaction values
WholeArray = [] # Array to store values for histogram
for m in range(0, matnum): # goes matnum number of matrices
    M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
    IntSum = np.zeros(S) # creaters an array that will store summed interaction values
    for i in range(0, S): # goes through each entry in the matrix
        for j in range(0, S):
            if i == j: # if it is on the diagonal
                M[i][j] = d # set it to d
            else:
                if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                    M[i][j] = 0
                if M[i][j] != 0: # Creates adjancency matrix
                    M[i][j] = 1
        IntSum[i] = np.sum(M[i]) # Stores summed values
    IntM[m] = IntSum

for m in range(0, matnum):
    WholeArray = np.concatenate((WholeArray,IntM[m]))

#print(IntM)
#print(WholeArray)

plt.title('S='+str(S)+', σ='+str(sigma)+', C='+str(C))
plt.ylabel('Probability')
plt.xlabel('k-interactions')
plt.hist(x=WholeArray, bins = range(S), density=True ,histtype = 'stepfilled',color = 'coral',edgecolor='black',linewidth=0.5)

plt.show()
