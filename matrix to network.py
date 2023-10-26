# Create random graph
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

S = 35  # size of system
C = 0.2  # connectance
d = 0  # value of diagonal

sigma = 0.1  # standard deviation

X = [] # real parts of the eigenvalues
Y = [] # imaginary parts of the eigenvalues

M = np.random.normal(0, sigma, size=(S, S)) # generates the matrix using normally distributed values
for i in range(0, S): # goes through each entry in the matrix
    for j in range(0, S):
        if i == j: # if it is on the diagonal
            M[i][j] = d # set it to d
        else:
            if np.random.random() > C: # sets the matrix values to 0 with chance (1-C)
                M[i][j] = 0
#G = nx.random_geometric_graph(500, 0.125)
A = M
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
G.edges(data=True)

nx.draw(G,arrowsize=15)  # networkx draw()
plt.draw()  # pyplot draw()
plt.show()
