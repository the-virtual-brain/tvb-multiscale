import sys
import numpy as np
import os
import glob

# Building the Weight Matrix Ground Truth
Weight_Matrix=np.loadtxt("iSTDP.dat")
SimResults1 = np.loadtxt("iSTDP1.csv")
SimResults4 = np.loadtxt("iSTDP4.csv")

print( ["Length Ground Truth: " +  str(len(Weight_Matrix)) + " Length Simulation1 Results: " + str(len(SimResults1)) + " Length Simulation4 Results: " + str(len(SimResults4)) ])

SimResults1=np.matrix(SimResults1)
SimResults4=np.matrix(SimResults4)
Weight_Matrix=np.matrix(Weight_Matrix)

Weight_Matrix.sort(axis=0)
SimResults1.sort(axis=0)
SimResults4.sort(axis=0)

Difference1 = SimResults1-Weight_Matrix
# Get rid of approximation errors
Error1 = np.sum(Difference1)

Difference4 = SimResults4-Weight_Matrix
# Get rid of approximation errors
Error4 = np.sum(Difference4)

Error = Error1 + Error4

if Error < 3.0:
    sys.exit(0)
else:
   sys.exit(-1)
