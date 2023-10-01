# Packages used during the lab2
import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Personnals imports
import kernel_function as kf
import sys
import personal_utils as putils
from sklearn.datasets import make_moons, make_circles # for other 2-dim dataset

# Using seed for random generation
numpy.random.seed(100)




# ************************************
# ******** Objective function ********
# ************************************
def objective(alphas):
    """
    Represents the equation (4) of subject (see dual form)

    Args:
        alpha: all alphas tested values 
    
    Returns:
        the alpha(i) values who minimize the dual form
    """
    firstTerm = 0.0
    for i in range(len(alphas)):
        for j in range(len(alphas)):
            firstTerm += alphas[i]*alphas[j]*P[i,j]
    firstTerm = firstTerm/2.0
    secondTerm = numpy.sum(alphas)
    return firstTerm - secondTerm



# ************************************
# ********* Zerofun function *********
# ************************************
def zerofun(vectors):
    """
    Represents the egality constraint equation (10) of subject

    Args:
        vectors of values used for the constraint

    Returns:
        Return the value to constraint
    """
    if(C==None): return numpy.sum(putils.allBetween0andC(vectors)*targets)
    else: return numpy.sum(putils.allBetween0andC(vectors, C)*targets)

# ************************************
# ******** Indicator function ********
# ************************************
def indicator(x,y):
    """
    Uses the non-zero alphas(i) values (with x(i) and t(i))
    to classify new points (if negative it's own to one class, if positive to the other one)
    Specific to 2-dim problems

    Args:
        x: coordinate x of the newpoint
        y: coordinate y of the newpoint

    Returns:
        A scalar value corresponding
    """
    firstPart = 0
    for i in range(N):
        firstPart += alpha[i]*targets[i]*kf.rbf_kernel([x,y],inputs[i])
    return firstPart - bias

# *** Generating basic 2-dim test data ***
# classA = numpy.concatenate(
#     (numpy.random.randn(10,2)*0.2 + [1.5, 0.5],
#      numpy.random.randn(10,2)*0.2 + [-1.5,0.5]))
# classB = numpy.random.randn(20,2)*0.2 + [0.0, -0.5]
# inputs = numpy.concatenate((classA, classB))
# N = inputs.shape[0] # nb of rows
# targets = numpy.concatenate(
#     (numpy.ones(classA.shape[0]),
#      -numpy.ones(classB.shape[0])))

# *** Generating moon 2-dim test data ***
inputs, targets = make_moons(n_samples=40, noise=0.1)
classA = []
classB = []
for i in range(len(targets)):
    targets[i] = (targets[i]-0.5)*2
for i in range(len(inputs)):
    if targets[i]==1.0:
        classA.append(inputs[i])
    elif targets[i]==-1.0:
        classB.append(inputs[i])
N = inputs.shape[0] # nb of rows
permute=list(range(N))
random.shuffle(permute)
inputs=inputs[permute, :]
targets=targets[permute]

# *** Generating circle 2-dim test data ***
# inputs, targets = make_circles(n_samples=40, noise=0.05)
# classA = []
# classB = []
# for i in range(len(targets)):
#     targets[i] = (targets[i]-0.5)*2
# for i in range(len(inputs)):
#     if targets[i]==1.0:
#         classA.append(inputs[i])
#     elif targets[i]==-1.0:
#         classB.append(inputs[i])
# N = inputs.shape[0] # nb of rows
# permute=list(range(N))
# random.shuffle(permute)
# inputs=inputs[permute, :]
# targets=targets[permute]


# Pre-computed matrix P 
P = numpy.ndarray((N,N))
for i in range(N):
    for j in range(N):
        P[i,j]=targets[i]*targets[j]*kf.rbf_kernel(inputs[i],inputs[j])


# initialization variables
start = numpy.zeros(N)
C= 40 # coefficient for slack variables
B=[(0,C) for b in range(N)] # Between 0 and C if constraints
XC= {'type':'eq', 'fun':zerofun}


# *** Heart of program ***
ret = minimize(objective, start,
               bounds=B, constraints=XC)
# ret = minimize(objective, start,
#                bounds=B)
alpha = putils.allNear0(ret['x'])
print(f"Does the model find a solution: ? {'Yes' if ret['success']==True else 'No'} because {ret['message']}")
print(f"alpha:\n{alpha}")

# calcultion of bias
bias = -1.05

# test model
inputs_test, targets_test = make_moons(n_samples=1000, noise=0.1)
for i in range(len(targets_test)):
    targets_test[i] = (targets_test[i]-0.5)*2
# print(inputs_test)
# print(targets_test)
sucess = 0
for i in range(len(inputs_test)):
    classIndicator = 1.0 if indicator(inputs_test[i][0],inputs_test[i][1])>0.0 else -1.0
    if classIndicator==targets_test[i]:sucess+=1
print(f"Error on testing set: {(sucess/len(inputs_test))*100}%")


# print(f"{[indicator(input[0], input[1]) for input in inputs]}") # compute indicator for each input (inferor or egal at -1 or superior or egal at 1))

# *** Plotting ***
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')
plt.axis('equal') # force same scale



# *** Plotting the Decision Boundary ***
xgrid=numpy.linspace(-1.5,1.5)
ygrid=numpy.linspace(-1.5,1.5)
grid=numpy.array([[indicator(x,y)
                    for x in xgrid]
                   for y in ygrid])
plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1,2,1))

plt.title("title") # title of the plot
plt.savefig('symplot.pdf') # save copy in a file
plt.show()