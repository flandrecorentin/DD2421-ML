# Packages used during the lab2
import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Personnals imports
import kernel_function as kf
import personal_utils as putils

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
        aaaa
        bbbb

    Returns:
        A scalar value
    """
    return numpy.sum(alpha*targets)

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
    bias = 0
    # for i in range(len(alpha)):
    #     if putils.near0(alpha[i])!=0.0:
    #         bias += alpha[i]*targets[i]*kf.linear_kernel([x,y],inputs[i])-targets[i]
    firstPart = 0
    for i in range(N):
        firstPart += putils.near0(alpha[i])*targets[i]*kf.linear_kernel([x,y],inputs[i])
    return firstPart - bias


# *** Generating 2-dim test data ***
classA = numpy.concatenate(
    (numpy.random.randn(10,2)*0.2 + [1.5, 0.5],
     numpy.random.randn(10,2)*0.2 + [-1.5,0.5]))
classB = numpy.random.randn(20,2)*0.2 + [0.0, -0.5]
inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
     -numpy.ones(classB.shape[0])))
N = inputs.shape[0] # nb of rows
permute=list(range(N))
random.shuffle(permute)
inputs=inputs[permute, :]
targets=targets[permute]


# Pre-computed matrix P 
P = numpy.ndarray((N,N))
for i in range(N):
    for j in range(N):
        P[i,j]=targets[i]*targets[j]*kf.linear_kernel(inputs[i],inputs[j])


# initialization variables
start = numpy.zeros(N)
C= 0.5 # coefficient for slack variables
B=[(0,C) for b in range(N)] # Between 0 and C if constraints
XC=constraint={'type':'eq', 'fun':zerofun}


# *** Heart of program ***
# ret = minimize(objective, start,
#                bounds=B, constraints=XC)
ret = minimize(objective, start,
               bounds=B)
alpha = ret['x']
# print(f"alpha:\n{alpha}") # print alpha values (majority of 0)
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
xgrid=numpy.linspace(-3,3)
ygrid=numpy.linspace(-2,2)
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