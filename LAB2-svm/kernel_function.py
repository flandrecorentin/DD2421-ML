import numpy
import math
# Implementations of all kernel functions

def linear_kernel(data1, data2):
    """
    This function correponds to the linear kernel
    
    Args:
        data1: data point used for the measure : array 2*1
        data2: same
    
    Returns:
        scalar value corresponding to the scalar value
    """
    return numpy.dot(data1,data2)


p = 2 # parameter: degree of the polynomials. p=2 -> quadratic shapes
def polynomial_kernel(data1,data2):
    """
    This function correponds to the polynomial kernel
    
    Args:
        data1: data point used for the measure : array 2*1
        data2: same
    
    Returns:
        scalar value corresponding to the polynomial measure
    """
    return (numpy.dot(data1,data2)+1)**p


sigma = 1.0 # parameter: control the smoothness of the boundary
def rbf_kernel(data1, data2):
    """
    This function correponds to the Radial Basis Function (RBF) kernel
    
    Args:
        data1: data point used for the measure : array 2*1
        data2: same
    
    Returns:
        scalar value corresponding to the RBF measure
    """
    l2_norm = numpy.linalg.norm(numpy.subtract(data1, data2))
    return math.exp(-(l2_norm)**2/(2*(sigma**2)))