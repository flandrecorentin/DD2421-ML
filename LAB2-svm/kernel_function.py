import numpy
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

def polynomial_kernel(data1,data2):
    """
    This function correponds to the polynomial kernel
    
    Args:
        data1: data point used for the measure : array 2*1
        data2: same
    
    Returns:
        scalar value corresponding to the measure
    """
    return 0