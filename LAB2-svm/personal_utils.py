import sys
# ************************************
# ********* Near 0 function **********
# ************************************
def near0(value):
    """
    Return 0 if approximately 0 (means lower than the threshold)

    Args:
        value: value to estimate if near 0

    Returns:
        0 or value depending of the approximation and the threshold
    """
    threshold = 0.00001
    if(value>threshold): return value
    else: return 0.0

def allNear0(value):
    """
    Return the table of each near0 value

    Args:
        value: table to estimate if near 0

    Returns:
       table contening 0 or value depending of the approximation and the threshold
    """
    for i in range(len(value)):
        value[i] = near0(value[i])
    return value

def nbNon0(value):
    """
    Compute the number of near0 values non nul

    Args:
        value: table to estimate if near 0

    Returns:
       scalar value of the number of 
    """
    nb =0
    for val in value:
        if near0(val)!=0.0:
            nb += 1
    return nb

def allBetween0andC(value, C=sys.maxsize):
    """
    Compute all the value between 0 and C

    Args:
        value: table to estimate if near 0

    Returns:
       scalar value of the number of 
    """
    for i in range(len(value)):
        if value[i]>C:
            value[i] = C
    return value