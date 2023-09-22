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
    threshold = 0.0001
    if(value>threshold): return value
    else: return 0.0