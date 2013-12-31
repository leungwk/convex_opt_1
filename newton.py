import numpy as np
def hessian(A,x):
    """
\grad^2 f(x)
= \grad \grad f(x)
= A^T \grad (1 -Ax)^{-1} +\grad (1 -x)^{-1} -\grad (1 +x)^{-1}
= A^T (-1) diag((1 -Ax)^{-2})(-A) +diag((-1)(1 -x)^{-2}(-1) -(-1)(1 +x)^{-2}(+1))
= A^T diag((1 -Ax)^{-2}) A +(1 -x)^{-2} +(1 +x)^{-2}
"""
    d = 1./(1 -np.dot(A,x))
    return np.dot(np.dot(A.T,np.diag(d)),A) +np.diag()
