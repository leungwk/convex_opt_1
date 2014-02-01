import numpy as np
from numpy.random import RandomState
from numpy.linalg import solve

"""
Efficient numerical method for a regularized least-squares problem: A numerical instance

hw7 extra2, convex opt 1 (SEE)

(comparing a generic versus a specialized method)

\[
    min. \sum_{i=1}^k (a_i^Tx -b_i)^2 +\delta \sum_{i=1}^{n-1} (x_i -x_{i+1})^2 +\eta\sum_{i=1}^n x_i^2
\]
has the normal equation:
\[
    (A^TA +\delta\Delta +\eta I)x^* = A^Tb
\]
where 
\[
\Delta =
\begin{bmatrix}
 1 & -1 &    &        &    &    &   \\
-1 &  2 & -1 &        &    &    &   \\
   & -1 &  2 &        &    &    &   \\
   &    &    & \ddots &    &    &   \\
   &    &    &        &  2 & -1 &   \\
   &    &    &        & -1 &  2 & -1\\
   &    &    &        &    & -1 &  1
\end{bmatrix}
\]


$\Delta$ derived from
\begin{align*}
\sum_{i=1}^{n-1} (x_i -x_{i+1})^2
&= \sum_{i=1}^{n-1} x_i^2 -2x_ix_{i+1} +x_{i+1}^2\\
&= x_1^2 -2x_1x_2 +x_2^2\\
 + x_2^2 -2x_2x_3 +x_3^2\\
 + \vdots\\
 + x_{n-1}^2 -2x_{n-1}x_n +x_n^2\\
&= x_1^2 -x_1x_2\quad\text{want to group by per dimension}\\
 + 2x_2^2 -x_1x_2\\
 \quad    -x_1x_2\\
 + 2x_{n-1}^2 -x_{n-2}x_{n-1}\\
 \quad    -x_{n-1}x_n\\
 + x_n^2  -x_{n-1}x_n\\
&= x_1(x_1 -x_2)\quad\text{inside brackets all linear terms}\\
 + x_2(-x_1 +2x_2 -x_3)\\
 + \dots\\
 + x_{n-1}(-x_{n-2} +2x_{n-1} -x_n)\\
 + x_n(x_n -x_{n-1})
&= [x_1-x_2, -x_1+2x_2-x_3, -x_2+2x_3-x_4, \dots, -x_{n-2}+2x_{n-1}-x_n, x_n -x_{n-1}]x\\
&= x^T\Delta x
\end{align*}
"""

precision = np.longfloat # 128-bits here

## q9.31a
n,k = (2000,100)
delta = eta = 1
prng = RandomState(1)
A = np.array(prng.randn(k,n),dtype=precision) # column major
b = np.array(prng.randn(k,1),dtype=precision)

# form D
lb = np.array([-1]*(n-1),dtype=precision) # lower band
db = np.array([2]*n,dtype=precision)
db[0] = db[-1] = 1
ub = lb.copy()




g = np.dot(A.T,b)


## generic method
### form \Delta
Delta = np.zeros((n,n))
Delta += 2*np.eye(n)
Delta += -1*np.diag(np.ones(n-1),-1) # lb
Delta += -1*np.diag(np.ones(n-1),1) # ub
Delta[0,0] = Delta[-1,-1] = 1

F = np.dot(A.T,A) +delta*Delta +eta*np.eye(n)
x_star_generic = solve(np.array(F,dtype=np.float64),np.array(g,dtype=np.float64)) # longfloat (float128 here) unsupported in linalg




## src: http://en.wikipedia.org/w/index.php?title=Tridiagonal_matrix_algorithm&oldid=589021090 Thomas algorithm
## might be numerically unstable
def TDMASolve(a, b, c, d):
    """a is lb, b is db, c is ub, d is RHS"""
    if len(b) != len(d):
        raise ValueError("Warning: diagonal and RHS lengths do not match")
    n = len(d) # n is the numbers of rows, a and c has length n-1
    for i in xrange(n-1):
        d[i+1] -= float(d[i] * a[i]) / b[i]
        b[i+1] -= float(c[i] * a[i]) / b[i]
    for i in reversed(xrange(n-1)):
        d[i] -= float(d[i+1] * c[i]) / b[i+1]
    return [float(d[i]) / b[i] for i in xrange(n)] # return the solution



## structure-aware method
D = {}
D['lb'] = lb*delta
D['db'] = db*delta +eta
D['ub'] = ub*delta

z1 = np.array([TDMASolve(D['lb'].copy(), D['db'].copy(), D['ub'].copy(), g.copy().ravel().tolist())]).T
Z2 = []
for i in xrange(A.shape[0]):
    col = A.T[:,i] # by analogy to g being a column vector
    Z2.append(TDMASolve(D['lb'].copy(), D['db'].copy(), D['ub'].copy(), col))
Z2 = np.array(Z2).T

z3 = solve(np.array(np.eye(k) +np.dot(A,Z2),dtype=np.float64),np.array(np.dot(A,z1),dtype=np.float64))
x_star = z1 -np.dot(Z2,z3)

print "||x^* struct - x^* generic||_2"
print np.linalg.norm(x_star-x_star_generic)
