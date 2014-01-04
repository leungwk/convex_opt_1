import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import scipy.linalg
from scipy.linalg import cho_factor, cho_solve

def f(A,x):
    """
f(x) could be written as
1^T*log(1 -Ax) -1^T*log(1 -x^2)
"""
    ## weird mix of row-wise and elementwise operations
    # res = -sum(np.log(1 -np.dot(A,x))) -sum(np.log(1 -x**2))
    res = -sum(np.log(1 -np.dot(A,x))) -sum(np.log(1 +x)) -sum(np.log(1 -x))
    return res


def grad_f(A,x):
    """
\grad f(x)
= -\grad 1^Tlog(1 -Ax) -\grad 1^Tlog(1 -x^2)
= -[-A^T*1/(1 -Ax)] -[1/(1 -x^2)(-2x)]
= A^T*1/(1 -Ax) +2x/(1 -x^2)
= A^T*1/(1 -Ax) +(1+x+x-1)/[(1-x)(1+x)]
= A^T*1/(1 -Ax) +1/(1-x) -1/(1+x)
"""
    ## unsure why A^T (ie. why is the transpose needed)
    # return np.dot(A.T,1./(1 -np.dot(A,x))) +2.*x/(1 -np.dot(x.T,x))
    return np.dot(A.T,1./(1 -np.dot(A,x))) -1./(1 +x) +1./(1 -x) # why does this lead to a lower objective function? (maybe the last two terms balance each other in errors, or fewer multiplications involved)


def hessian(A,x):
    """
\grad^2 f(x)
= \grad \grad f(x)
= A^T \grad (1 -Ax)^{-1} +\grad (1 -x)^{-1} -\grad (1 +x)^{-1}
= A^T (-1) diag((1 -Ax)^{-2})(-A) +diag((-1)(1 -x)^{-2}(-1) -(-1)(1 +x)^{-2}(+1))
= A^T diag((1 -Ax)^{-2}) A +(1 -x)^{-2} +(1 +x)^{-2}
% remember that the off diagonals go to zero because the mixed derivates don't have such terms
"""
    d = 1./(1 -np.dot(A,x))
    return np.dot(np.dot(A.T,np.diag(np.power(d.ravel(),2))),A) +np.diag(1./np.power((1 +x),2) +1./np.power((1 -x),2))







def line_search(A,x,delta_x,grad_x,alpha,beta):
    """backtracking line search"""
    t = 1.
    lhs, rhs = 1,0
    while lhs > rhs:
        lhs = f(A, x +t*delta_x)
        if np.isnan(lhs):
            lhs = np.inf # otherwise the condition might not break eventually
        rhs = f(A,x) +alpha*t*np.dot(grad_x.T,delta_x)
        if np.isnan(rhs):
            rhs = np.inf
        t *= beta
    return t



def gradient_method(A, x, alpha=0.25,beta=0.5):
    """aka. gradient descent"""
    n_iter = long(1e8)
    iters = xrange(n_iter)
    eta = 1e-03
    delta_norm = 5e-6
    prev_n2, cur_n2 = np.inf, np.inf
    max_iter = 100

    acc_stat = []
    for iter_n in iters:

        tmp = grad_f(A,x)
        if tmp is None:
            delta_x = -np.array([[np.inf]*len(x)]).T
        else:
            delta_x = -tmp

        ## stopping conditions
        if iter_n > max_iter:
            break

        n2 = np.linalg.norm(-delta_x)
        if n2 <= eta: # gradient
            break
        prev_n2 = cur_n2
        cur_n2 = n2
        if np.abs(cur_n2 -prev_n2) <= delta_norm: # change in gradient
            break

        t = line_search(A,x,delta_x,grad_f(A,x),alpha,beta)
        x += t*delta_x # take step
        acc_stat.append( (float(f(A,x)), iter_n, t) )
    df = pd.DataFrame(acc_stat, columns=['f','k','t'])
    df.set_index('k',inplace=True)
    df['alpha'] = alpha
    df['beta'] = beta
    return x, df


def solve(A, x0_in, func):
    x_stars = []
    df_stats = []
    x0_orig = x0_in.copy()
    ## try several alpha and beta
    for alpha, beta in itertools.product([0.01,0.1,0.2,0.4],[0.1,0.5,0.9]):
        x0 = x0_orig.copy()
        x_star, df_stat = func(A, x0, alpha, beta)
        x_stars.append(x_star)
        df_stats.append(df_stat)
    return x_stars, df_stats


def solve_reuse_hessian(A, x0_in):
    x_stars = []
    df_stats = []
    x0_orig = x0_in.copy()
    ## try several alpha and beta
    for N in [1,2,15,30,50]:
        x0 = x0_orig.copy()
        x_star, df_stat = newtons_method(A, x0, N, alpha=0.2, beta=0.5)
        x_stars.append(x_star)
        df_stat['N'] = N
        df_stats.append(df_stat)
    return x_stars, df_stats


def f_iters_reuse_hessian(df_stats, n, title='', p_star=None, path_template=''):
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    ax.set_xlabel('~flops')
    ax.set_ylabel(r'$f(x^{(k)})$' if p_star is None else r'$f(x^{(k)}) -p^*$')
    for df in df_stats:
        N = df['N'][0]
        label = str(N)
        xs = []
        ys = []
        tot_flop = 0
        for k, row in df.iterrows():
            if k % N == 0:
                tot_flop += np.power(n,3)/3.
            tot_flop += 2*np.power(n,2)
            xs.append(tot_flop)
            ys.append(row['f'] -(0 if p_star is None else p_star))
        ax.plot(xs, ys, label=label)
    ax.set_xlim([0,2.5e7])
    if p_star is not None:
        ax.set_yscale('log')
    ax.legend(title='N',prop={'size':10})
    # plt.show()
    plt.savefig(path_template,bbox_inches='tight',pad_inches=0)
    print "Wrote to {0}".format(path_template)


def f_iters(df_stats, title='', path_template=''):
    """"""
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title) # "r'" necessary otherwise "ParseFatalException: Expected end of math '$'"
    ax.set_xlabel('k')
    ax.set_ylabel(r'$f(x^{(k)})$')
    lines = []
    labels = []
    for df in df_stats:
        label = "{0},{1}".format(df['alpha'][0], df['beta'][0])
        l, = ax.plot(df.index, df['f'], label=label) # use "l," not "l", otheriwse it will say "UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0xbc9306c>]"
        lines.append(l)
        labels.append(label)
    ax.legend(title=r'$\alpha,\beta$',prop={'size':10})
    # ax.legend(lines, labels)
    # plt.show()
    plt.savefig(path_template,bbox_inches='tight',pad_inches=0)
    print "Wrote to {0}".format(path_template)


def f_ps(df_stats, A, x_stars, title, path_template):
    # plt.ion()
    # plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    for df, x_star in zip(df_stats,x_stars):
        ys = df['f'] -float(f(A,x_star))
        label = "{0},{1}".format(df['alpha'][0], df['beta'][0])
        ax.plot(df.index, ys, label=label)
        ax.set_yscale('log')
        ax.set_xlim([0,max(df.index)])
        ax.set_ylim([1e-8,max(ys)])
    ax.set_xlabel('k')
    ax.set_ylabel(r'$f(x^{(k)}) -p^*$')
    ax.legend(title=r'$\alpha,\beta$',prop={'size':10},loc='lower left')
    # plt.show()
    plt.savefig(path_template,bbox_inches='tight',pad_inches=0)
    print "Wrote to {0}".format(path_template)
    

def f_t(df):
    plt.ion()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.stem(df.index, df['t'], '-.')
    ax.xlim(0,max(df.index))
    ax.ylim(0,max(df['t']))
    plt.show()


def newtons_method(A,x,N=None,alpha=0.25,beta=0.5):
    """N denotes how often to update the Hessian. N == 1 means continually. N == None means skip factorization"""
    epsilon = 1e-8
    max_iter = 100

    acc_stat = []
    cnt_H = None
    H_old = None
    L_struct = None
    for iter_n in xrange(max_iter):
        ## for re-using the Hessian
        if not cnt_H:
            H = hessian(A,x)
            if N is not None:
                L_struct = cho_factor(H)
            H_old = H
            cnt_H = N
        if cnt_H is not None:
            cnt_H -= 1
        H = H_old
        g = grad_f(A,x)
        delta_nt = -cho_solve(L_struct,g) if N is not None else -scipy.linalg.solve(H,g)
        lambda_2 = np.dot(g.T,-delta_nt)

        # stop?
        if lambda_2/2. <= epsilon:
            break

        t = line_search(A,x,delta_nt,g,alpha,beta)
        x += t*delta_nt
        acc_stat.append( (float(f(A,x)), iter_n, t) )
    df = pd.DataFrame(acc_stat, columns=['f','k','t'])
    df.set_index('k',inplace=True)
    df['alpha'] = alpha
    df['beta'] = beta
    return x, df
