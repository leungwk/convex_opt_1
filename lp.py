import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import itertools

"""
Solve a standard form LP using the barrier method
"""

def _f(c,x):
    """Note: For part 2, t*c will give the log barrier approximation of the LP. The original objective is c^Tx"""
    return np.dot(c.T,x) -np.sum(np.log(x))


def _grad(c,x):
    return c -1./x


def _hessian(x):
    if len(x.shape) == 2:
        x = x.ravel()
    return np.diag(1/(np.power(x,2)))


def line_search(c, x, grad_x, delta_x_nt, alpha, beta):
    t = 1.
    ## select t such that it is feasible (move "x+" into domain)
    while any(x +t*delta_x_nt <= 0):
        t *= beta

    # ## part 1, reduced number of terms ... appears to have no effect on suboptimal output
    # while any(np.dot(c.T,t*delta_x_nt) -np.sum(np.log(x +t*delta_x_nt)) +np.sum(np.log(x)) -alpha*t*np.dot(grad_x.T,delta_x_nt) > 0):
    #     t *= beta
    # return t

    while True:
        lhs = _f(c, x +t*delta_x_nt)
        if np.isnan(lhs):
            lhs = np.inf # otherwise the condition might not break eventually
        rhs = _f(c, x) +alpha*t*np.dot(grad_x.T,delta_x_nt)
        if np.isnan(rhs):
            rhs = np.inf
        if not (lhs > rhs):
            break
        t *= beta
    return t


def lp_center(A, b, c, x, epsilon=1e-6, max_iter=100, alpha=0.25, beta=0.5):
    acc_stats = []
    for iter_n in xrange(max_iter):
        ## solve kkt system
        H = _hessian(x)
        H_inv = np.diag(1./np.diag(H)) # avoid double inversion
        # H_inv = np.diag((x**2).ravel()) # doesn't really do much
        A_H_inv = np.dot(A,H_inv)
        grad_x = _grad(c,x)
        #
        t1 = np.linalg.inv(np.dot(A_H_inv,A.T))
        t2 = -np.dot(A_H_inv,grad_x)
        w = np.dot(t1,t2)
        # t1 = np.dot(A_H_inv,A.T)
        # w = scipy.linalg.solve(t1,t2) # no difference in output
        #
        delta_x_nt = -np.dot(H_inv,grad_x +np.dot(A.T,w))

        lambda_2 = -np.dot(delta_x_nt.T,grad_x)
        if lambda_2/2. <= epsilon:
            break

        t = line_search(c, x, grad_x, delta_x_nt, alpha, beta)
        x += t*delta_x_nt

        row = (float(lambda_2/2.), iter_n) # \lambda_2/2 est. of f(x) -p^* using quad approx of f at x
        acc_stats.append(row)

    df = pd.DataFrame(acc_stats, columns=['l2_2', 'k'])
    df.set_index('k',inplace=True)
    return x, w, df


def run_lp_center(A, b, c, x0):
    acc_df_stats = []
    ops_params = []
    for alpha, beta in itertools.product([0.01,0.1,0.3],[0.1,0.5,0.8]):
        x_star, nu_star, df_stat = lp_center(A, b, c, x0.copy(), alpha=alpha, beta=beta)
        acc_df_stats.append(df_stat)
        ops_params.append((x_star, nu_star, alpha, beta, _f(c, x_star)))
    df_op = pd.DataFrame(ops_params, columns=['x_star', 'nu_star', 'alpha', 'beta', 'f'])
    return acc_df_stats, df_op


def _plot_lp_center(dfs, df_op, image_path):
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    rainbow = ax._get_lines.color_cycle
    markers = itertools.cycle(['o',">","v","<","^","s","+","x","D"])
    line_cycle = itertools.cycle(["-","-.","--",":",])
    for df, c, alpha, beta, marker, ls in zip(dfs, rainbow, df_op['alpha'], df_op['beta'], markers, line_cycle):
        label = "{0},{1}".format(alpha, beta)
        ax.plot(df.index, df['l2_2'], color=c, label=label, marker=marker, linestyle=ls, markersize=4)
    ax.set_yscale('log')
    ax.set_xlabel('k')
    ax.set_ylabel(r'$\lambda_2/2$')
    ax.set_title('Newton\'s method for centering step using LP')
    ax.legend(title=r'$\alpha,\beta$',prop={'size':10},loc='upper right')
    plt.savefig(image_path,bbox_inches='tight',pad_inches=0)
    print "Wrote to {0}".format(image_path)
    # plt.show()


def lp_strict(A, b, c, x, t=1, mu=10, epsilon=1e-3):
    """Assuming a strictly feasible starting point"""
    n = len(x)
    acc_stats = []
    while True:
        x_star, nu_star, df_stat = lp_center(A, b, c*t, x, alpha=0.01, beta=0.5) # c*t will create objective tc^Tx -1^Tlog(x), the original objective plus log barrier
        x = x_star
        gap = 1.*n/t
        row = (len(df_stat), gap)
        acc_stats.append( row ) # num Newton steps per centering
        if gap < epsilon:
            break
        t *= mu
    df = pd.DataFrame(acc_stats, columns=['k_newton','gap'])
    return x_star, nu_star, df


def run_lp_strict(A, b, c, x0):
    acc_df_stats = []
    op_params = []
    for mu in [2,10,50,200]:
        x_star_strict, nu_star_strict, df_stats_strict = lp_strict(A, b, c, x0.copy(), mu=mu)
        acc_df_stats.append( df_stats_strict )
        op_params.append( (x_star_strict, nu_star_strict, mu) )
    df_op = pd.DataFrame(op_params, columns=['x_star','nu_star','mu'])
    return acc_df_stats, df_op


def _plot_lp_strict(dfs, df_op, image_path):
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    rainbow = ax._get_lines.color_cycle
    markers = itertools.cycle(['o',">","v","<","^","s","+","x","D"])
    line_cycle = itertools.cycle(["-","-.","--",":",])
    for df, mu, c, marker, ls in zip(dfs, df_op['mu'], rainbow, markers, line_cycle):
        label = str(mu)
        ax.step(df['k_newton'].cumsum(), df['gap'], color=c, label=label, linestyle=ls, markersize=4)
    ax.set_yscale('log')
    ax.set_xlabel('Newton iterations')
    ax.set_ylabel('duality gap')
    ax.set_title('Effect of step size in LP using barrier method')
    ax.legend(title='mu',prop={'size':10},loc='upper right')
    plt.savefig(image_path,bbox_inches='tight',pad_inches=0)
    print "Wrote to {0}".format(image_path)
