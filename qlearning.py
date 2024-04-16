import numpy as np
import itertools
from scipy.optimize import linprog

##############################################################################
# Global Variables
X = ['B','G']
U = [0, 1]
kernel = np.array([[[0.5, 0.5], 
                      [0.1, 0.9]] , [[0.2, 0.8], 
                                     [0.9, 0.1]] ])
##############################################################################

##############################################################################
# Global functions to be used by all methods
def cost(x, u, eta):
    return eta*u + (-1 if x == 'G' and u == 1 else 0)

# T for transition kernel
def T(x_iplus1, x_i, u):
    i = 0 if x_i == 'B' else 1
    j = 0 if x_iplus1 == 'B' else 1
    return kernel[u][i][j]
##############################################################################

##############################################################################
# Implementation of Value Iteration
def value_iteration(eta, beta, epochs=100000):
    V = {current_state: 0 for current_state in X}
    for _ in range(epochs):
        V_new = {}
        for x_t in X:
            V_new[x_t] = min([cost(x_t, u, eta) + beta * sum([T(x_tplus1, x_t, u) * V[x_tplus1] for x_tplus1 in X]) for u in U])
        V = V_new
    policy = {} 
    for x_t in X:
        policy[x_t] = np.argmin([cost(x_t, u, eta) + beta * sum([T(x_tplus1, x_t, u) * V[x_tplus1] for x_tplus1 in X]) for u in U]) 
    return V, policy
##############################################################################

##############################################################################
# Implementation of Policy Iteration along with a helper function to 
# evaluate the policy
def policy_eval(policy, eta, beta, epochs=100000):
    V = {x: 0 for x in X}
    for _ in range(epochs):
        V_new = {}
        for x_t in X:
            u = policy[x_t]
            V_new[x_t] = cost(x_t, u, eta) + beta * sum([T(x_tplus1, x_t, u) * V[x_tplus1] for x_tplus1 in X])
        V = V_new
    return V

def policy_iteration(eta, beta, epochs=100000):
    policy = {x: 0 for x in X}
    stable=False
    i = 0
    while (not stable) and (i:=i+1 <= epochs):
        V = policy_eval(policy, eta, beta)
        stable=True
        for x_t in X:
            old_action = policy[x_t]
            policy[x_t] = np.argmin([cost(x_t, u, eta) + beta * sum([T(x_tplus1, x_t, u) * V[x_tplus1] for x_tplus1 in X]) for u in U])
            if old_action != policy[x_t]:
                stable=False 
    return V, policy
##############################################################################

##############################################################################
# Implementation of Q-Learning along with the alpha coefficient helper function
def alpha(x, u, history):
    total = 1
    for pair in history:
        if pair == [x, u]:
            total += 1
    return 1 / total

def q_learning(eta, beta, epochs=10000):
    X_dict = {x: idx for idx, x in enumerate(X)}
    U_dict = {u: idx for idx, u in enumerate(U)}
    Qtable = np.zeros((len(X), len(U)))
    x_t = np.random.choice(X)
    hist = []

    for _ in range(epochs):
        x_t_idx = X_dict[x_t]

        u = np.random.choice(U) if np.random.rand() < 1 else np.argmin(Qtable[x_t_idx, :])
        hist.append([x_t, u])

        u_idx = U_dict[u]

        p_x_tplus1 = [T(x_tplus1, x_t, u) for x_tplus1 in X]
        x_tplus1 = np.random.choice(X, p=p_x_tplus1)
        x_tplus1_idx = X_dict[x_tplus1]

        Qtable[x_t_idx, u_idx] += alpha(x_t, u, hist) * (cost(x_t, u, eta) + beta * np.min(Qtable[x_tplus1_idx, :]) - Qtable[x_t_idx, u_idx])
        x_t = x_tplus1

    # Find the optimal policy using Q table
    optimal_policy = {x: U[np.argmin(Qtable[X_dict[x], :])] for x in X}
    optimal_cost = {x: Qtable[X_dict[x], np.argmin(Qtable[X_dict[x], :])] for x in X}
    
    return optimal_policy, optimal_cost
##############################################################################

##############################################################################
# Implementation of Convex Analytic method
def convex_analytic(eta):
    n = len(X) * len(U)
    A = np.zeros((len(X) + 1, n))
    A[0, :] = 1
    for idx, x in enumerate(X):
        for u in U:
            A[idx + 1, u + 2 * (x == 'G')] = 1 - T(x, x, u)
            A[idx + 1, u + 2 * (x == 'B')] = -T(x, 'B' if x == 'G' else 'G', u)
    b = np.zeros(len(X) + 1)
    b[0] = 1
    c = np.array([cost(x, u, eta) for x, u in itertools.product(X, U)])
    res = linprog(c, A_eq=A, b_eq=b)
    if res.success:
        p = res.x
        policy = {}
        policy['G'] = round(p[1] / (p[0] + p[1]))
        policy['B'] = round(p[3] / (p[2] + p[3]))
        return policy
    else:
        raise ValueError(f"Optimization failed: {res.message}")
##############################################################################

##############################################################################
# Functions that run the above learning approaches
def run_value_iteration(beta, etas):
    print("##############################################################################")
    print("Running Value Iteration")
    for eta in etas:
        V, policy = value_iteration(eta, beta)
        print(f"Results for η = {eta}:")
        print(f"Optimal Cost: {V}")
        print(f"Optimal Policy: {policy}")
        print()
    print("##############################################################################")

def run_policy_iteration(beta, etas):
    print("##############################################################################")
    print("Running Policy Iteration")
    for eta in etas:
        V, policy = policy_iteration(eta, beta)
        print(f"Results for η = {eta}:")
        print(f"Optimal Cost: {V}")
        print(f"Optimal Policy: {policy}")
        print()
    print("##############################################################################")

def run_qlearning():
    print("##############################################################################")
    print("Running Q-Learning")
    optimal_policy, optimal_cost = q_learning(0.7, 0.1)
    print("Optimal Cost:", optimal_cost)
    print("Optimal Policy:", optimal_policy)
    print()
    print("##############################################################################")

def run_convex_analytic(etas):
    print("##############################################################################")
    print("Running Convex Analytic Method")
    for eta in etas:
        policy = convex_analytic(eta)
        print(f"Results for η = {eta}:")
        print(f"Optimal Policy: {policy}")
        print()
    print("##############################################################################")
#############################################################################

##############################################################################
# Execution script that runs everything
def main(): 
    # Chosen arbitrarily
    beta = 0.5
    
    # As given in the assignment
    etas = [0.9, 0.7, 0.01]

    # Running all the methods
    run_value_iteration(beta, etas)     #Q1 (a)
    run_policy_iteration(beta, etas)    #Q1 (b)
    run_qlearning()                     #Q1 (c)
    run_convex_analytic(etas)           #Q2
##############################################################################

if __name__=="__main__":
    main()
