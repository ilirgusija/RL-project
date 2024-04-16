from scipy.integrate import quad
import numpy as np

X = ['B','G']
U = [0, 1]

# Transition kernel
def T(bounds, u):
    lower_bound, upper_bound = bounds
    if u == 1:
        integrand = lambda z: 2 * (1 - z) if lower_bound <= z <= upper_bound else 0
    else:  # u == 0
        integrand = lambda z: 2 * z if lower_bound <= z <= upper_bound else 0
    result, _ = quad(integrand, lower_bound, upper_bound)
    return result

def cost(x, u, eta):
    return -x * u + eta * u

def quantize_X(X_size):
    return np.linspace(0, 1, X_size + 1)[:-1]

def q_learning(X_Q, eta, beta, epochs=1000):
    X_dict = {x: idx for idx, x in enumerate(X_Q)}
    Q = np.zeros((len(X_Q), len(U)))
    
    for _ in range(epochs):
        x_t = np.random.choice(X_Q)
        x_t_idx = X_dict[x_t]     
        u = np.argmax(Q[x_t_idx])
        
        bounds = [(x_t, x_t + 1/len(X_Q)) for x_t in X_Q]
        p_x_tplus1 = [T(bounds[X_dict[x_t]], u) for x_t in X_Q]
        p_x_tplus1 = np.array(p_x_tplus1)
        p_x_tplus1 /= p_x_tplus1.sum()
        x_tplus1 = np.random.choice(X_Q, p=p_x_tplus1) 
        x_tplus1_idx = X_dict[x_tplus1]
        
        Q[x_t_idx, u] += eta * (cost(x_t, u, eta) + beta * np.min(Q[x_tplus1_idx]) - Q[x_t_idx, u])
    
    optimal_policy = {x: U[np.argmin(Q[X_dict[x]])] for x in X_Q}
    optimal_cost = {x: np.min(Q[X_dict[x]]) for x in X_Q}
    
    return optimal_policy, optimal_cost

def main():
    granularity = [10, 20, 50, 100]

    for n in granularity:
        X_Q = quantize_X(n)
        
        optimal_policy, optimal_cost = q_learning(X_Q, 0.7, 0.1)

        print(f"Granularity Level: {n}")
        print("Optimal Cost:", optimal_cost)
        print("Optimal Policy:", optimal_policy)

if __name__ == "__main__":
    main()
