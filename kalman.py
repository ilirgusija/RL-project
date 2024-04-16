import numpy as np 
import matplotlib.pyplot as plt

def kalman_filter(A, C, Q, R, T=1000):
    x = np.zeros((4, T))
    y = np.zeros((1, T))
    x_hat = np.zeros((4, T))
    sigma = np.zeros((4, 4, T))
    x[:, 0] = np.zeros((4,))
    sigma[:, :, 0] = np.eye(4)
    for t in range(T - 1):
        w = np.random.normal(0, 1, (4,))
        v = np.random.normal(0, 1, (1,))

        x[:, t + 1] = A @ x[:, t] + w
        y[:, t + 1] = C @ x[:, t + 1] + v
        x_hat_prior = A @ x_hat[:, t]
        sigma = A @ sigma[:, :, t] @ A.T + Q
        K = sigma @ C.T @ np.linalg.inv(C @ sigma @ C.T + R)

        x_hat[:, t + 1] = x_hat_prior + K @ (y[:, t + 1] - C @ x_hat_prior)
        sigma[:, :, t + 1] = (np.eye(4) - K @ C) @ sigma

    plt.figure()
    plt.plot(x[0, :])
    plt.xlabel('Time')
    plt.ylabel('x_t')
    plt.show()

    plt.figure()
    plt.plot(x_hat[0, :])
    plt.xlabel('Time')
    plt.ylabel('m_t')
    plt.show()

    plt.figure()
    plt.plot(x[0, :] - x_hat[0, :])
    plt.xlabel('Time')
    plt.ylabel('xt - m_t')
    plt.show()

def controllable(A, B):
    n = A.shape[0]
    ctrlability_matr = np.column_stack([B] + [A @ B])
    for _ in range(2, n):
        ctrlability_matr = np.column_stack((ctrlability_matr, A @ ctrlability_matr[:, -1]))
    return np.linalg.matrix_rank(ctrlability_matr) == n

def updateCov(sigma_t, A, C, W, V):
    sigma = A @ sigma_t @ A.T + W
    K = sigma @ C.T @ np.linalg.inv(C @ sigma @ C.T + V)
    sigma_tplus1 = (np.eye(4) - K @ C) @ sigma
    return sigma_tplus1

def riccati(A, C, Q, R, T = 5000):
    sigma = np.zeros((4, 4, T))
    sigma[:, :, 0] = np.eye(4)
    for t in range(T - 1):
        sigma[:, :, t + 1] = updateCov(sigma[:, :, t], A, C, Q, R)
    convergence_diff = np.abs(sigma[:, :, -1] - sigma[:, :, -2])
    print(f"Final difference in the Cov matrices: {convergence_diff}")

    plt.figure()
    plt.plot(np.max(np.abs(np.diff(sigma, axis=2)), axis=(0, 1)))
    plt.xlabel('Time')
    plt.ylabel('Max Absolute Difference in Cov')
    plt.title('Max Absolute Difference in Cov over time')
    plt.show()

def main():
    A = np.array([[2,1,0,0], [0, 2, 1, 0], [0, 0, 2, 1], [0, 0, 0, 4]])
    B = np.eye(4)
    C = np.array([2, 0, 0, 0])
    Q = np.eye(4)
    R = np.array([[1]])

    kalman_filter(A, C, Q, R)
    ctrlable = controllable(A, B)
    obsrvble = controllable(A.T, C.T)
    riccati(A, C, Q, R)

    print(f"Controllable: {ctrlable}")
    print(f"Observable: {obsrvble}")
