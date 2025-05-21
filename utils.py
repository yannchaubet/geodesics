import numpy as np
from scipy import signal


def gradient(f):

    # reshape
    s = np.shape(f)
    s_bis = np.array(np.shape(f)) + 2
    z = np.zeros(s_bis)
    z[1:s[0] + 1, 1:s[1] + 1] = f

    # shift
    z_left = np.zeros(s_bis)
    z_left[0:s[0], 1:s[1] + 1] = f
    z_right = np.zeros(s_bis)
    z_right[2:s[0] + 2, 1:s[1] + 1] = f
    z_down = np.zeros(s_bis)
    z_down[1:s[0] + 1, 0:s[1]] = f
    z_up = np.zeros(s_bis)
    z_up[1:s[0] + 1, 2:s[1] + 2] = f

    # compute gradient
    grad_x = ((z_left - z_right) / 2)[1:s[0] + 1, 1:s[1] + 1]
    grad_y = ((z_down - z_up) / 2)[1:s[0] + 1, 1:s[1] + 1]

    return [grad_x, grad_y]


def phase(xi, n):
    m = 2 * n + 1
    xi_x, xi_y = xi
    z = np.zeros((m, m))
    origin = np.array([n, n])

    for i in range(m):

        for j in range(m):
            z[i, j] = ((j - n) * xi_x + (i - n) * xi_y)
            # z[i, j] = (j - n) ** 2 + (i - n) ** 2

    return z[:, ::-1]


def metric_graph(f):
    m, n = np.shape(f)
    # print(m, n)
    g = np.zeros((m, n, 2, 2))
    id = np.diag([1, 1])
    grad = np.zeros((m, n, 2, 2))
    grad_f = gradient(f)
    # print(np.shape(grad_f[0]))

    for i in range(m):

        for j in range(n):
            # print(i, j)
            grad[i, j] = np.array([[grad_f[0][i, j] * grad_f[0][i, j],
                                    grad_f[0][i, j] * grad_f[1][i, j]],
                                   [grad_f[1][i, j] * grad_f[0][i, j],
                                    grad_f[1][i, j] * grad_f[1][i, j]]])

    g[:, :] = id + grad[:, :]
    return g


def unit(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def kernel(n, kern="exponential"):
    kernel = np.zeros((n, n))
    o = int((n - 1) / 2)

    for i in range(n):
        for j in range(n):

            if kern == "exponential":
                kernel[i, j] = np.exp(- np.abs(i - o) - np.abs(j - o))

            elif kern == "mean":
                kernel[i, j] = 1 / (n ** 2)

    return kernel / np.sum(kernel)


def random_graph(n, m, M, k=10, lissage=True):

    if lissage:
        fin = np.random.random((n - k + 1, n - k + 1)) * (M - m) + m
        noyau = kernel(k, kern="mean")
        f = signal.convolve2d(fin, noyau)

    else:
        fin = np.random.random((n, n)) * (M - m) + m
        f = fin

    return f

