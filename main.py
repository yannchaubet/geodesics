import numpy as np
from utils import gradient


class Hamilton:

    def __init__(self, S_0, dt=0.1):
        self.S = S_0
        self.t = 0
        self.dt = dt

    def iterate(self):
        grad_x, grad_y = gradient(self.S)
        norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
        self.S -= self.dt * norm / 2
        self.t += self.dt

    def solve(self, N):

        for n in range(N):
            self.iterate()


class Metric:

    def __init__(self, g):
        self.g = g
        self.shape = np.shape(self.g)
        self.g_inv = self.compute_inv()
        self.christoffel = self.compute_christoffel()

    def compute_inv(self):
        g_inv = np.copy(self.g)
        g_inv[:, :] = np.linalg.inv(self.g[:, :])
        return g_inv

    def compute_christoffel(self):
        christoffel = np.zeros(self.shape + (2,))

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    g_ij = self.g[:, :, i, j]
                    g_i0 = self.g[:, :, i, 0]
                    g_j0 = self.g[:, :, j, 0]
                    g_i1 = self.g[:, :, i, 1]
                    g_j1 = self.g[:, :, j, 1]
                    christoffel[:, :, i, j, k] = \
                        (1 / 2) * self.g_inv[:, :, k, 0] * (gradient(g_j0)[i] + gradient(g_i0)[j] - gradient(g_ij)[0]) \
                        + (1 / 2) * self.g_inv[:, :, k, 1] * (gradient(g_j1)[i] + gradient(g_i1)[j] - gradient(g_ij)[1])

        return christoffel

    def norm(self, x, v):
        m, n = x
        return np.sum(np.dot(self.g[m, n], v) * v)

    def evolve(self, x, v, dt=0.1):
        m, n = np.int(x[0]), np.int(x[1])
        x_bis = x + dt * v
        v_bis = np.copy(v)
        # print(v_bis)
        # print(x)

        for k in range(2):
            v_bis[k] -= dt * sum(self.christoffel[m, n, i, j, k] * v[i] * v[j] for i in range(2) for j in range(2))
            # print(sum(self.christoffel[m, n, i, j, k] * v[i] * v[j] for i in range(2) for j in range(2)))
            # print(v_bis[k])

        # print(v_bis)

        return x_bis, v_bis

    def compute_geodesic(self, x0, v0, f, dt=0.1, N=10):

        def fu(x):
            a, b = x
            a = int(a)
            b = int(b)
            return f[b, a]

        xt, vt = x0, v0
        z0 = fu(x0)
        trajectory = [x0]
        velocities = [v0]
        heights = [z0]

        for n in range(N):

            if np.abs(int(xt[0])) > 0.95 * self.shape[0] \
                    or np.abs(int(xt[1])) > 0.95 * self.shape[1]\
                    or np.abs(int(xt[0])) < 0.05 * self.shape[0]\
                    or np.abs(int(xt[1])) < 0.05 * self.shape[1]:
                return np.array(trajectory)[:-1], np.array(velocities)[:-1], np.array(heights)[:-1]

            try:
                xt, vt = self.evolve(x=xt, v=vt, dt=dt)
                zt = fu(xt)
                trajectory += [xt]
                velocities += [vt]
                heights += [zt]

            except IndexError:
                return np.array(trajectory)[:-1], np.array(velocities)[:-1], np.array(heights)[:-1]

        return np.array(trajectory), np.array(velocities), np.array(heights)




