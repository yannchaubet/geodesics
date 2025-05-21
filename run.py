from utils import *
from main import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""
Parameters
"""
n = 200  # size of grid
o = (n - 1) / 2  # size of the half grid
alpha_2d = 0.3  # opacity of the 3d figure
alpha_3d = 0.5  # opacity of the 3d figure
beta = 10  # parameter for the function f(x,y) = beta * (x^2 - y^2)
gamma = 5
delta = 13e-2  # parameters for the function f(x,y) = delta * cos(x / gamma) cos(y / gamma)
beta, gamma, delta = beta / (n ** 2), gamma, delta * n  # renormalisation des paramètres
x0 = (o + int(n / 4), o)  # origin of the grid
norm = 2  # norm of the initial vectors
dt = 0.1  # time step parameter
N = 4000  # number of iterations
Mv = 15  # number of geodesics computed
N_contour = 30  # number of lines used to draw the surface in 3d
k = 20  # smoothing parameter for the random surface
m = -20  # minimal possible height for the random surface
M = 20  # maximal possible height for the random surface


# Define the set of initial vectors
v0s = [norm * unit(theta + np.pi / 4) for theta in np.linspace(0, 2 * np.pi, Mv + 1)]

"""
Define the axes and b centered at the origin
"""
a = np.arange(n) - o
b = np.arange(n) - o
# print(x, y)
p = np.meshgrid(a, b)  # coordinates of the surface


"""
Choose the height function f for the surface
"""
# f = beta * (p[0] ** 2 - p[1] ** 2)  # selle de cheval
f = gamma * np.cos(p[0] / delta) * np.cos(p[1] / delta)  # wave surface
f = f + random_graph(n, m, M, k=k, lissage=True)  # random surface
# f = np.zeros((n, n))  # flat surface


"""
Define the metric on the surface z = f(x,y)
"""
g = metric_graph(f)
M = Metric(g)
fig1 = plt.figure(1)
plt.imshow(f, cmap="binary", alpha=alpha_2d)  # plot the height function in 2d
plt.colorbar()


"""
Initialize the list of geodesics and heights
"""
geodesics = []
heights_list = []


"""
Plot the surface in 3d
"""
fig2 = plt.figure(2)
ax = plt.axes(projection='3d')
ax.set_axis_off()
ax.contour3D(a, b, f, N_contour, cmap='binary', alpha=alpha_3d)


"""
Compute the geodesics
"""
mv = 0  # initialize color parameter
for v0 in v0s:
    cmap = plt.get_cmap("twilight")
    trajectory, velocities, heights = M.compute_geodesic(x0=x0, v0=v0, dt=dt, N=N, f=f)
    geodesics += [trajectory]
    heights_list += [heights]
    # times = np.linspace(0, N * dt, len(trajectory[:, 0]))
    # colors = iter(cm.rainbow(times))
    plt.figure(1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], '.', color=cmap(mv/Mv))
    plt.figure(2)
    ax.scatter(trajectory[:, 0] - o, trajectory[:, 1] - o, heights, color=cmap(mv / Mv))
    mv += 1

plt.show()
