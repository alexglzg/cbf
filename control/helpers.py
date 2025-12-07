import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# LSE max and min approximation
def basic(e):
  m = ca.mmax(e)
  return m+ca.log(ca.sum1(ca.exp(e-m)))

def smooth_max(e, alpha=10):
  return 1/alpha*basic(alpha*e)
def smooth_min(e, alpha=10):
  return -smooth_max(-e, alpha)

def get_transformed_polytope(A_obs, b_obs, x, y, theta):
    """
    Returns (A_obs_world, b_obs_world) for a rectangle at (x, y, theta)
    """
    # 2D rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    # Transform A and b
    A_obs_world = A_obs @ R.T
    b_obs_world = b_obs + A_obs_world @ np.array([x, y])
    return A_obs_world, b_obs_world

def get_transformed_polytope_ca(A_obs, b_obs):
    """
    Returns (A_obs_world, b_obs_world) for a rectangle at (x, y, theta)
    """
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    theta = ca.MX.sym('theta')

    # 2D rotation matrix
    R = ca.horzcat(
        ca.vertcat(ca.cos(theta), -ca.sin(theta)),
        ca.vertcat(ca.sin(theta),  ca.cos(theta))
    )

    # Transform A and b
    A_obs_world = A_obs @ R.T
    b_obs_world = b_obs + A_obs_world @ ca.vertcat(x, y)
    func = ca.Function('transform_polytope', [x, y, theta], [A_obs_world, b_obs_world])
    return func

def robot_vertices_ca(x, y, theta, L, W):
    """
    CasADi-compatible: Returns (4,2) MX array of rectangle vertices for robot at (x, y, theta).
    Vertices are ordered counter-clockwise starting from front-right.
    """
    # Half-dimensions
    l2 = L / 2
    w2 = W / 2

    # Rectangle corners in body frame (counter-clockwise)
    corners = ca.horzcat(
        ca.vertcat([ l2,  w2]),
        ca.vertcat([ l2, -w2]),
        ca.vertcat([-l2, -w2]),
        ca.vertcat([-l2,  w2])
    ).T

    # Rotation matrix
    c = ca.cos(theta)
    s = ca.sin(theta)
    R = ca.horzcat(ca.vertcat(c, -s), ca.vertcat(s, c))

    # Transform corners to world frame
    verts = ca.mtimes(corners, R.T) + ca.repmat(ca.vertcat(x, y), 1, 4).T

    return verts  # shape (4,2)

def rectangle_vertices_from_polytope_ca(A, b):
    """
    CasADi-compatible: Compute rectangle vertices from polytope Ax <= b.
    Assumes A is (4,2), b is (4,).
    Returns a (4,2) CasADi MX matrix of vertices.
    """
    verts = []
    for i in range(4):
        j = (i + 1) % 4
        # Stack the two lines
        A_stack = ca.vertcat(A[i, :], A[j, :])
        b_stack = ca.vertcat(b[i], b[j])
        # Solve A_stack @ v = b_stack
        v = ca.solve(A_stack, b_stack)
        verts.append(v.T)
    return ca.vertcat(*verts)

def rectangle_vertices_from_polytope(A, b):
    """
    Convert polytope defined by Ax <= b into corner vertices (for a rectangle).
    Assumes 2D box shape.
    """
    from scipy.spatial import HalfspaceIntersection, ConvexHull
    import matplotlib.pyplot as plt

    # Build halfspaces
    halfspaces = np.hstack((A, -b.reshape(-1, 1)))

    # Find an interior point (center)
    center = np.linalg.lstsq(A, b, rcond=None)[0] * 0.5

    hs = HalfspaceIntersection(halfspaces, center)
    hull = ConvexHull(hs.intersections)
    vertices = hs.intersections[hull.vertices]
    return vertices

def transform_box(vertices, x, y, theta):
    """Rotate and translate box vertices to world frame"""
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    transformed = (R @ vertices.T).T + np.array([x, y])
    return transformed

def get_dist_region_to_region(mat_A1, vec_b1, mat_A2, vec_b2):
    opti = ca.Opti()
    # variables and cost
    point1 = opti.variable(mat_A1.shape[-1], 1)
    point2 = opti.variable(mat_A2.shape[-1], 1)
    cost = 0
    # constraints
    constraint1 = ca.mtimes(mat_A1, point1) <= vec_b1
    constraint2 = ca.mtimes(mat_A2, point2) <= vec_b2
    opti.subject_to(constraint1)
    opti.subject_to(constraint2)
    dist_vec = point1 - point2
    cost += ca.mtimes(dist_vec.T, dist_vec)
    # solve optimization
    opti.minimize(cost)
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", option)
    opt_sol = opti.solve()
    # minimum distance & dual variables
    dist = opt_sol.value(ca.norm_2(dist_vec))
    if dist > 0:
        lamb = opt_sol.value(opti.dual(constraint1)) / (2 * dist)
        mu = opt_sol.value(opti.dual(constraint2)) / (2 * dist)
    else:
        lamb = np.zeros(shape=(mat_A1.shape[0],))
        mu = np.zeros(shape=(mat_A2.shape[0],))
    return dist, lamb, mu

def plot_rectangle_from_normals_and_offsets(A, b, ax, color='red'):
    """
    Plot a rectangle defined by normal vectors A and offsets b.
    
    Args:
        A: List of normal vectors (2D list or array of shape (4, 2)).
        b: List of offsets (1D list or array of shape (4,)).
        ax: Matplotlib axis to plot on.
        color: Color of the rectangle edges.
    """
    # Convert A and b to numpy arrays for easier manipulation
    A = np.array(A)
    b = np.array(b)
    
    # Find the vertices of the rectangle by solving intersections of the planes
    vertices = []
    for i in range(len(A)):
        # Get the current and next normal vectors and offsets
        A1, b1 = A[i], b[i]
        A2, b2 = A[(i + 1) % len(A)], b[(i + 1) % len(b)]
        
        # Solve the linear system [A1; A2] * [x; y] = [-b1; -b2]
        A_matrix = np.array([A1, A2])
        b_vector = np.array([-b1, -b2])
        vertex = np.linalg.solve(A_matrix, b_vector)
        vertices.append(vertex)
    
    # Convert vertices to a numpy array for plotting
    vertices = np.array(vertices)
    
    # Plot the rectangle
    polygon = plt.Polygon(vertices, closed=True, fill=None, edgecolor=color)
    ax.add_patch(polygon)