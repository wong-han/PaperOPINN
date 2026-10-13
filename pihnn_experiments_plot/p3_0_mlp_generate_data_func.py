import numpy as np
import matplotlib.pyplot as plt



def make_grid(low, high, step):
    """Generate a uniform grid with spacing no greater than step and including both endpoints."""
    L = high - low
    n = max(1, int(np.ceil(L / step)))
    return np.linspace(low, high, n + 1)

def project_to_surface(x0, low, high):
    """
    Project an arbitrary point onto the nearest point on the cube surface.
    - For an exterior point, clamp it to [low, high], which places it on a face, edge, or corner.
    - For an interior point, move it to the nearest face.
    """
    x0 = np.array(x0, dtype=float)
    p = np.clip(x0, low, high)
    inside = np.all((x0 > low) & (x0 < high))
    if inside:
        dists = np.array([x0[0]-low, high-x0[0],
                          x0[1]-low, high-x0[1],
                          x0[2]-low, high-x0[2]])
        k = int(np.argmin(dists))
        if k == 0: p[0] = low
        elif k == 1: p[0] = high
        elif k == 2: p[1] = low
        elif k == 3: p[1] = high
        elif k == 4: p[2] = low
        else:        p[2] = high
    return p

def ring_path(z, X, Y, low, high):
    """Traverse the four edges of a fixed-z cross-section as a closed loop to cover side-surface grid points."""
    pts = []

    for x in X:
        pts.append((x, low, z))

    for y in Y[1:]:
        pts.append((high, y, z))

    for x in X[-2::-1]:
        pts.append((x, high, z))

    for y in Y[-2::-1]:
        pts.append((low, y, z))
    return np.array(pts)

def snake_face_xy(z, X, Y):
    """Generate a serpentine scan over the top or bottom face at fixed z."""
    pts = []
    for ix, x in enumerate(X):
        col = [(x, y, z) for y in Y]
        if ix % 2 == 1:
            col.reverse()
        pts += col
    return np.array(pts)

def connect_on_plane(p_from, p_to, X, Y, Z, fixed_axis='z'):
    """Connect two points in a fixed-coordinate plane with an axis-aligned path using the grid spacing."""
    pts = []
    x, y, z = p_from
    tx, ty, tz = p_to
    dx = X[1]-X[0] if len(X)>1 else 0.0
    dy = Y[1]-Y[0] if len(Y)>1 else 0.0
    dz = Z[1]-Z[0] if len(Z)>1 else 0.0

    def step_axis(v, tv, d):
        sgn = 1 if tv > v else -1
        out = []
        while (tv - v) * sgn > 1e-12 and d > 0:
            nv = v + sgn * min(abs(tv - v), d)
            out.append(nv); v = nv
        return out

    if fixed_axis == 'z':
        for nv in step_axis(x, tx, dx): pts.append((nv, y, z)); x = nv
        for nv in step_axis(y, ty, dy): pts.append((x, nv, z)); y = nv
    elif fixed_axis == 'x':
        for nv in step_axis(y, ty, dy): pts.append((x, nv, z)); y = nv
        for nv in step_axis(z, tz, dz): pts.append((x, y, nv)); z = nv
    elif fixed_axis == 'y':
        for nv in step_axis(x, tx, dx): pts.append((nv, y, z)); x = nv
        for nv in step_axis(z, tz, dz): pts.append((x, y, nv)); z = nv
    return np.array(pts)

def snap_to_surface_grid(p, X, Y, Z, low, high):
    """Snap a surface point to the nearest grid node on the same face, edge, or corner."""
    x, y, z = p
    xs = X[np.argmin(np.abs(X - x))]
    ys = Y[np.argmin(np.abs(Y - y))]
    zs = Z[np.argmin(np.abs(Z - z))]

    if abs(x - low)  <= 1e-9: xs = low
    if abs(x - high) <= 1e-9: xs = high
    if abs(y - low)  <= 1e-9: ys = low
    if abs(y - high) <= 1e-9: ys = high
    if abs(z - low)  <= 1e-9: zs = low
    if abs(z - high) <= 1e-9: zs = high
    return np.array([xs, ys, zs])

def linear_steps(a, b, max_step):
    """Subdivide the line from a to b so that each segment is no longer than max_step."""
    a, b = np.array(a, float), np.array(b, float)
    d = np.linalg.norm(b - a)
    if d < 1e-12: return np.empty((0, 3))
    n = int(np.ceil(d / max_step))
    return np.array([a + (i / n) * (b - a) for i in range(1, n + 1)])




def build_closed_surface_path(low, high, x_step):
    X = make_grid(low, high, x_step)
    Y = make_grid(low, high, x_step)
    Z = make_grid(low, high, x_step)
    path_parts = []


    for k, z in enumerate(Z):
        path_parts.append(ring_path(z, X, Y, low, high))
        if k < len(Z) - 1:
            path_parts.append(np.array([(low, low, Z[k + 1])]))


    path_parts.append(snake_face_xy(high, X, Y))

    end_top = path_parts[-1][-1]
    if not np.allclose(end_top, (low, low, high)):
        path_parts.append(connect_on_plane(end_top, (low, low, high), X, Y, Z, fixed_axis='z'))


    for z in Z[-2::-1]:
        path_parts.append(np.array([(low, low, z)]))


    path_parts.append(snake_face_xy(low, X, Y))

    end_bot = path_parts[-1][-1]
    if not np.allclose(end_bot, (low, low, low)):
        path_parts.append(connect_on_plane(end_bot, (low, low, low), X, Y, Z, fixed_axis='z'))

    path = np.vstack(path_parts)
    return path, X, Y, Z



def build_full_walk(x0, low, high, x_step):

    surface_path, X, Y, Z = build_closed_surface_path(low, high, x_step)


    p_surf = project_to_surface(x0, low, high)
    walk_parts = [linear_steps(x0, p_surf, x_step)]


    p_snap = snap_to_surface_grid(p_surf, X, Y, Z, low, high)

    if abs(p_surf[2]-low)<1e-9 or abs(p_surf[2]-high)<1e-9:
        walk_parts.append(connect_on_plane(p_surf, (p_snap[0], p_snap[1], p_surf[2]), X, Y, Z, fixed_axis='z'))
    elif abs(p_surf[0]-low)<1e-9 or abs(p_surf[0]-high)<1e-9:
        walk_parts.append(connect_on_plane(p_surf, (p_surf[0], p_snap[1], p_snap[2]), X, Y, Z, fixed_axis='x'))
    else:
        walk_parts.append(connect_on_plane(p_surf, (p_snap[0], p_surf[1], p_snap[2]), X, Y, Z, fixed_axis='y'))



    idxs = np.where(np.isclose(surface_path, p_snap, atol=1e-9).all(axis=1))[0]
    start_idx = int(idxs[0]) if len(idxs) else int(np.argmin(np.linalg.norm(surface_path - p_snap, axis=1)))
    walk_parts += [surface_path[start_idx:], surface_path[:start_idx]]

    return np.vstack(walk_parts)



def draw_cube_wireframe(ax, low, high):
    """Draw a cube wireframe to show the path relative to the domain."""
    L, H = low, high
    edges = [
        [(L,L,L),(H,L,L)], [(L,L,L),(L,H,L)], [(L,L,L),(L,L,H)],
        [(H,H,H),(L,H,H)], [(H,H,H),(H,L,H)], [(H,H,H),(H,H,L)],
        [(L,H,H),(L,L,H)], [(L,H,H),(H,H,H)],
        [(H,L,H),(H,L,L)], [(H,L,H),(H,H,H)],
        [(H,H,L),(H,L,L)], [(H,H,L),(L,H,L)],
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2], linewidth=1, alpha=0.4)

def animate_walk(walk, low, high, pause_time=0.02, point_size=8):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = [], [], []
    for p in walk:
        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        ax.clear()
        ax.set_xlim(low-0.2, high+0.2)
        ax.set_ylim(low-0.2, high+0.2)
        ax.set_zlim(low-0.2, high+0.2)
        ax.set_box_aspect([1,1,1])
        draw_cube_wireframe(ax, low, high)
        ax.scatter(xs, ys, zs, s=point_size)
        ax.plot(xs, ys, zs)
        plt.pause(pause_time)
    plt.show()


if __name__ == "__main__":
    xd_low, xd_high = -1.0, 1.0
    x_step = 0.3
    x0 = [0.2, -1.6, 1.4]

    walk = build_full_walk(x0, xd_low, xd_high, x_step)

    # print("max step =", np.max(np.linalg.norm(np.diff(walk, axis=0), axis=1)))

    animate_walk(walk, xd_low, xd_high, pause_time=0.02, point_size=10)
