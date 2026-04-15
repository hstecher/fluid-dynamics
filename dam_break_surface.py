"""
PBD fluid dam break with marching cubes surface reconstruction.
Static boundary particles along walls/corners for proper density estimation.

Run: python3 dam_break_surface.py
Controls: Space=pause, R=reset, Esc=quit, P=toggle particles, W=toggle wireframe
"""

import taichi as ti
import numpy as np
import math
from skimage.measure import marching_cubes

# --- Simulation parameters ---
NUM_FLUID = 4000
DT = 1.0 / 60.0
SUB_STEPS = 3
SOLVER_ITERS = 3
GRAVITY = ti.Vector([0.0, -9.81, 0.0])

DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0

PARTICLE_RADIUS = 0.015
PARTICLE_DIAMETER = 2.0 * PARTICLE_RADIUS
H = 4.0 * PARTICLE_RADIUS
H_SQ = H * H
MASS = 1.0
EPSILON = 1e-6
VISCOSITY = 0.01
BOUNDARY_EPS = PARTICLE_RADIUS

POLY6_COEFF = 315.0 / (64.0 * math.pi * H ** 9)
SPIKY_COEFF = -45.0 / (math.pi * H ** 6)

GRID_SIZE = H
GRID_DIM = int(np.ceil((DOMAIN_MAX - DOMAIN_MIN) / GRID_SIZE)) + 1
MAX_NEIGHBORS = 96
MAX_PER_CELL = 96

MC_RES = 50
MC_CELL = (DOMAIN_MAX - DOMAIN_MIN) / MC_RES
MC_SPLAT_RADIUS = H


def compute_rest_density():
    spacing = PARTICLE_DIAMETER
    rho = MASS * POLY6_COEFF * H_SQ ** 3
    shells = int(np.ceil(H / spacing)) + 1
    for ix in range(-shells, shells + 1):
        for iy in range(-shells, shells + 1):
            for iz in range(-shells, shells + 1):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                r_sq = (ix * spacing) ** 2 + (iy * spacing) ** 2 + (iz * spacing) ** 2
                if r_sq < H_SQ:
                    rho += MASS * POLY6_COEFF * (H_SQ - r_sq) ** 3
    return rho


REST_DENSITY = compute_rest_density()
CFM = 1e-3


def generate_boundary_particles():
    """Generate a single layer of static particles on all 6 faces of the domain."""
    spacing = PARTICLE_DIAMETER
    lo = DOMAIN_MIN
    hi = DOMAIN_MAX
    positions = set()
    n_per_edge = int((hi - lo) / spacing) + 1
    for i in range(n_per_edge):
        for j in range(n_per_edge):
            x = lo + i * spacing
            y = lo + j * spacing
            x = min(x, hi)
            y = min(y, hi)
            # Bottom and top (Y faces)
            positions.add((round(x, 5), lo, round(y, 5)))
            positions.add((round(x, 5), hi, round(y, 5)))
            # Front and back (Z faces)
            positions.add((round(x, 5), round(y, 5), lo))
            positions.add((round(x, 5), round(y, 5), hi))
            # Left and right (X faces)
            positions.add((lo, round(x, 5), round(y, 5)))
            positions.add((hi, round(x, 5), round(y, 5)))
    return np.array(list(positions), dtype=np.float32)


BOUNDARY_POS = generate_boundary_particles()
NUM_BOUNDARY = len(BOUNDARY_POS)
NUM_TOTAL = NUM_FLUID + NUM_BOUNDARY

print(f"H = {H:.4f}, REST_DENSITY = {REST_DENSITY:.2f}")
print(f"Fluid: {NUM_FLUID}, Boundary: {NUM_BOUNDARY}, Total: {NUM_TOTAL}")

ti.init(arch=ti.gpu)

# All particles in one array: [0..NUM_FLUID) are fluid, [NUM_FLUID..NUM_TOTAL) are boundary
pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_TOTAL)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_FLUID)
pos_pred = ti.Vector.field(3, dtype=ti.f32, shape=NUM_TOTAL)
delta_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_FLUID)
lambdas = ti.field(dtype=ti.f32, shape=NUM_FLUID)

neighbor_count = ti.field(dtype=ti.i32, shape=NUM_FLUID)
neighbors = ti.field(dtype=ti.i32, shape=(NUM_FLUID, MAX_NEIGHBORS))

grid_count = ti.field(dtype=ti.i32, shape=(GRID_DIM, GRID_DIM, GRID_DIM))
grid_particles = ti.field(dtype=ti.i32, shape=(GRID_DIM, GRID_DIM, GRID_DIM, MAX_PER_CELL))

density_grid = ti.field(dtype=ti.f32, shape=(MC_RES + 1, MC_RES + 1, MC_RES + 1))


@ti.func
def poly6_w(r_sq: ti.f32) -> ti.f32:
    result = 0.0
    if r_sq < H_SQ:
        x = H_SQ - r_sq
        result = POLY6_COEFF * x * x * x
    return result


@ti.func
def spiky_gradient(r: ti.template(), r_len: ti.f32) -> ti.template():
    result = ti.Vector([0.0, 0.0, 0.0])
    if EPSILON < r_len < H:
        x = H - r_len
        result = SPIKY_COEFF * x * x / r_len * r
    return result


@ti.func
def get_cell(p: ti.template()) -> ti.Vector:
    cx = int(ti.floor((p.x - DOMAIN_MIN) / GRID_SIZE))
    cy = int(ti.floor((p.y - DOMAIN_MIN) / GRID_SIZE))
    cz = int(ti.floor((p.z - DOMAIN_MIN) / GRID_SIZE))
    return ti.Vector([ti.math.clamp(cx, 0, GRID_DIM - 1),
                      ti.math.clamp(cy, 0, GRID_DIM - 1),
                      ti.math.clamp(cz, 0, GRID_DIM - 1)])


@ti.kernel
def build_grid():
    """Insert ALL particles (fluid + boundary) into the spatial grid."""
    for I in ti.grouped(grid_count):
        grid_count[I] = 0
    for i in range(NUM_TOTAL):
        cell = get_cell(pos_pred[i])
        idx = ti.atomic_add(grid_count[cell.x, cell.y, cell.z], 1)
        if idx < MAX_PER_CELL:
            grid_particles[cell.x, cell.y, cell.z, idx] = i


@ti.kernel
def find_neighbors():
    """Find neighbors for fluid particles only, but from all particles."""
    for i in range(NUM_FLUID):
        count = 0
        cell = get_cell(pos_pred[i])
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    nx = cell.x + dx
                    ny = cell.y + dy
                    nz = cell.z + dz
                    if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM and 0 <= nz < GRID_DIM:
                        for k in range(ti.min(grid_count[nx, ny, nz], MAX_PER_CELL)):
                            j = grid_particles[nx, ny, nz, k]
                            if j != i:
                                r = pos_pred[i] - pos_pred[j]
                                if r.dot(r) < H_SQ and count < MAX_NEIGHBORS:
                                    neighbors[i, count] = j
                                    count += 1
        neighbor_count[i] = count


@ti.kernel
def predict_positions():
    """Predict fluid positions; copy boundary positions unchanged."""
    sub_dt = DT / SUB_STEPS
    for i in range(NUM_FLUID):
        vel[i] += sub_dt * GRAVITY
        pos_pred[i] = pos[i] + sub_dt * vel[i]
    # Boundary particles stay fixed
    for i in range(NUM_FLUID, NUM_TOTAL):
        pos_pred[i] = pos[i]


@ti.kernel
def compute_lambdas():
    for i in range(NUM_FLUID):
        rho_i = MASS * poly6_w(0.0)
        for k in range(neighbor_count[i]):
            j = neighbors[i, k]
            r = pos_pred[i] - pos_pred[j]
            rho_i += MASS * poly6_w(r.dot(r))
        C_i = rho_i / REST_DENSITY - 1.0
        grad_sum_sq = 0.0
        grad_ci = ti.Vector([0.0, 0.0, 0.0])
        for k in range(neighbor_count[i]):
            j = neighbors[i, k]
            r = pos_pred[i] - pos_pred[j]
            r_len = r.norm()
            grad_j = (MASS / REST_DENSITY) * spiky_gradient(r, r_len)
            grad_sum_sq += grad_j.dot(grad_j)
            grad_ci += grad_j
        grad_sum_sq += grad_ci.dot(grad_ci)
        lambdas[i] = -ti.max(C_i, 0.0) / (grad_sum_sq + CFM)


@ti.kernel
def apply_corrections():
    for i in range(NUM_FLUID):
        dp = ti.Vector([0.0, 0.0, 0.0])
        for k in range(neighbor_count[i]):
            j = neighbors[i, k]
            r = pos_pred[i] - pos_pred[j]
            r_len = r.norm()
            # For boundary neighbors, use lambda=0 (they don't move)
            lam_j = 0.0
            if j < NUM_FLUID:
                lam_j = lambdas[j]
            dp += (lambdas[i] + lam_j) * spiky_gradient(r, r_len)
        delta_pos[i] = (MASS / REST_DENSITY) * dp
    for i in range(NUM_FLUID):
        pos_pred[i] += delta_pos[i]


@ti.kernel
def enforce_boundary():
    for i in range(NUM_FLUID):
        for d in ti.static(range(3)):
            if pos_pred[i][d] < DOMAIN_MIN + BOUNDARY_EPS:
                pos_pred[i][d] = DOMAIN_MIN + BOUNDARY_EPS
            if pos_pred[i][d] > DOMAIN_MAX - BOUNDARY_EPS:
                pos_pred[i][d] = DOMAIN_MAX - BOUNDARY_EPS


@ti.kernel
def update_velocity():
    inv_sub_dt = float(SUB_STEPS) / DT
    for i in range(NUM_FLUID):
        new_vel = (pos_pred[i] - pos[i]) * inv_sub_dt
        for d in ti.static(range(3)):
            if pos[i][d] <= DOMAIN_MIN + BOUNDARY_EPS and new_vel[d] < 0.0:
                new_vel[d] = 0.0
            if pos[i][d] >= DOMAIN_MAX - BOUNDARY_EPS and new_vel[d] > 0.0:
                new_vel[d] = 0.0
        vel[i] = new_vel * (1.0 - VISCOSITY)
        pos[i] = pos_pred[i]


@ti.kernel
def splat_density():
    """Splat FLUID particle density onto the MC grid (not boundary)."""
    for I in ti.grouped(density_grid):
        density_grid[I] = 0.0
    for i in range(NUM_FLUID):
        p = pos[i]
        lo_x = int(ti.floor((p.x - MC_SPLAT_RADIUS - DOMAIN_MIN) / MC_CELL))
        lo_y = int(ti.floor((p.y - MC_SPLAT_RADIUS - DOMAIN_MIN) / MC_CELL))
        lo_z = int(ti.floor((p.z - MC_SPLAT_RADIUS - DOMAIN_MIN) / MC_CELL))
        hi_x = int(ti.ceil((p.x + MC_SPLAT_RADIUS - DOMAIN_MIN) / MC_CELL))
        hi_y = int(ti.ceil((p.y + MC_SPLAT_RADIUS - DOMAIN_MIN) / MC_CELL))
        hi_z = int(ti.ceil((p.z + MC_SPLAT_RADIUS - DOMAIN_MIN) / MC_CELL))
        for gx in range(ti.max(lo_x, 0), ti.min(hi_x + 1, MC_RES + 1)):
            for gy in range(ti.max(lo_y, 0), ti.min(hi_y + 1, MC_RES + 1)):
                for gz in range(ti.max(lo_z, 0), ti.min(hi_z + 1, MC_RES + 1)):
                    gp = ti.Vector([DOMAIN_MIN + gx * MC_CELL,
                                    DOMAIN_MIN + gy * MC_CELL,
                                    DOMAIN_MIN + gz * MC_CELL])
                    r = p - gp
                    r_sq = r.dot(r)
                    ti.atomic_add(density_grid[gx, gy, gz], poly6_w(r_sq))


def initialize():
    spacing = PARTICLE_DIAMETER
    positions = []
    x = 0.05
    while x < 0.45 and len(positions) < NUM_FLUID:
        y = 0.05
        while y < 0.85 and len(positions) < NUM_FLUID:
            z = 0.05
            while z < 0.45 and len(positions) < NUM_FLUID:
                positions.append([x, y, z])
                z += spacing
            y += spacing
        x += spacing
    n = min(len(positions), NUM_FLUID)
    fluid_pos = np.array(positions[:n], dtype=np.float32)
    if n < NUM_FLUID:
        extra = NUM_FLUID - n
        pad = np.random.uniform(0.05, 0.45, (extra, 3)).astype(np.float32)
        fluid_pos = np.vstack([fluid_pos, pad])

    # Combine fluid + boundary into single array
    all_pos = np.vstack([fluid_pos, BOUNDARY_POS])
    pos.from_numpy(all_pos)
    vel.from_numpy(np.zeros((NUM_FLUID, 3), dtype=np.float32))
    print(f"Initialized {n} fluid + {NUM_BOUNDARY} boundary particles")


LIGHT_DIR = np.array([0.4, 0.7, 0.5])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
VIEW_DIR = np.array([0.0, 0.0, 1.0])

RES = 700

HALF_VEC = LIGHT_DIR + VIEW_DIR
HALF_VEC = HALF_VEC / np.linalg.norm(HALF_VEC)

# Mouse interaction force
MOUSE_RADIUS = 0.2         # world-space radius of influence
MOUSE_STRENGTH = 3.0       # direct velocity impulse

mouse_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

# Framebuffer for smooth-shaded software rasterizer
framebuf = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))

# Triangle data
MAX_TRIS = 30000
tri_v0 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIS)
tri_v1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIS)
tri_v2 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIS)
tri_n0 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIS)
tri_n1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIS)
tri_n2 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIS)

ti_light_dir = ti.Vector([float(LIGHT_DIR[0]), float(LIGHT_DIR[1]), float(LIGHT_DIR[2])])
ti_half_vec = ti.Vector([float(HALF_VEC[0]), float(HALF_VEC[1]), float(HALF_VEC[2])])


@ti.func
def shade_normal(n: ti.template()) -> ti.template():
    nn = n.normalized()
    ndotl = ti.max(nn.dot(ti_light_dir), 0.0)
    ndoth = ti.max(nn.dot(ti_half_vec), 0.0)
    spec = ndoth ** 40
    r = ti.math.clamp(0.05 + 0.15 * ndotl + 0.6 * spec, 0.0, 1.0)
    g = ti.math.clamp(0.12 + 0.25 * ndotl + 0.6 * spec, 0.0, 1.0)
    b = ti.math.clamp(0.25 + 0.40 * ndotl + 0.7 * spec, 0.0, 1.0)
    return ti.Vector([r, g, b])


@ti.func
def edge_func(a: ti.template(), b: ti.template(), c: ti.template()) -> ti.f32:
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)


zbuf = ti.field(dtype=ti.i32, shape=(RES, RES))  # integer Z for atomic_max


@ti.kernel
def clear_framebuf():
    for i, j in framebuf:
        framebuf[i, j] = ti.Vector([0.05, 0.05, 0.08])
        zbuf[i, j] = -2147483647


@ti.kernel
def rasterize_batch(batch_start: ti.i32, batch_end: ti.i32):
    """Rasterize triangles with per-pixel smooth shading and atomic Z-buffer."""
    for t in range(batch_start, batch_end):
        v0 = tri_v0[t]
        v1 = tri_v1[t]
        v2 = tri_v2[t]
        n0 = tri_n0[t]
        n1 = tri_n1[t]
        n2 = tri_n2[t]

        min_x = ti.max(int(ti.floor(ti.min(v0.x, ti.min(v1.x, v2.x)))), 0)
        max_x = ti.min(int(ti.ceil(ti.max(v0.x, ti.max(v1.x, v2.x)))), RES - 1)
        min_y = ti.max(int(ti.floor(ti.min(v0.y, ti.min(v1.y, v2.y)))), 0)
        max_y = ti.min(int(ti.ceil(ti.max(v0.y, ti.max(v1.y, v2.y)))), RES - 1)

        area = edge_func(v0, v1, v2)
        if ti.abs(area) > 1.0:
            if area < 0.0:
                v0, v1 = v1, v0
                n0, n1 = n1, n0
                area = -area
            inv_area = 1.0 / area
            for px in range(min_x, max_x + 1):
                for py in range(min_y, max_y + 1):
                    p = ti.Vector([float(px) + 0.5, float(py) + 0.5, 0.0])
                    w0 = edge_func(v1, v2, p) * inv_area
                    w1 = edge_func(v2, v0, p) * inv_area
                    w2 = 1.0 - w0 - w1
                    if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                        z = w0 * v0.z + w1 * v1.z + w2 * v2.z
                        # Scale Z to int for atomic_max (avoids float race)
                        z_int = int(z * 1e6)
                        old = ti.atomic_max(zbuf[px, py], z_int)
                        if z_int >= old:
                            interp_n = w0 * n0 + w1 * n1 + w2 * n2
                            framebuf[px, py] = shade_normal(interp_n)


@ti.kernel
def apply_mouse_force():
    mp = mouse_pos[None]
    for i in range(NUM_FLUID):
        r = pos[i] - mp
        dist = r.norm()
        if dist < MOUSE_RADIUS and dist > EPSILON:
            strength = MOUSE_STRENGTH * (1.0 - dist / MOUSE_RADIUS)
            vel[i] += strength * r / dist


def transform_verts(verts, cam_angle, cam_tilt):
    cos_a = math.cos(cam_angle)
    sin_a = math.sin(cam_angle)
    cos_t = math.cos(cam_tilt)
    sin_t = math.sin(cam_tilt)
    vx = verts[:, 0] - 0.5
    vy = verts[:, 1] - 0.5
    vz = verts[:, 2] - 0.5
    x_rot = vx * cos_a + vz * sin_a
    z_rot = -vx * sin_a + vz * cos_a
    y_proj = vy * cos_t - z_rot * sin_t
    z_proj = vy * sin_t + z_rot * cos_t
    scale = 0.85
    sx = (x_rot * scale + 0.5) * RES
    sy = (y_proj * scale + 0.5) * RES
    return np.column_stack([sx, sy, z_proj])


def rotate_normals(normals, cam_angle, cam_tilt):
    cos_a = math.cos(cam_angle)
    sin_a = math.sin(cam_angle)
    cos_t = math.cos(cam_tilt)
    sin_t = math.sin(cam_tilt)
    nx = normals[:, 0] * cos_a + normals[:, 2] * sin_a
    nz = -normals[:, 0] * sin_a + normals[:, 2] * cos_a
    ny_r = normals[:, 1] * cos_t - nz * sin_t
    nz_r = normals[:, 1] * sin_t + nz * cos_t
    rot = np.column_stack([nx, ny_r, nz_r])
    lengths = np.linalg.norm(rot, axis=1, keepdims=True).clip(1e-8)
    rot /= lengths
    return rot.astype(np.float32)


def screen_to_world(sx, sy, cam_angle, cam_tilt):
    """Unproject normalized screen coords [0,1] to a world-space ray hit on y=0.3 plane."""
    cos_a = math.cos(cam_angle)
    sin_a = math.sin(cam_angle)
    cos_t = math.cos(cam_tilt)
    sin_t = math.sin(cam_tilt)
    # Invert the projection: screen -> rotated space
    scale = 0.85
    x_rot = (sx - 0.5) / scale
    y_proj = (sy - 0.5) / scale
    # Approximate: assume z_rot ~ 0 (center of domain)
    # y_proj = vy * cos_t - z_rot * sin_t, with z_rot ~ 0 -> vy ~ y_proj / cos_t
    vy = y_proj / cos_t if abs(cos_t) > 0.01 else 0.0
    # x_rot = vx * cos_a + vz * sin_a, assume vz ~ 0 -> vx ~ x_rot / cos_a
    vx = x_rot / cos_a if abs(cos_a) > 0.01 else 0.0
    return np.array([vx + 0.5, vy + 0.5, 0.5])


def main():
    initialize()

    gui = ti.GUI("PBD Dam Break + Shaded Surface", res=(RES, RES),
                 background_color=0x0D0D14)

    paused = False
    frame = 0
    iso_threshold = REST_DENSITY * 0.4
    cam_angle = 0.75
    cam_tilt = 0.35
    rot_speed = 0.05

    print(f"Iso threshold = {iso_threshold:.2f}")
    print("Controls: Arrows=rotate, Space=pause, R=reset, LMB=push water, Esc=quit")

    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == ' ':
                paused = not paused
            elif e.key == 'r':
                initialize()

        # Arrow keys for camera (check held state)
        if gui.is_pressed(gui.LEFT):
            cam_angle -= rot_speed
        if gui.is_pressed(gui.RIGHT):
            cam_angle += rot_speed
        if gui.is_pressed(gui.UP):
            cam_tilt = min(cam_tilt + rot_speed, 1.2)
        if gui.is_pressed(gui.DOWN):
            cam_tilt = max(cam_tilt - rot_speed, -0.2)

        # Mouse interaction — push water on LMB
        if gui.is_pressed(gui.LMB):
            mx, my = gui.get_cursor_pos()
            wp = screen_to_world(mx, my, cam_angle, cam_tilt)
            wp = np.clip(wp, 0.0, 1.0)
            mouse_pos[None] = ti.Vector([float(wp[0]), float(wp[1]), float(wp[2])])
            apply_mouse_force()

        if not paused:
            for _ in range(SUB_STEPS):
                predict_positions()
                build_grid()
                find_neighbors()
                for _ in range(SOLVER_ITERS):
                    compute_lambdas()
                    apply_corrections()
                    enforce_boundary()
                update_velocity()

        # Surface reconstruction and rendering
        clear_framebuf()
        splat_density()
        dg = density_grid.to_numpy()
        try:
            verts, faces, normals, _ = marching_cubes(dg, level=iso_threshold,
                                                       spacing=(MC_CELL, MC_CELL, MC_CELL))
            verts += DOMAIN_MIN

            if len(faces) > 0:
                rot_normals = rotate_normals(normals, cam_angle, cam_tilt)
                screen_verts = transform_verts(verts, cam_angle, cam_tilt)

                # Sort back-to-front
                face_z = (screen_verts[faces[:, 0], 2] +
                          screen_verts[faces[:, 1], 2] +
                          screen_verts[faces[:, 2], 2]) / 3.0
                order = np.argsort(face_z)
                faces_sorted = faces[order]

                n_tris = min(len(faces_sorted), MAX_TRIS)
                f = faces_sorted[:n_tris]
                tri_v0.from_numpy(screen_verts[f[:, 0]].astype(np.float32))
                tri_v1.from_numpy(screen_verts[f[:, 1]].astype(np.float32))
                tri_v2.from_numpy(screen_verts[f[:, 2]].astype(np.float32))
                tri_n0.from_numpy(rot_normals[f[:, 0]])
                tri_n1.from_numpy(rot_normals[f[:, 1]])
                tri_n2.from_numpy(rot_normals[f[:, 2]])

                rasterize_batch(0, n_tris)
        except (ValueError, IndexError):
            pass
        gui.set_image(framebuf)
        gui.show()
        frame += 1


if __name__ == "__main__":
    main()
