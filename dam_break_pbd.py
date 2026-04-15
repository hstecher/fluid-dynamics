"""
Basic 3D PBD fluid simulation — dam break in a box.
Particles start as a cube of water in one corner, fall under gravity,
and splash against the walls. Uses Position-Based Dynamics for
incompressibility constraint enforcement.

Run: python3 dam_break_pbd.py
Controls: Space=pause, R=reset, Esc=quit.
"""

import taichi as ti
import numpy as np
import math

# --- Simulation parameters ---
NUM_PARTICLES = 8000
DT = 1.0 / 60.0
SUB_STEPS = 3
SOLVER_ITERS = 4
GRAVITY = ti.Vector([0.0, -9.81, 0.0])

# Domain: unit cube [0,1]^3
DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0

# Particle spacing / kernel
PARTICLE_RADIUS = 0.012
PARTICLE_DIAMETER = 2.0 * PARTICLE_RADIUS
H = 4.0 * PARTICLE_RADIUS
H_SQ = H * H
MASS = 1.0
EPSILON = 1e-6
VISCOSITY = 0.01
BOUNDARY_EPS = PARTICLE_RADIUS

# Kernel normalization constants
POLY6_COEFF = 315.0 / (64.0 * math.pi * H ** 9)
SPIKY_COEFF = -45.0 / (math.pi * H ** 6)

# Neighbor search grid
GRID_SIZE = H
GRID_DIM = int(np.ceil((DOMAIN_MAX - DOMAIN_MIN) / GRID_SIZE)) + 1
MAX_NEIGHBORS = 64
MAX_PER_CELL = 64


def compute_rest_density():
    """Compute SPH density at a particle in a perfect cubic lattice with spacing = PARTICLE_DIAMETER."""
    spacing = PARTICLE_DIAMETER
    rho = MASS * POLY6_COEFF * H_SQ ** 3  # self term: W(0)
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


# Calibrate REST_DENSITY to match the initial packing
REST_DENSITY = compute_rest_density()
# CFM (constraint force mixing) — regularization for the constraint solve.
# Scale it relative to the gradient magnitudes which depend on REST_DENSITY.
CFM = 1e-4

print(f"Kernel radius H = {H:.4f}")
print(f"Poly6 W(0) = {POLY6_COEFF * H_SQ**3:.2f}")
print(f"Calibrated REST_DENSITY = {REST_DENSITY:.2f}")

# --- Initialize Taichi AFTER computing constants ---
ti.init(arch=ti.gpu)

# --- Taichi fields ---
pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
pos_pred = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
delta_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
lambdas = ti.field(dtype=ti.f32, shape=NUM_PARTICLES)

neighbor_count = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)
neighbors = ti.field(dtype=ti.i32, shape=(NUM_PARTICLES, MAX_NEIGHBORS))

grid_count = ti.field(dtype=ti.i32, shape=(GRID_DIM, GRID_DIM, GRID_DIM))
grid_particles = ti.field(dtype=ti.i32, shape=(GRID_DIM, GRID_DIM, GRID_DIM, MAX_PER_CELL))


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
    for I in ti.grouped(grid_count):
        grid_count[I] = 0
    for i in pos_pred:
        cell = get_cell(pos_pred[i])
        idx = ti.atomic_add(grid_count[cell.x, cell.y, cell.z], 1)
        if idx < MAX_PER_CELL:
            grid_particles[cell.x, cell.y, cell.z, idx] = i


@ti.kernel
def find_neighbors():
    for i in pos_pred:
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
    sub_dt = DT / SUB_STEPS
    for i in pos:
        vel[i] += sub_dt * GRAVITY
        pos_pred[i] = pos[i] + sub_dt * vel[i]


@ti.kernel
def compute_lambdas():
    for i in pos_pred:
        # SPH density
        rho_i = MASS * poly6_w(0.0)
        for k in range(neighbor_count[i]):
            j = neighbors[i, k]
            r = pos_pred[i] - pos_pred[j]
            rho_i += MASS * poly6_w(r.dot(r))

        # Density constraint: C_i = rho_i / rho_0 - 1
        C_i = rho_i / REST_DENSITY - 1.0

        # Sum of squared gradients
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

        # Only enforce when density exceeds rest (free surface)
        lambdas[i] = -ti.max(C_i, 0.0) / (grad_sum_sq + CFM)


@ti.kernel
def apply_corrections():
    for i in pos_pred:
        dp = ti.Vector([0.0, 0.0, 0.0])
        for k in range(neighbor_count[i]):
            j = neighbors[i, k]
            r = pos_pred[i] - pos_pred[j]
            r_len = r.norm()
            dp += (lambdas[i] + lambdas[j]) * spiky_gradient(r, r_len)
        delta_pos[i] = (MASS / REST_DENSITY) * dp

    for i in pos_pred:
        pos_pred[i] += delta_pos[i]


@ti.kernel
def enforce_boundary():
    for i in pos_pred:
        for d in ti.static(range(3)):
            if pos_pred[i][d] < DOMAIN_MIN + BOUNDARY_EPS:
                pos_pred[i][d] = DOMAIN_MIN + BOUNDARY_EPS
            if pos_pred[i][d] > DOMAIN_MAX - BOUNDARY_EPS:
                pos_pred[i][d] = DOMAIN_MAX - BOUNDARY_EPS


@ti.kernel
def update_velocity():
    inv_sub_dt = float(SUB_STEPS) / DT
    for i in pos:
        new_vel = (pos_pred[i] - pos[i]) * inv_sub_dt
        vel[i] = new_vel * (1.0 - VISCOSITY)
        pos[i] = pos_pred[i]


def initialize():
    spacing = PARTICLE_DIAMETER
    positions = []
    x = 0.05
    while x < 0.45 and len(positions) < NUM_PARTICLES:
        y = 0.05
        while y < 0.95 and len(positions) < NUM_PARTICLES:
            z = 0.05
            while z < 0.45 and len(positions) < NUM_PARTICLES:
                positions.append([x, y, z])
                z += spacing
            y += spacing
        x += spacing

    n = min(len(positions), NUM_PARTICLES)
    pos_np = np.array(positions[:n], dtype=np.float32)
    vel_np = np.zeros((n, 3), dtype=np.float32)

    if n < NUM_PARTICLES:
        extra = NUM_PARTICLES - n
        pad_pos = np.random.uniform(0.05, 0.45, (extra, 3)).astype(np.float32)
        pos_np = np.vstack([pos_np, pad_pos])
        vel_np = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)

    pos.from_numpy(pos_np)
    vel.from_numpy(vel_np)
    print(f"Initialized {n} particles")


def main():
    initialize()

    res = 800
    gui = ti.GUI("PBD Dam Break  |  Side (XY)  left  |  Top (XZ)  right",
                 res=(res * 2, res), background_color=0x0D0D14)

    paused = False
    frame = 0

    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == ' ':
                paused = not paused
            elif e.key == 'r':
                initialize()

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

        pos_np = pos.to_numpy()

        margin = 0.03
        side_x = pos_np[:, 0] * (0.5 - 2 * margin) + margin
        side_y = pos_np[:, 1] * (1.0 - 2 * margin) + margin
        top_x = pos_np[:, 0] * (0.5 - 2 * margin) + 0.5 + margin
        top_y = pos_np[:, 2] * (1.0 - 2 * margin) + margin

        # Domain boxes
        gui.line([margin, margin], [0.5 - margin, margin], color=0x4A4A4A, radius=1)
        gui.line([0.5 - margin, margin], [0.5 - margin, 1.0 - margin], color=0x4A4A4A, radius=1)
        gui.line([0.5 - margin, 1.0 - margin], [margin, 1.0 - margin], color=0x4A4A4A, radius=1)
        gui.line([margin, 1.0 - margin], [margin, margin], color=0x4A4A4A, radius=1)
        gui.line([0.5 + margin, margin], [1.0 - margin, margin], color=0x4A4A4A, radius=1)
        gui.line([1.0 - margin, margin], [1.0 - margin, 1.0 - margin], color=0x4A4A4A, radius=1)
        gui.line([1.0 - margin, 1.0 - margin], [0.5 + margin, 1.0 - margin], color=0x4A4A4A, radius=1)
        gui.line([0.5 + margin, 1.0 - margin], [0.5 + margin, margin], color=0x4A4A4A, radius=1)
        gui.line([0.5, 0.0], [0.5, 1.0], color=0x333333, radius=1)

        gui.circles(np.column_stack([side_x, side_y]), radius=2, color=0x3388EE)
        gui.circles(np.column_stack([top_x, top_y]), radius=2, color=0x3388EE)

        gui.text(f"Frame {frame}  |  Space=pause  R=reset  Esc=quit",
                 pos=(0.01, 0.97), color=0x888888)
        gui.text("Side (XY)", pos=(0.18, 0.01), color=0x666666)
        gui.text("Top (XZ)", pos=(0.70, 0.01), color=0x666666)

        gui.show()
        frame += 1


if __name__ == "__main__":
    main()
