"""
PBD fluid dam break with marching cubes surface reconstruction.
Static boundary particles along walls/corners for proper density estimation.
Full GPU pipeline: density splat -> marching cubes -> transform -> rasterize (all Taichi).

Run: python3 dam_break_surface.py
Controls: Space=pause, R=reset, Esc=quit, P=toggle particles, W=toggle wireframe
"""

import taichi as ti
import numpy as np
import math

# --- Simulation parameters ---
NUM_FLUID = 70000
DT = 1.0 / 60.0
SUB_STEPS = 3
SOLVER_ITERS = 3
GRAVITY = ti.Vector([0.0, -9.81, 0.0])

DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0

PARTICLE_RADIUS = 0.009
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

MC_RES = 48
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

import platform as _platform
# Use CUDA on NVIDIA (fast compute) — GGUI handles Vulkan rendering via interop.
# Use Vulkan on Mac (Metal has no GGUI support; MoltenVK translates Vulkan).
_arch = ti.cuda if _platform.system() != "Darwin" else ti.vulkan
ti.init(arch=_arch)

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
def apply_corrections_and_clamp():
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
        p = pos_pred[i] + delta_pos[i]
        for d in ti.static(range(3)):
            if p[d] < DOMAIN_MIN + BOUNDARY_EPS:
                p[d] = DOMAIN_MIN + BOUNDARY_EPS
            if p[d] > DOMAIN_MAX - BOUNDARY_EPS:
                p[d] = DOMAIN_MAX - BOUNDARY_EPS
        pos_pred[i] = p


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
def clear_density():
    for I in ti.grouped(density_grid):
        density_grid[I] = 0.0


@ti.kernel
def splat_density():
    """Splat FLUID particle density onto the MC grid (not boundary)."""
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


# ============================================================
# Rendering fields — GGUI handles camera/rasterization,
# MC outputs world-space vertices/normals into mesh buffers
# ============================================================

RES = 700

MOUSE_RADIUS = 0.3
MOUSE_STRENGTH = 12.0

mouse_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

# Origin axis marker: 3 short lines along X (red), Y (green), Z (blue)
AXIS_LEN = 0.15
AXIS_ORIGIN = ti.Vector([0.0, 0.5, 0.0])  # Raised to mid-height for visibility
axis_x_verts = ti.Vector.field(3, dtype=ti.f32, shape=2)
axis_y_verts = ti.Vector.field(3, dtype=ti.f32, shape=2)
axis_z_verts = ti.Vector.field(3, dtype=ti.f32, shape=2)
axis_x_verts[0] = AXIS_ORIGIN; axis_x_verts[1] = AXIS_ORIGIN + [AXIS_LEN, 0.0, 0.0]
axis_y_verts[0] = AXIS_ORIGIN; axis_y_verts[1] = AXIS_ORIGIN + [0.0, AXIS_LEN, 0.0]
axis_z_verts[0] = AXIS_ORIGIN; axis_z_verts[1] = AXIS_ORIGIN + [0.0, 0.0, AXIS_LEN]

# World-space triangle data from marching cubes
MAX_MC_TRIS = 2000000
MAX_MC_VERTS = MAX_MC_TRIS * 3
mc_tri_count = ti.field(dtype=ti.i32, shape=())
mc_vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_MC_VERTS)
mc_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_MC_VERTS)
mc_indices = ti.field(dtype=ti.i32, shape=MAX_MC_VERTS)

# Render fields — sized to hold typical MC output. GGUI copies the entire
# field to a VBO each frame, so oversizing kills framerate.
RENDER_MAX_VERTS = 200000
render_vertices = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_MAX_VERTS)
render_normals = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_MAX_VERTS)
render_indices = ti.field(dtype=ti.i32, shape=RENDER_MAX_VERTS)


@ti.kernel
def copy_mc_to_render(n_verts: ti.i32):
    """GPU-side copy of active MC vertices into the smaller render fields."""
    for i in range(n_verts):
        render_vertices[i] = mc_vertices[i]
        render_normals[i] = mc_normals[i]
        render_indices[i] = mc_indices[i]


# ============================================================
# GPU Marching Cubes — full Taichi implementation
# ============================================================

# Standard marching cubes edge table: for each of the 256 cube configurations,
# a 12-bit mask indicating which edges are intersected by the isosurface.
_EDGE_TABLE = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]

# Standard marching cubes triangle table: for each of the 256 cube configs,
# up to 5 triangles (15 edge indices), terminated by -1.
_TRI_TABLE = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1],
    [9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1],
    [8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1],
    [3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1],
    [4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1],
    [9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1],
    [10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1],
    [5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1],
    [1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1],
    [2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1],
    [7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1],
    [11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1],
    [2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1],
    [6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1],
    [6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1],
    [6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1],
    [10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1],
    [1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1],
    [0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1],
    [10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1],
    [6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1],
    [10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1],
    [7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1],
    [7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1],
    [2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1],
    [10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1],
    [7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1],
    [6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1],
    [6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1],
    [0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1],
    [1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1],
    [10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1],
    [9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1],
    [6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1],
    [6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1],
    [9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1],
    [0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1],
    [5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1],
    [11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1],
    [9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1],
    [9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1],
    [5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1],
    [8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1],
    [9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1],
    [4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1],
    [0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1],
    [9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1],
    [1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
]

# Upload lookup tables to Taichi fields
mc_edge_table = ti.field(dtype=ti.i32, shape=256)
mc_edge_table.from_numpy(np.array(_EDGE_TABLE, dtype=np.int32))

mc_tri_table = ti.field(dtype=ti.i32, shape=(256, 16))
mc_tri_table.from_numpy(np.array(_TRI_TABLE, dtype=np.int32))



@ti.func
def mc_interp_vertex(iso: ti.f32, p1: ti.template(), p2: ti.template(),
                     v1: ti.f32, v2: ti.f32) -> ti.template():
    """Interpolate vertex position along an edge based on iso value."""
    result = p1
    if ti.abs(v1 - v2) > 1e-10:
        t = (iso - v1) / (v2 - v1)
        result = p1 + t * (p2 - p1)
    return result


@ti.func
def mc_grid_pos(ix: ti.i32, iy: ti.i32, iz: ti.i32) -> ti.template():
    """World position of a grid vertex."""
    return ti.Vector([DOMAIN_MIN + ix * MC_CELL,
                      DOMAIN_MIN + iy * MC_CELL,
                      DOMAIN_MIN + iz * MC_CELL])


@ti.func
def mc_grid_normal(ix: ti.i32, iy: ti.i32, iz: ti.i32) -> ti.template():
    """Compute normal at grid vertex using central differences on density_grid."""
    # Gradient of density field = normal pointing away from surface
    nx = 0.0
    ny = 0.0
    nz = 0.0
    if ix > 0 and ix < MC_RES:
        nx = density_grid[ix + 1, iy, iz] - density_grid[ix - 1, iy, iz]
    elif ix == 0:
        nx = density_grid[ix + 1, iy, iz] - density_grid[ix, iy, iz]
    else:
        nx = density_grid[ix, iy, iz] - density_grid[ix - 1, iy, iz]

    if iy > 0 and iy < MC_RES:
        ny = density_grid[ix, iy + 1, iz] - density_grid[ix, iy - 1, iz]
    elif iy == 0:
        ny = density_grid[ix, iy + 1, iz] - density_grid[ix, iy, iz]
    else:
        ny = density_grid[ix, iy, iz] - density_grid[ix, iy - 1, iz]

    if iz > 0 and iz < MC_RES:
        nz = density_grid[ix, iy, iz + 1] - density_grid[ix, iy, iz - 1]
    elif iz == 0:
        nz = density_grid[ix, iy, iz + 1] - density_grid[ix, iy, iz]
    else:
        nz = density_grid[ix, iy, iz] - density_grid[ix, iy, iz - 1]

    n = ti.Vector([nx, ny, nz])
    length = n.norm()
    if length > 1e-8:
        n = -n / length  # Flip to point outward (from high to low density)
    return n


@ti.func
def mc_interp_normal(iso: ti.f32, n1: ti.template(), n2: ti.template(),
                     v1: ti.f32, v2: ti.f32) -> ti.template():
    """Interpolate normal along an edge."""
    result = n1
    if ti.abs(v1 - v2) > 1e-10:
        t = (iso - v1) / (v2 - v1)
        result = n1 + t * (n2 - n1)
    length = result.norm()
    if length > 1e-8:
        result = result / length
    return result


@ti.kernel
def reset_mc_counter():
    mc_tri_count[None] = 0


@ti.kernel
def gpu_marching_cubes(iso: ti.f32):
    """Full marching cubes on GPU. Each cell processed in parallel."""
    ti.loop_config(block_dim=64)
    for idx in range(MC_RES * MC_RES * MC_RES):
        ix = idx // (MC_RES * MC_RES)
        iy = (idx // MC_RES) % MC_RES
        iz = idx % MC_RES

        # 8 corner values
        v0 = density_grid[ix, iy, iz]
        v1 = density_grid[ix + 1, iy, iz]
        v2 = density_grid[ix + 1, iy + 1, iz]
        v3 = density_grid[ix, iy + 1, iz]
        v4 = density_grid[ix, iy, iz + 1]
        v5 = density_grid[ix + 1, iy, iz + 1]
        v6 = density_grid[ix + 1, iy + 1, iz + 1]
        v7 = density_grid[ix, iy + 1, iz + 1]

        # Cube index from corner signs
        cube_index = 0
        if v0 < iso: cube_index |= 1
        if v1 < iso: cube_index |= 2
        if v2 < iso: cube_index |= 4
        if v3 < iso: cube_index |= 8
        if v4 < iso: cube_index |= 16
        if v5 < iso: cube_index |= 32
        if v6 < iso: cube_index |= 64
        if v7 < iso: cube_index |= 128

        edge_bits = mc_edge_table[cube_index]

        if edge_bits != 0:
            # 8 corner positions
            p0 = mc_grid_pos(ix, iy, iz)
            p1 = mc_grid_pos(ix + 1, iy, iz)
            p2 = mc_grid_pos(ix + 1, iy + 1, iz)
            p3 = mc_grid_pos(ix, iy + 1, iz)
            p4 = mc_grid_pos(ix, iy, iz + 1)
            p5 = mc_grid_pos(ix + 1, iy, iz + 1)
            p6 = mc_grid_pos(ix + 1, iy + 1, iz + 1)
            p7 = mc_grid_pos(ix, iy + 1, iz + 1)

            # 8 corner normals
            n0 = mc_grid_normal(ix, iy, iz)
            n1 = mc_grid_normal(ix + 1, iy, iz)
            n2 = mc_grid_normal(ix + 1, iy + 1, iz)
            n3 = mc_grid_normal(ix, iy + 1, iz)
            n4 = mc_grid_normal(ix, iy, iz + 1)
            n5 = mc_grid_normal(ix + 1, iy, iz + 1)
            n6 = mc_grid_normal(ix + 1, iy + 1, iz + 1)
            n7 = mc_grid_normal(ix, iy + 1, iz + 1)

            # Interpolate vertices and normals on active edges
            # Use local arrays for the 12 edges
            vert0 = ti.Vector([0.0, 0.0, 0.0])
            vert1 = ti.Vector([0.0, 0.0, 0.0])
            vert2 = ti.Vector([0.0, 0.0, 0.0])
            vert3 = ti.Vector([0.0, 0.0, 0.0])
            vert4 = ti.Vector([0.0, 0.0, 0.0])
            vert5 = ti.Vector([0.0, 0.0, 0.0])
            vert6 = ti.Vector([0.0, 0.0, 0.0])
            vert7 = ti.Vector([0.0, 0.0, 0.0])
            vert8 = ti.Vector([0.0, 0.0, 0.0])
            vert9 = ti.Vector([0.0, 0.0, 0.0])
            vert10 = ti.Vector([0.0, 0.0, 0.0])
            vert11 = ti.Vector([0.0, 0.0, 0.0])

            norm0 = ti.Vector([0.0, 0.0, 0.0])
            norm1 = ti.Vector([0.0, 0.0, 0.0])
            norm2 = ti.Vector([0.0, 0.0, 0.0])
            norm3 = ti.Vector([0.0, 0.0, 0.0])
            norm4 = ti.Vector([0.0, 0.0, 0.0])
            norm5 = ti.Vector([0.0, 0.0, 0.0])
            norm6 = ti.Vector([0.0, 0.0, 0.0])
            norm7 = ti.Vector([0.0, 0.0, 0.0])
            norm8 = ti.Vector([0.0, 0.0, 0.0])
            norm9 = ti.Vector([0.0, 0.0, 0.0])
            norm10 = ti.Vector([0.0, 0.0, 0.0])
            norm11 = ti.Vector([0.0, 0.0, 0.0])

            if edge_bits & 1:
                vert0 = mc_interp_vertex(iso, p0, p1, v0, v1)
                norm0 = mc_interp_normal(iso, n0, n1, v0, v1)
            if edge_bits & 2:
                vert1 = mc_interp_vertex(iso, p1, p2, v1, v2)
                norm1 = mc_interp_normal(iso, n1, n2, v1, v2)
            if edge_bits & 4:
                vert2 = mc_interp_vertex(iso, p2, p3, v2, v3)
                norm2 = mc_interp_normal(iso, n2, n3, v2, v3)
            if edge_bits & 8:
                vert3 = mc_interp_vertex(iso, p3, p0, v3, v0)
                norm3 = mc_interp_normal(iso, n3, n0, v3, v0)
            if edge_bits & 16:
                vert4 = mc_interp_vertex(iso, p4, p5, v4, v5)
                norm4 = mc_interp_normal(iso, n4, n5, v4, v5)
            if edge_bits & 32:
                vert5 = mc_interp_vertex(iso, p5, p6, v5, v6)
                norm5 = mc_interp_normal(iso, n5, n6, v5, v6)
            if edge_bits & 64:
                vert6 = mc_interp_vertex(iso, p6, p7, v6, v7)
                norm6 = mc_interp_normal(iso, n6, n7, v6, v7)
            if edge_bits & 128:
                vert7 = mc_interp_vertex(iso, p7, p4, v7, v4)
                norm7 = mc_interp_normal(iso, n7, n4, v7, v4)
            if edge_bits & 256:
                vert8 = mc_interp_vertex(iso, p0, p4, v0, v4)
                norm8 = mc_interp_normal(iso, n0, n4, v0, v4)
            if edge_bits & 512:
                vert9 = mc_interp_vertex(iso, p1, p5, v1, v5)
                norm9 = mc_interp_normal(iso, n1, n5, v1, v5)
            if edge_bits & 1024:
                vert10 = mc_interp_vertex(iso, p2, p6, v2, v6)
                norm10 = mc_interp_normal(iso, n2, n6, v2, v6)
            if edge_bits & 2048:
                vert11 = mc_interp_vertex(iso, p3, p7, v3, v7)
                norm11 = mc_interp_normal(iso, n3, n7, v3, v7)

            # Helper: map edge index to interpolated vertex/normal
            # We can't use arrays in Taichi easily, so use a function-like approach
            # Emit triangles from tri_table
            k = 0
            while k < 15:
                e0 = mc_tri_table[cube_index, k]
                if e0 == -1:
                    break
                e1 = mc_tri_table[cube_index, k + 1]
                e2 = mc_tri_table[cube_index, k + 2]

                # Map edge indices to vertices/normals
                tv0 = ti.Vector([0.0, 0.0, 0.0])
                tv1 = ti.Vector([0.0, 0.0, 0.0])
                tv2 = ti.Vector([0.0, 0.0, 0.0])
                tn0 = ti.Vector([0.0, 0.0, 0.0])
                tn1 = ti.Vector([0.0, 0.0, 0.0])
                tn2 = ti.Vector([0.0, 0.0, 0.0])

                if e0 == 0: tv0 = vert0; tn0 = norm0
                elif e0 == 1: tv0 = vert1; tn0 = norm1
                elif e0 == 2: tv0 = vert2; tn0 = norm2
                elif e0 == 3: tv0 = vert3; tn0 = norm3
                elif e0 == 4: tv0 = vert4; tn0 = norm4
                elif e0 == 5: tv0 = vert5; tn0 = norm5
                elif e0 == 6: tv0 = vert6; tn0 = norm6
                elif e0 == 7: tv0 = vert7; tn0 = norm7
                elif e0 == 8: tv0 = vert8; tn0 = norm8
                elif e0 == 9: tv0 = vert9; tn0 = norm9
                elif e0 == 10: tv0 = vert10; tn0 = norm10
                elif e0 == 11: tv0 = vert11; tn0 = norm11

                if e1 == 0: tv1 = vert0; tn1 = norm0
                elif e1 == 1: tv1 = vert1; tn1 = norm1
                elif e1 == 2: tv1 = vert2; tn1 = norm2
                elif e1 == 3: tv1 = vert3; tn1 = norm3
                elif e1 == 4: tv1 = vert4; tn1 = norm4
                elif e1 == 5: tv1 = vert5; tn1 = norm5
                elif e1 == 6: tv1 = vert6; tn1 = norm6
                elif e1 == 7: tv1 = vert7; tn1 = norm7
                elif e1 == 8: tv1 = vert8; tn1 = norm8
                elif e1 == 9: tv1 = vert9; tn1 = norm9
                elif e1 == 10: tv1 = vert10; tn1 = norm10
                elif e1 == 11: tv1 = vert11; tn1 = norm11

                if e2 == 0: tv2 = vert0; tn2 = norm0
                elif e2 == 1: tv2 = vert1; tn2 = norm1
                elif e2 == 2: tv2 = vert2; tn2 = norm2
                elif e2 == 3: tv2 = vert3; tn2 = norm3
                elif e2 == 4: tv2 = vert4; tn2 = norm4
                elif e2 == 5: tv2 = vert5; tn2 = norm5
                elif e2 == 6: tv2 = vert6; tn2 = norm6
                elif e2 == 7: tv2 = vert7; tn2 = norm7
                elif e2 == 8: tv2 = vert8; tn2 = norm8
                elif e2 == 9: tv2 = vert9; tn2 = norm9
                elif e2 == 10: tv2 = vert10; tn2 = norm10
                elif e2 == 11: tv2 = vert11; tn2 = norm11

                tri_idx = ti.atomic_add(mc_tri_count[None], 1)
                if tri_idx < MAX_MC_TRIS:
                    vi = tri_idx * 3
                    mc_vertices[vi] = tv0
                    mc_vertices[vi + 1] = tv1
                    mc_vertices[vi + 2] = tv2
                    mc_normals[vi] = tn0
                    mc_normals[vi + 1] = tn1
                    mc_normals[vi + 2] = tn2
                    mc_indices[vi] = vi
                    mc_indices[vi + 1] = vi + 1
                    mc_indices[vi + 2] = vi + 2

                k += 3


@ti.kernel
def apply_mouse_force():
    mp = mouse_pos[None]
    for i in range(NUM_FLUID):
        r = pos[i] - mp
        dist = r.norm()
        if dist < MOUSE_RADIUS and dist > EPSILON:
            strength = MOUSE_STRENGTH * (1.0 - dist / MOUSE_RADIUS)
            vel[i] += strength * r / dist


def initialize():
    spacing = PARTICLE_DIAMETER
    positions = []
    x = 0.05
    while x < 0.75 and len(positions) < NUM_FLUID:
        y = 0.05
        while y < 0.95 and len(positions) < NUM_FLUID:
            z = 0.05
            while z < 0.75 and len(positions) < NUM_FLUID:
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

    all_pos = np.vstack([fluid_pos, BOUNDARY_POS])
    pos.from_numpy(all_pos)
    vel.from_numpy(np.zeros((NUM_FLUID, 3), dtype=np.float32))
    print(f"Initialized {n} fluid + {NUM_BOUNDARY} boundary particles")


def main():
    initialize()

    window = ti.ui.Window("PBD Dam Break + GPU Surface", (RES, RES))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.2, 1.0, 1.2)
    camera.lookat(0.5, 0.35, 0.5)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(45)

    paused = False
    iso_threshold = REST_DENSITY * 0.4

    # Camera state — managed manually so orbit/pan stay consistent
    cam_lookat = [0.5, 0.35, 0.5]
    cam_dist = 1.5
    cam_theta = math.atan2(1.2 - 0.5, 1.2 - 0.5)  # horizontal angle
    cam_phi = math.atan2(1.0 - 0.35, math.sqrt((1.2-0.5)**2 + (1.2-0.5)**2))  # vertical angle

    print(f"Iso threshold = {iso_threshold:.2f}")
    print("Controls: Space=pause, R=reset, Arrows=orbit, Shift+Arrows=pan, W/S=zoom, LMB=push fluid, Esc=quit")

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ' ':
                paused = not paused
            elif e.key == 'r':
                initialize()

        # Camera controls — all state in cam_lookat/cam_theta/cam_phi/cam_dist
        orbit_speed = 0.02
        pan_speed = 0.015
        panning = window.is_pressed(ti.ui.SHIFT)

        if panning:
            # Shift+Arrows: pan lookat point relative to camera orientation
            # Right vector (perpendicular to view direction, in XZ plane)
            rx = -math.sin(cam_theta)
            rz = math.cos(cam_theta)
            # Forward vector (toward camera, projected onto XZ plane)
            fwd_x = -math.cos(cam_theta)
            fwd_z = -math.sin(cam_theta)
            if window.is_pressed(ti.ui.LEFT):
                cam_lookat[0] += rx * pan_speed
                cam_lookat[2] += rz * pan_speed
            if window.is_pressed(ti.ui.RIGHT):
                cam_lookat[0] -= rx * pan_speed
                cam_lookat[2] -= rz * pan_speed
            if window.is_pressed(ti.ui.UP):
                cam_lookat[0] += fwd_x * pan_speed
                cam_lookat[2] += fwd_z * pan_speed
            if window.is_pressed(ti.ui.DOWN):
                cam_lookat[0] -= fwd_x * pan_speed
                cam_lookat[2] -= fwd_z * pan_speed
        else:
            # Arrows: orbit
            if window.is_pressed(ti.ui.LEFT):
                cam_theta += orbit_speed
            if window.is_pressed(ti.ui.RIGHT):
                cam_theta -= orbit_speed
            if window.is_pressed(ti.ui.UP):
                cam_phi = min(cam_phi + orbit_speed, math.pi/2 - 0.05)
            if window.is_pressed(ti.ui.DOWN):
                cam_phi = max(cam_phi - orbit_speed, -math.pi/2 + 0.05)

        # WASD: zoom in/out and horizontal orbit
        if window.is_pressed('w'):
            cam_dist = max(0.3, cam_dist - 0.02)
        if window.is_pressed('s'):
            cam_dist = min(5.0, cam_dist + 0.02)

        # Apply camera state
        cx = cam_lookat[0] + cam_dist * math.cos(cam_phi) * math.cos(cam_theta)
        cy = cam_lookat[1] + cam_dist * math.sin(cam_phi)
        cz = cam_lookat[2] + cam_dist * math.cos(cam_phi) * math.sin(cam_theta)
        camera.position(cx, cy, cz)
        camera.lookat(cam_lookat[0], cam_lookat[1], cam_lookat[2])

        if not paused:
            for _ in range(SUB_STEPS):
                predict_positions()
                build_grid()
                find_neighbors()
                for _ in range(SOLVER_ITERS):
                    compute_lambdas()
                    apply_corrections_and_clamp()
                update_velocity()

        # Mouse interaction: left-click pushes fluid away
        # Applied after sim so vel isn't overwritten by PBD solver
        if window.is_pressed(ti.ui.LMB):
            cursor = window.get_cursor_pos()
            cp = camera.curr_position
            cl = camera.curr_lookat
            cu = camera.curr_up
            fx, fy, fz = cl[0]-cp[0], cl[1]-cp[1], cl[2]-cp[2]
            fl = math.sqrt(fx*fx + fy*fy + fz*fz)
            fx, fy, fz = fx/fl, fy/fl, fz/fl
            rx = fy*cu[2] - fz*cu[1]
            ry = fz*cu[0] - fx*cu[2]
            rz = fx*cu[1] - fy*cu[0]
            rl = math.sqrt(rx*rx + ry*ry + rz*rz)
            if rl > 1e-8:
                rx, ry, rz = rx/rl, ry/rl, rz/rl
                ux = ry*fz - rz*fy
                uy = rz*fx - rx*fz
                uz = rx*fy - ry*fx
                half_t = math.tan(45.0 * math.pi / 360.0)
                cx = (cursor[0] - 0.5) * 2.0 * half_t
                cy = (cursor[1] - 0.5) * 2.0 * half_t
                dx = fx + cx*rx + cy*ux
                dy = fy + cx*ry + cy*uy
                dz = fz + cx*rz + cy*uz
                dl = math.sqrt(dx*dx + dy*dy + dz*dz)
                dx, dy, dz = dx/dl, dy/dl, dz/dl
                if abs(dy) > 1e-6:
                    t = (0.35 - cp[1]) / dy
                    if t > 0:
                        hx = max(DOMAIN_MIN, min(DOMAIN_MAX, cp[0] + t*dx))
                        hz = max(DOMAIN_MIN, min(DOMAIN_MAX, cp[2] + t*dz))
                        mouse_pos[None] = [hx, 0.35, hz]
                        apply_mouse_force()

        # GPU pipeline: clear -> splat density -> marching cubes
        clear_density()
        splat_density()
        reset_mc_counter()
        gpu_marching_cubes(iso_threshold)

        # Render with GGUI
        # Set up scene first to give GPU time to finish MC before we read the count
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(2.0, 2.0, 2.0), color=(0.9, 0.9, 0.9))
        scene.point_light(pos=(-1.0, 1.5, 0.0), color=(0.3, 0.3, 0.4))

        n_tris = mc_tri_count[None]
        n_verts = min(min(n_tris, MAX_MC_TRIS) * 3, RENDER_MAX_VERTS)

        if n_verts > 0:
            copy_mc_to_render(n_verts)
            scene.mesh(render_vertices,
                       indices=render_indices,
                       normals=render_normals,
                       vertex_count=n_verts,
                       index_count=n_verts,
                       color=(0.15, 0.35, 0.65),
                       two_sided=True)

        # Origin axis marker: X=red, Y=green, Z=blue
        scene.lines(axis_x_verts, width=4.0, color=(1.0, 0.0, 0.0))
        scene.lines(axis_y_verts, width=4.0, color=(0.0, 1.0, 0.0))
        scene.lines(axis_z_verts, width=4.0, color=(0.0, 0.0, 1.0))

        canvas.set_background_color((0.8, 0.8, 0.8))
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
