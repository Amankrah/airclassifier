# NVIDIA Warp Developer Guide
## Reusable Components for Modeling & Processing

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Spatial Computing Primitives](#2-spatial-computing-primitives)
3. [Mathematical Types & Operations](#3-mathematical-types--operations)
4. [Geometry Built-in Functions](#4-geometry-built-in-functions)
5. [warp.sparse - Sparse Linear Algebra](#5-warpsparse---sparse-linear-algebra)
6. [warp.fem - Finite Element Methods](#6-warpfem---finite-element-methods)
7. [Array Utilities](#7-array-utilities)
8. [Random Number Generation](#8-random-number-generation)
9. [Tile Programming Model](#9-tile-programming-model)
10. [Atomic Operations](#10-atomic-operations)
11. [Framework Integration](#11-framework-integration)
12. [Automatic Differentiation](#12-automatic-differentiation)
13. [Best Practices](#13-best-practices)

---

## 1. Introduction

NVIDIA Warp is an open-source Python framework for writing high-performance, differentiable simulation and graphics code. It provides JIT compilation of Python functions to efficient CPU and GPU kernels, making it ideal for:

- Physics simulation
- Robotics
- Machine learning
- Geometry processing
- Computer graphics

### 1.1 Installation

```bash
# Basic installation
pip install warp-lang

# With examples and USD support
pip install warp-lang[examples]

# Nightly builds
pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
```

### 1.2 Basic Usage

```python
import warp as wp

# Initialize Warp
wp.init()

# Define a kernel
@wp.kernel
def my_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0

# Launch kernel
wp.launch(my_kernel, dim=n, inputs=[a], outputs=[b], device="cuda")
```

---

## 2. Spatial Computing Primitives

Warp provides efficient data structures for spatial queries and geometric operations, essential for collision detection, neighbor searches, and ray tracing.

### 2.1 wp.Mesh - Triangle Mesh Operations

The `wp.Mesh` class provides a built-in type for managing triangle mesh data with support for geometric queries including closest-point, ray-cast, and overlap checks. It uses a BVH (Bounding Volume Hierarchy) acceleration structure.

#### Constructor

```python
mesh = wp.Mesh(points, indices, velocities=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `wp.array(dtype=wp.vec3)` | Vertex positions |
| `indices` | `wp.array(dtype=int)` | Triangle indices (3 per face) |
| `velocities` | `wp.array(dtype=wp.vec3)` | Optional vertex velocities for motion blur |

#### Key Methods & Built-ins

| Function | Description |
|----------|-------------|
| `mesh.refit()` | Rebuild BVH after vertex updates |
| `wp.mesh_query_point()` | Find closest point on mesh from position |
| `wp.mesh_query_ray()` | Ray-mesh intersection query |
| `wp.mesh_query_aabb()` | AABB overlap query |
| `wp.mesh_eval_position()` | Evaluate position at (face, u, v) |
| `wp.mesh_eval_velocity()` | Evaluate velocity at (face, u, v) |
| `wp.mesh_eval_face_normal()` | Get face normal |

#### Example: Closest Point Query

```python
@wp.kernel
def project_to_mesh(
    positions: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    output_pos: wp.array(dtype=wp.vec3),
    output_face: wp.array(dtype=int)
):
    tid = wp.tid()
    x = positions[tid]
    
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    max_dist = 2.0
    
    if wp.mesh_query_point(mesh, x, max_dist, sign, face_index, face_u, face_v):
        p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        output_pos[tid] = p
        output_face[tid] = face_index
```

#### Example: Ray Casting

```python
@wp.kernel
def raycast(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    hits: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)
    
    if wp.mesh_query_ray(mesh, origins[tid], directions[tid], 1000.0,
                         t, u, v, sign, normal, face):
        hits[tid] = origins[tid] + directions[tid] * t
```

---

### 2.2 wp.Bvh - Bounding Volume Hierarchy

Generic BVH for arbitrary point cloud queries. Supports multiple construction algorithms and group-aware queries for multi-world simulations.

#### Constructor

```python
bvh = wp.Bvh(
    lowers,              # wp.array(dtype=wp.vec3) - AABB lower bounds
    uppers,              # wp.array(dtype=wp.vec3) - AABB upper bounds
    constructor="lbvh", # "lbvh", "sah", or "median"
    leaf_size=1,         # Primitives per leaf node
    groups=None          # Optional group assignments
)
```

| Parameter | Description |
|-----------|-------------|
| `lowers` | AABB lower bounds for each primitive |
| `uppers` | AABB upper bounds for each primitive |
| `constructor` | `"lbvh"` (fast), `"sah"` (quality), `"median"` (balanced) |
| `leaf_size` | 1 for intersections, 4-8 for closest-point queries |
| `groups` | Assign primitives to independent groups |

#### Key Methods

| Function | Description |
|----------|-------------|
| `bvh.refit()` | Update bounds without rebuilding structure |
| `bvh.rebuild()` | Full in-place rebuild (CUDA graph safe) |
| `wp.bvh_query_aabb()` | Query overlapping AABBs |
| `wp.bvh_query_ray()` | Ray intersection query |

#### Example: AABB Query

```python
@wp.kernel
def find_overlaps(
    bvh: wp.uint64,
    query_lower: wp.array(dtype=wp.vec3),
    query_upper: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=int)
):
    tid = wp.tid()
    
    query = wp.bvh_query_aabb(bvh, query_lower[tid], query_upper[tid])
    
    count = int(0)
    index = int(0)
    while wp.bvh_query_next(query, index):
        count += 1
    
    counts[tid] = count
```

---

### 2.3 wp.HashGrid - Spatial Hash Grid

Hash grids accelerate nearest-neighbor queries for particle-based simulations like SPH (Smoothed Particle Hydrodynamics) and DEM (Discrete Element Method).

#### Constructor & Usage

```python
# Create grid
grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device="cuda")

# Build from points
grid.build(points=positions, radius=search_radius)

# Reserve capacity (optional)
grid.reserve(num_points)
```

#### Example: Neighbor Search

```python
@wp.kernel
def sum_neighbors(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=wp.vec3),
    radius: float
):
    tid = wp.tid()
    p = points[tid]
    result = wp.vec3(0.0, 0.0, 0.0)
    
    # Create query around point
    query = wp.hash_grid_query(grid, p, radius)
    index = int(0)
    
    # Iterate over neighbors
    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]
        dist = wp.length(p - neighbor)
        if dist < radius and index != tid:
            result = result + neighbor
    
    output[tid] = result
```

#### Example: SPH Density Computation

```python
@wp.kernel
def compute_density(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    mass: float,
    radius: float
):
    tid = wp.tid()
    p = positions[tid]
    density = float(0.0)
    
    query = wp.hash_grid_query(grid, p, radius)
    index = int(0)
    
    while wp.hash_grid_query_next(query, index):
        neighbor = positions[index]
        dist = wp.length(p - neighbor)
        if dist < radius:
            # SPH kernel (simplified)
            q = dist / radius
            w = (1.0 - q) * (1.0 - q) * (1.0 - q)
            density += mass * w
    
    densities[tid] = density
```

---

### 2.4 wp.Volume - Sparse Volumes (NanoVDB)

Sparse volumes represent grid data over large domains efficiently. Ideal for signed distance fields (SDFs), velocity fields, and large-scale fluid simulations. Uses the NanoVDB standard.

#### Creating Volumes

```python
# Load from file
volume = wp.Volume.load_from_nvdb("volume.nvdb", device="cuda")

# Allocate new volume
volume = wp.Volume.allocate(
    min=[0, 0, 0],
    max=[128, 128, 128],
    voxel_size=0.1,
    bg_value=0.0,
    device="cuda"
)
```

#### Key Functions

| Function | Description |
|----------|-------------|
| `wp.volume_sample_world()` | World-space sampling with interpolation |
| `wp.volume_sample_local()` | Volume-space sampling |
| `wp.volume_lookup()` | Direct voxel lookup (no interpolation) |
| `wp.volume_store()` | Write to voxel |
| `wp.volume_transform()` | Voxel space → World space |
| `wp.volume_transform_inv()` | World space → Voxel space |
| `volume.save_to_nvdb()` | Export to .nvdb file |

#### Interpolation Modes

- `wp.Volume.CLOSEST` - Nearest neighbor (fastest)
- `wp.Volume.LINEAR` - Trilinear interpolation (smooth)

#### Example: SDF Sampling

```python
@wp.kernel
def sample_sdf(
    volume: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float)
):
    tid = wp.tid()
    p = positions[tid]
    
    # Sample with trilinear interpolation
    d = wp.volume_sample_world(volume, p, wp.Volume.LINEAR)
    distances[tid] = d
```

---

### 2.5 wp.MarchingCubes - Isosurface Extraction

Fully differentiable marching cubes implementation for extracting triangle meshes from volumetric data. Written entirely in Warp, runs on both CPU and GPU.

#### Usage

```python
# Create marching cubes object
mc = wp.MarchingCubes(
    nx=64, ny=64, nz=64,    # Grid dimensions
    max_verts=100000,        # Max output vertices
    max_tris=200000,         # Max output triangles
    device="cuda"
)

# Extract isosurface
mc.surface(field=sdf_field, threshold=0.0)

# Access results
vertices = mc.verts      # wp.array(dtype=wp.vec3)
triangles = mc.indices   # wp.array(dtype=int)
num_verts = mc.nverts    # Number of vertices
num_tris = mc.ntris      # Number of triangles
```

---

## 3. Mathematical Types & Operations

Warp provides built-in types similar to high-level shading languages, optimized for simulation and graphics workloads.

### 3.1 Vector Types

| Type | Description |
|------|-------------|
| `wp.vec2`, `wp.vec3`, `wp.vec4` | Float vectors (32-bit default) |
| `wp.vec2f`, `wp.vec3f`, `wp.vec4f` | Explicit float32 vectors |
| `wp.vec2d`, `wp.vec3d`, `wp.vec4d` | Float64 (double) vectors |
| `wp.vec2i`, `wp.vec3i`, `wp.vec4i` | Integer vectors |
| `wp.vec2h`, `wp.vec3h`, `wp.vec4h` | Float16 (half) vectors |

#### Creating Custom Vector Types

```python
# Custom vector type
vec5d = wp.types.vector(length=5, dtype=wp.float64)

# Use in arrays
arr = wp.zeros(100, dtype=vec5d)
```

#### Vector Operations

| Function | Description |
|----------|-------------|
| `wp.dot(a, b)` | Dot product |
| `wp.cross(a, b)` | Cross product (3D only) |
| `wp.length(v)` | Vector magnitude |
| `wp.length_sq(v)` | Squared magnitude (faster) |
| `wp.normalize(v)` | Unit vector |
| `wp.lerp(a, b, t)` | Linear interpolation |
| `wp.outer(a, b)` | Outer product → matrix |
| `wp.min(a, b)` | Element-wise minimum |
| `wp.max(a, b)` | Element-wise maximum |
| `wp.clamp(v, lo, hi)` | Clamp to range |
| `wp.abs(v)` | Element-wise absolute value |

#### Example

```python
@wp.kernel
def vector_ops(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float)
):
    tid = wp.tid()
    
    v1 = a[tid]
    v2 = b[tid]
    
    # Dot product
    d = wp.dot(v1, v2)
    
    # Cross product
    c = wp.cross(v1, v2)
    
    # Normalize
    n = wp.normalize(v1)
    
    # Distance
    dist = wp.length(v1 - v2)
    
    results[tid] = dist
```

---

### 3.2 Matrix Types

| Type | Description |
|------|-------------|
| `wp.mat22`, `wp.mat33`, `wp.mat44` | Square matrices (float32) |
| `wp.mat22f`, `wp.mat33f`, `wp.mat44f` | Explicit float32 |
| `wp.mat22d`, `wp.mat33d`, `wp.mat44d` | Float64 matrices |
| `wp.spatial_matrix` | 6×6 spatial algebra matrix |

#### Creating Custom Matrix Types

```python
# Custom matrix type
mat35 = wp.types.matrix(shape=(3, 5), dtype=wp.float32)
```

#### Matrix Operations

| Function | Description |
|----------|-------------|
| `wp.transpose(m)` | Matrix transpose |
| `wp.inverse(m)` | Matrix inverse |
| `wp.determinant(m)` | Matrix determinant |
| `wp.trace(m)` | Matrix trace |
| `wp.diag(v)` | Create diagonal matrix from vector |
| `wp.identity(n, dtype)` | Identity matrix |
| `wp.matrix_from_cols(c0, c1, ...)` | Build from column vectors |
| `wp.matrix_from_rows(r0, r1, ...)` | Build from row vectors |
| `m @ v` | Matrix-vector multiplication |
| `m @ n` | Matrix-matrix multiplication |

#### Example

```python
@wp.kernel
def transform_points(
    points: wp.array(dtype=wp.vec3),
    matrix: wp.mat44,
    output: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    p = points[tid]
    
    # Homogeneous transform
    p4 = wp.vec4(p[0], p[1], p[2], 1.0)
    result = matrix @ p4
    
    output[tid] = wp.vec3(result[0], result[1], result[2])
```

---

### 3.3 Quaternion Types

Quaternions represent rotations efficiently, avoiding gimbal lock issues.

| Type | Description |
|------|-------------|
| `wp.quat` | Quaternion (float32, default) |
| `wp.quatf` | Explicit float32 quaternion |
| `wp.quatd` | Float64 quaternion |
| `wp.quath` | Float16 quaternion |

#### Quaternion Operations

| Function | Description |
|----------|-------------|
| `wp.quat(x, y, z, w)` | Create quaternion |
| `wp.quat_identity()` | Identity rotation |
| `wp.quat_from_axis_angle(axis, angle)` | From axis + angle (radians) |
| `wp.quat_from_matrix(m)` | From 3×3 or 4×4 matrix |
| `wp.quat_to_matrix(q)` | Convert to 3×3 matrix |
| `wp.quat_rotate(q, v)` | Rotate vector by quaternion |
| `wp.quat_rotate_inv(q, v)` | Inverse rotation |
| `wp.quat_inverse(q)` | Quaternion inverse |
| `wp.quat_slerp(a, b, t)` | Spherical interpolation |
| `wp.quat_rpy(roll, pitch, yaw)` | From roll-pitch-yaw |
| `wp.quat_to_axis_angle(q)` | Extract axis and angle |
| `q * p` | Quaternion multiplication |

#### Example

```python
@wp.kernel
def rotate_vectors(
    vectors: wp.array(dtype=wp.vec3),
    axis: wp.vec3,
    angle: float,
    output: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    v = vectors[tid]
    
    # Create rotation quaternion
    q = wp.quat_from_axis_angle(axis, angle)
    
    # Rotate vector
    rotated = wp.quat_rotate(q, v)
    
    output[tid] = rotated
```

---

### 3.4 Transform Types

Transforms combine translation (position) and rotation (quaternion) for rigid body representations.

| Type | Description |
|------|-------------|
| `wp.transform` | Transform (float32, default) |
| `wp.transformf` | Explicit float32 |
| `wp.transformd` | Float64 transform |
| `wp.transformh` | Float16 transform |

#### Transform Operations

| Function | Description |
|----------|-------------|
| `wp.transform(pos, rot)` | Create transform |
| `wp.transform_identity()` | Identity transform |
| `wp.transform_get_translation(t)` | Extract position |
| `wp.transform_get_rotation(t)` | Extract rotation |
| `wp.transform_point(t, p)` | Transform a point |
| `wp.transform_vector(t, v)` | Transform a direction |
| `wp.transform_inverse(t)` | Inverse transform |
| `wp.transform_multiply(a, b)` | Compose transforms |
| `wp.transform_compose(pos, quat, scale)` | Create 4×4 matrix |
| `wp.transform_decompose(mat44)` | Extract pos, quat, scale |
| `wp.transform_from_matrix(mat44)` | Create from 4×4 matrix |

#### Example

```python
@wp.kernel
def apply_transforms(
    points: wp.array(dtype=wp.vec3),
    transforms: wp.array(dtype=wp.transform),
    output: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    p = points[tid]
    t = transforms[tid]
    
    # Transform point
    result = wp.transform_point(t, p)
    
    output[tid] = result
```

---

### 3.5 Spatial Types

For robotics and articulated body dynamics.

| Type | Description |
|------|-------------|
| `wp.spatial_vector` | 6D spatial vector (angular, linear) |
| `wp.spatial_matrix` | 6×6 spatial matrix |

---

## 4. Geometry Built-in Functions

### 4.1 Distance & Intersection

| Function | Description |
|----------|-------------|
| `wp.closest_point_edge_edge(p1, q1, p2, q2)` | Closest points between line segments |
| `wp.closest_point_on_tri(p, a, b, c)` | Closest point on triangle |
| `wp.closest_point_on_plane(p, plane_point, plane_normal)` | Project point to plane |

### 4.2 Sampling Functions

| Function | Description |
|----------|-------------|
| `wp.sample_triangle(a, b, c, u, v)` | Point in triangle from barycentric |
| `wp.sample_unit_disk(state)` | Random point on unit disk |
| `wp.sample_unit_sphere(state)` | Random point on unit sphere |
| `wp.sample_unit_hemisphere(state)` | Random point on hemisphere |
| `wp.sample_unit_cube(state)` | Random point in unit cube |

### 4.3 Triangle Operations

| Function | Description |
|----------|-------------|
| `wp.triangle_closest_point(a, b, c, p)` | Closest point on triangle to p |
| `wp.triangle_area(a, b, c)` | Triangle area |

---

## 5. warp.sparse - Sparse Linear Algebra

Block Sparse Row (BSR) matrices for efficient linear algebra on sparse data. Supports CUDA graph capture for allocation-free execution.

### 5.1 Matrix Creation

```python
import warp.sparse as sp

# Create zero matrix
A = sp.bsr_zeros(rows_of_blocks, cols_of_blocks, block_type, device)

# From triplets (row, col, value)
A = sp.bsr_set_from_triplets(
    rows_of_blocks, cols_of_blocks,
    row_indices, col_indices, values,
    device
)

# Copy matrix
B = sp.bsr_copy(A)

# Assign with potential block size change
sp.bsr_assign(src=A, dest=B)
```

### 5.2 Matrix Operations

| Function | Description |
|----------|-------------|
| `sp.bsr_axpy(x, y, alpha, beta)` | `y = alpha*x + beta*y` |
| `sp.bsr_mm(a, b, c, alpha, beta)` | `c = alpha*a@b + beta*c` |
| `sp.bsr_mv(a, x, y, alpha, beta)` | `y = alpha*a@x + beta*y` |
| `sp.bsr_set_transpose(src, dest)` | Transpose matrix |
| `sp.bsr_transposed(a)` | Create transposed view |

### 5.3 Iterative Solvers

```python
from warp.sparse import cg, bicgstab, gmres

# Conjugate Gradient
x, info = cg(A, b, x0=None, tol=1e-6, maxiter=100)

# BiCGSTAB (for non-symmetric matrices)
x, info = bicgstab(A, b, x0=None, tol=1e-6, maxiter=100)

# GMRES
x, info = gmres(A, b, x0=None, tol=1e-6, maxiter=100, restart=30)
```

### 5.4 Example: Sparse Matrix-Vector Multiply

```python
import warp as wp
import warp.sparse as sp

# Create sparse matrix from triplets
rows = wp.array([0, 0, 1, 1, 2], dtype=int)
cols = wp.array([0, 1, 0, 1, 2], dtype=int)
vals = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

A = sp.bsr_set_from_triplets(3, 3, rows, cols, vals)

# Multiply
x = wp.array([1.0, 1.0, 1.0], dtype=float)
y = wp.zeros(3, dtype=float)

sp.bsr_mv(A, x, y)
```

---

## 6. warp.fem - Finite Element Methods

High-level FEM toolkit for solving PDEs on various geometries with GPU acceleration.

### 6.1 Supported Geometries

| Geometry | Description |
|----------|-------------|
| `fem.Grid2D`, `fem.Grid3D` | Regular 2D/3D grids |
| `fem.Trimesh2D`, `fem.Trimesh3D` | Triangle meshes (2D/3D surfaces) |
| `fem.Quadmesh2D`, `fem.Quadmesh3D` | Quadrilateral meshes |
| `fem.Tetmesh` | Tetrahedral volume meshes |
| `fem.Hexmesh` | Hexahedral volume meshes |
| NanoVDB volumes | Sparse volumetric grids |

### 6.2 Function Spaces

Supports:
- Continuous/discontinuous Lagrange (P_k, Q_k) up to order 3
- Serendipity (S_k) up to order 3
- Nédélec (first kind) vector spaces
- Raviart-Thomas vector spaces

```python
import warp.fem as fem

# Create geometry
geo = fem.Grid2D(res=wp.vec2i(64, 64))

# Create function space
space = fem.make_polynomial_space(geo, degree=2)

# Create field
field = space.make_field()
```

### 6.3 Key Operators

| Function | Description |
|----------|-------------|
| `fem.integrate()` | Numerical integration |
| `fem.interpolate()` | Interpolate to field |
| `fem.lookup()` | Position-based field lookup |
| `fem.grad()` | Compute gradient |
| `fem.div()` | Compute divergence |
| `fem.curl()` | Compute curl |

### 6.4 Example: Poisson Equation

```python
import warp as wp
import warp.fem as fem

@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def rhs_form(s: fem.Sample, v: fem.Field):
    return v(s)  # Source term

# Setup
geo = fem.Grid2D(res=wp.vec2i(32, 32))
space = fem.make_polynomial_space(geo, degree=1)

# Create test/trial functions
u = fem.make_trial(space)
v = fem.make_test(space)

# Assemble matrix
A = fem.integrate(laplacian_form, fields={"u": u, "v": v})

# Assemble RHS
b = fem.integrate(rhs_form, fields={"v": v})
```

---

## 7. Array Utilities

### 7.1 Array Creation

| Function | Description |
|----------|-------------|
| `wp.zeros(shape, dtype, device)` | Zero-initialized array |
| `wp.empty(shape, dtype, device)` | Uninitialized array |
| `wp.full(shape, value, dtype, device)` | Array filled with value |
| `wp.ones(shape, dtype, device)` | Array of ones |
| `wp.copy(src, dest)` | Copy array (differentiable) |
| `wp.clone(arr)` | Deep clone array |

### 7.2 NumPy Interop

```python
# Import from NumPy
wp_arr = wp.from_numpy(np_arr, dtype=wp.float32, device="cuda")

# Export to NumPy
np_arr = wp_arr.numpy()

# Zero-copy view (same memory)
np_view = wp_arr.numpy()  # CPU arrays only
```

### 7.3 Scan & Reduction

```python
import warp.utils as utils

# Prefix sum (inclusive)
utils.array_scan(input, output, inclusive=True)

# Prefix sum (exclusive)
utils.array_scan(input, output, inclusive=False)

# Sum reduction
total = utils.array_sum(arr)

# Radix sort
utils.radix_sort_pairs(keys, values)
```

### 7.4 Map Operations

```python
# Apply function element-wise
@wp.func
def my_func(x: float) -> float:
    return x * 2.0

wp.map(my_func, inputs=[a], outputs=[b])
```

---

## 8. Random Number Generation

### 8.1 Random Functions

| Function | Description |
|----------|-------------|
| `wp.rand_init(seed)` | Initialize RNG state |
| `wp.randf(state)` | Random float [0, 1) |
| `wp.randf(state, lo, hi)` | Random float [lo, hi) |
| `wp.randi(state)` | Random integer |
| `wp.randi(state, lo, hi)` | Random integer [lo, hi) |
| `wp.randn(state)` | Normal distribution (μ=0, σ=1) |

### 8.2 Noise Functions

| Function | Description |
|----------|-------------|
| `wp.noise(state, x)` | Perlin noise (1D-4D) |
| `wp.pnoise(state, x, period)` | Periodic Perlin noise |
| `wp.curlnoise(state, x)` | Curl of Perlin noise (divergence-free) |

### 8.3 Example

```python
@wp.kernel
def generate_random(
    seed: int,
    output: wp.array(dtype=float)
):
    tid = wp.tid()
    
    # Initialize state with unique seed per thread
    state = wp.rand_init(seed, tid)
    
    # Generate random values
    r = wp.randf(state)          # [0, 1)
    n = wp.randn(state)          # Normal distribution
    
    output[tid] = r
```

---

## 9. Tile Programming Model

Cooperative tile-based primitives for high-performance matrix operations. Leverages shared memory and thread synchronization.

### 9.1 Tile Operations

| Function | Description |
|----------|-------------|
| `wp.tile_load(arr, shape, offset)` | Load tile from array |
| `wp.tile_store(tile, arr, offset)` | Store tile to array |
| `wp.tile_zeros(shape, dtype)` | Create zero tile |
| `wp.tile_full(shape, value, dtype)` | Create constant tile |
| `wp.tile_map(func, tile)` | Apply function element-wise |
| `wp.tile_reduce(tile, axis)` | Reduce along axis |
| `wp.tile_sum(tile)` | Sum all elements |
| `wp.tile_broadcast(tile, shape)` | Broadcast to larger shape |
| `wp.tile_matmul(a, b)` | Matrix multiplication |
| `wp.tile_transpose(tile)` | Transpose tile |
| `wp.tile_cholesky_solve(L, b)` | Solve Lx=b via Cholesky |

### 9.2 Example: Tiled Matrix Multiply

```python
TILE_M = 32
TILE_N = 32
TILE_K = 32

@wp.kernel
def tiled_matmul(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    
    sum_tile = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
    
    # Accumulate over K dimension
    K = A.shape[1]
    for k in range(0, K, TILE_K):
        a_tile = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i*TILE_M, k))
        b_tile = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k, j*TILE_N))
        
        wp.tile_matmul(a_tile, b_tile, sum_tile)
    
    wp.tile_store(sum_tile, C, offset=(i*TILE_M, j*TILE_N))
```

---

## 10. Atomic Operations

Thread-safe atomic operations for parallel reductions and accumulations.

| Function | Description |
|----------|-------------|
| `wp.atomic_add(arr, idx, value)` | Atomic addition, returns old value |
| `wp.atomic_sub(arr, idx, value)` | Atomic subtraction |
| `wp.atomic_min(arr, idx, value)` | Atomic minimum |
| `wp.atomic_max(arr, idx, value)` | Atomic maximum |
| `wp.atomic_and(arr, idx, value)` | Atomic bitwise AND |
| `wp.atomic_or(arr, idx, value)` | Atomic bitwise OR |
| `wp.atomic_xor(arr, idx, value)` | Atomic bitwise XOR |
| `wp.atomic_cas(arr, idx, compare, value)` | Compare-and-swap |
| `wp.atomic_exch(arr, idx, value)` | Atomic exchange |

### Example: Histogram

```python
@wp.kernel
def histogram(
    values: wp.array(dtype=int),
    bins: wp.array(dtype=int),
    num_bins: int
):
    tid = wp.tid()
    v = values[tid]
    
    if v >= 0 and v < num_bins:
        wp.atomic_add(bins, v, 1)
```

---

## 11. Framework Integration

### 11.1 PyTorch Integration

```python
import torch
import warp as wp

# Convert PyTorch → Warp (zero-copy on same device)
torch_tensor = torch.randn(100, 3, device="cuda")
wp_array = wp.from_torch(torch_tensor)

# Convert Warp → PyTorch (differentiable!)
wp_array = wp.zeros(100, dtype=wp.vec3, device="cuda", requires_grad=True)
torch_tensor = wp.to_torch(wp_array)

# Gradients flow through wp.to_torch()
```

### 11.2 JAX Integration

```python
import jax.numpy as jnp
import warp as wp

# Convert JAX → Warp
jax_array = jnp.zeros((100, 3))
wp_array = wp.from_jax(jax_array)

# Convert Warp → JAX
wp_array = wp.zeros(100, dtype=float, device="cuda")
jax_array = wp.to_jax(wp_array)

# Compatible with jax.pmap() for multi-device
```

### 11.3 NumPy Integration

```python
import numpy as np
import warp as wp

# NumPy → Warp
np_array = np.zeros((100, 3), dtype=np.float32)
wp_array = wp.from_numpy(np_array, dtype=wp.vec3, device="cuda")

# Warp → NumPy
np_array = wp_array.numpy()
```

### 11.4 Array Interface Support

Warp arrays support standard array interfaces:

```python
# __array_interface__ (CPU)
# __cuda_array_interface__ (GPU)

# Works with CuPy, Numba, etc.
import cupy as cp
cupy_array = cp.asarray(wp_array)
```

---

## 12. Automatic Differentiation

Warp kernels are fully differentiable for optimization and machine learning applications.

### 12.1 Enabling Gradients

```python
# Create array with gradients enabled
arr = wp.zeros(n, dtype=float, requires_grad=True, device="cuda")
```

### 12.2 Tape-Based Differentiation

```python
# Create tape
tape = wp.Tape()

# Record operations
with tape:
    wp.launch(forward_kernel, dim=n, inputs=[x], outputs=[y])
    wp.launch(loss_kernel, dim=1, inputs=[y], outputs=[loss])

# Backward pass
tape.backward(loss)

# Access gradients
x_grad = tape.gradients[x]
```

### 12.3 Gradient Utilities

| Function | Description |
|----------|-------------|
| `wp.Tape()` | Create recording tape |
| `tape.backward(loss)` | Compute gradients |
| `tape.gradients[arr]` | Access gradient array |
| `tape.zero()` | Zero all gradients |
| `tape.reset()` | Clear tape |
| `wp.grad(arr)` | Get gradient array directly |

### 12.4 Jacobian Computation

```python
from warp.autograd import jacobian, jacobian_fd, gradcheck

# Compute Jacobian matrix
J = jacobian(func, inputs=[x], outputs=[y])

# Finite difference Jacobian (for verification)
J_fd = jacobian_fd(func, inputs=[x], outputs=[y], eps=1e-4)

# Verify gradient correctness
gradcheck(func, inputs=[x], outputs=[y], eps=1e-4, atol=1e-3)
```

### 12.5 Example: Optimization Loop

```python
import warp as wp
import warp.optim as optim

# Parameters
params = wp.zeros(10, dtype=float, requires_grad=True)

# Optimizer
optimizer = optim.Adam([params], lr=0.01)

for i in range(100):
    tape = wp.Tape()
    
    with tape:
        wp.launch(forward, dim=n, inputs=[params], outputs=[loss])
    
    tape.backward(loss)
    
    optimizer.step([tape.gradients[params]])
    tape.zero()
```

---

## 13. Best Practices

### 13.1 Performance Tips

1. **Kernel Launch Configuration**
   ```python
   # Specify block_dim for GPU occupancy
   wp.launch(kernel, dim=n, inputs=[...], block_dim=256)
   ```

2. **Use Tile Operations for Matrix-Heavy Workloads**
   - Leverage shared memory
   - Reduce global memory bandwidth

3. **CUDA Graphs for Repeated Sequences**
   ```python
   wp.capture_begin()
   # ... kernel launches ...
   graph = wp.capture_end()
   wp.capture_launch(graph)
   ```

4. **Disable Gradients When Not Needed**
   ```python
   arr = wp.zeros(n, dtype=float, requires_grad=False)
   ```

5. **Use `length_sq()` Instead of `length()`**
   - Avoids expensive square root when comparing distances

6. **Tune BVH `leaf_size`**
   - 1 for intersection queries
   - 4-8 for closest-point queries

### 13.2 Memory Management

1. **Keep Spatial Primitives in Scope**
   ```python
   # ❌ Wrong - mesh may be garbage collected
   def create_mesh():
       return wp.Mesh(points, indices)
   
   # ✅ Correct - keep reference
   mesh = wp.Mesh(points, indices)
   wp.launch(kernel, inputs=[mesh.id])
   ```

2. **Use `ScopedDevice` for Explicit Control**
   ```python
   with wp.ScopedDevice("cuda:0"):
       arr = wp.zeros(n, dtype=float)
   ```

3. **Pre-allocate Arrays**
   ```python
   # ❌ Avoid allocations in loops
   for i in range(100):
       result = wp.zeros(n)  # Bad!
   
   # ✅ Pre-allocate
   result = wp.zeros(n)
   for i in range(100):
       wp.launch(kernel, outputs=[result])
   ```

4. **Use In-Place Operations for CUDA Graph Compatibility**
   ```python
   bvh.rebuild()  # In-place, graph safe
   mesh.refit()   # In-place, graph safe
   ```

### 13.3 Debugging

1. **Verbose Mode**
   ```python
   wp.config.verbose = True
   ```

2. **Synchronize for Error Detection**
   ```python
   wp.launch(kernel, dim=n, inputs=[...])
   wp.synchronize()  # Catches errors immediately
   ```

3. **Runtime Assertions**
   ```python
   @wp.kernel
   def checked_kernel(arr: wp.array(dtype=float)):
       tid = wp.tid()
       val = arr[tid]
       wp.expect_near(val, 1.0, 0.01)  # Assert value ≈ 1.0
   ```

4. **Print from Kernels**
   ```python
   @wp.kernel
   def debug_kernel(arr: wp.array(dtype=float)):
       tid = wp.tid()
       wp.printf("tid=%d, val=%f\n", tid, arr[tid])
   ```

### 13.4 Code Organization

```python
# Recommended project structure
project/
├── kernels/
│   ├── physics.py      # Physics kernels
│   ├── geometry.py     # Geometry processing
│   └── render.py       # Rendering kernels
├── utils/
│   ├── mesh_io.py      # Mesh loading/saving
│   └── visualization.py
└── main.py
```

---

## Quick Reference Card

### Common Imports

```python
import warp as wp
import warp.sparse as sp
import warp.fem as fem
import warp.utils as utils
```

### Kernel Template

```python
@wp.kernel
def my_kernel(
    input: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
    param: float
):
    tid = wp.tid()
    
    # Read input
    v = input[tid]
    
    # Compute
    result = wp.length(v) * param
    
    # Write output
    output[tid] = result
```

### Launch Pattern

```python
wp.launch(
    kernel=my_kernel,
    dim=n,
    inputs=[input_arr, param],
    outputs=[output_arr],
    device="cuda"
)
```

---

## Resources

- **Documentation**: https://nvidia.github.io/warp/
- **GitHub**: https://github.com/NVIDIA/warp
- **Examples**: `warp/examples/` directory
- **Tutorials**: NVIDIA Accelerated Computing Hub

---

*Documentation compiled from NVIDIA Warp v1.11.0*
