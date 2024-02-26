import numpy as np
from numba import jit # , cuda
from scipy.sparse import coo_matrix
from synmorph.utils import grid_xy

"""
Triangulation functions
-----------------------

Misc functions that have been optimized for the triangulated data-structure.

This includes rolls etc. 

Also includes functions to convert an array of cell-properties to the triangulated form, and to sum components from each triangle back into a cell-type property 1D array 
"""

# @cuda.jit
# def remove_repeats(tri, n_c, output, output_size):
#     # CUDA kernel to remove repeats in parallel
#     # Each thread will handle one row of the input array
#     x = cuda.grid(1)
#     row_length = tri.shape[1]  # Assuming all rows have the same length
#
#     if x < tri.shape[0]:
#         is_unique = True
#         current_row_temp = cuda.local.array(shape=(row_length,), dtype=np.int64)
#         other_row_temp = cuda.local.array(shape=(row_length,), dtype=np.int64)
#
#         # Process the current row
#         order_tris_single(tri[x], n_c, row_length, current_row_temp)
#
#         # Compare the current row with all other rows
#         for i in range(tri.shape[0]):
#             if x != i:
#                 order_tris_single(tri[i], n_c, row_length, other_row_temp)
#                 if np.all(current_row_temp == other_row_temp):
#                     is_unique = False
#                     break
#
#         # If the row is unique, add it to the output array
#         if is_unique:
#             size = cuda.atomic.add(output_size, 0, 1)
#             for j in range(row_length):
#                 output[size, j] = current_row_temp[j]
#
# @cuda.jit
# def order_tris_cuda(tri, n_c, ordered_tri):
#     # CUDA kernel to reorder triangles
#     x = cuda.grid(1)
#
#     if x < tri.shape[0]:
#         ordered_tri[x] = order_tris_single(tri[x], n_c)
#
# @cuda.jit(device=True)
# def order_tris_single(row, n_c, row_length, result):
#     # Use provided 'result' array instead of creating a new one
#     mod_row = cuda.local.array(shape=(row_length,), dtype=np.int64)
#
#     # Manual modulus operation for each element in the row
#     for i in range(row_length):
#         mod_row[i] = row[i] % n_c
#
#     # Find the index of the minimum value in mod_row
#     min_index = 0
#     for i in range(1, row_length):
#         if mod_row[i] < mod_row[min_index]:
#             min_index = i
#
#     # Manually roll the row to reorder
#     for i in range(row_length):
#         new_index = (i - min_index) % row_length
#         result[new_index] = row[i]


## todo: potential bottleneck
@jit(nopython=True, parallel=True)
def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    ordered_tri = np.zeros(tri.shape, dtype=np.int32)
    for i, row in enumerate(tri):
        ordered_tri[i] = np.roll(row, -np.argmin(row))

    return ordered_tri

@jit(nopython=True)
def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)
    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)
    Then remove repeated rows via lexsort.
    NB: order of vertices changes via the conventions of lexsort
    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    new_list = []
    for row in tri:
        l = list(row)
        if not l in new_list:
            new_list.append(l)

    return np.asarray(new_list, dtype=np.int32)

@jit(nopython=True)
def make_y(x, L):
    """
    Makes the (9) tiled set of coordinates used to perform the periodic triangulation.
    :param x: Cell centroids (n_c x 2) np.float32 array
    :param L: Side length of the bounding box
    :return: Tiled set of coordinates (9n_c x 2) np.float32 array
    """
    n_c, nd = x.shape
    y = np.zeros((n_c * 9, nd), dtype=np.float32)
    for k in range(9):
        y[k * n_c : (k + 1) * n_c] = x + L * grid_xy[k]
    return y


@jit(nopython=True)
def tnorm(x):
    """
    Calculate the L1 norm of a set of vectors that are given in triangulated form:

    (nv x 3 x 2) ->> (nv x 3)
    :param x:
    :return:
    """
    return np.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)


@jit(nopython=True)
def tri_call(val, tri):
    """
    when val has shape (n,3)
    Equiv. to:
    >> val[tri]
    :param val:
    :param tri:
    :return:
    """
    return val.take(tri.ravel()).reshape(-1, 3)


@jit(nopython=True)
def tri_call3(val, tri):
    """
    When val has shape (n,3,2)
    Equiv to:
    >> val[tri]
    :param val:
    :param tri:
    :return:
    """
    vali, valj = val[:, 0], val[:, 1]
    return np.dstack((tri_call(vali, tri), tri_call(valj, tri)))


@jit(nopython=True)
def tri_mat_call(mat, tri, direc=-1):
    """
    If matrix element {i,j} corresponds to the edge value connecting cells i and j,
    then this function returns the edge value connecting a vertex to its counter-clockwise neighbour
    Or equivalently the case where j is CW to i in a given triangle.
    Swap CCW for CW if direc = 1
    :param mat:
    :param tri:
    :param direc:
    :return:
    """
    # return np.dstack((mat[i, j] for (i, j) in zip(tri, roll(tri, direc))))

    nv = tri.shape[0]
    tmat = np.empty((nv, 3))
    tri_roll = roll(tri, direc)
    for k in range(nv):
        for m in range(3):
            tri_i, tri_k = tri[k, m], tri_roll[k, m]
            tmat[k, m] = mat[tri_i, tri_k]
    return tmat


@jit(nopython=True)
def roll(x, direc=1):
    """
    Jitted equivalent to np.roll(x,-direc,axis=1)
    direc = 1 --> counter-clockwise
    direc = -1 --> clockwise
    :param x:
    :return:
    """
    if direc == -1:  # old "roll_forward"
        return np.column_stack((x[:, 2], x[:, :2]))
    elif direc == 1:  # old "roll_reverse"
        return np.column_stack((x[:, 1:3], x[:, 0]))


@jit(nopython=True)
def roll3(x, direc=1):
    """
    Like roll, but when x has shape (nv x 3 x 2) ie is a vector, rather than scalar, quantity.
    :param x:
    :param direc:
    :return:
    """
    x_out = np.empty_like(x)
    x_out[:, :, 0], x_out[:, :, 1] = roll(x[:, :, 0], direc=direc), roll(
        x[:, :, 1], direc=direc
    )
    return x_out


@jit(nopython=True)
def tri_sum(n_c, CV_matrix, tval):
    val_sum = np.zeros(n_c)
    for i in range(3):
        val_sum += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(tval[:, i])
    return val_sum


@jit(nopython=True)
def cosine_rule(a, b, c):
    return np.arccos((b**2 + c**2 - a**2) / (2 * b * c))


@jit(nopython=True)
def clip(x, xmin, xmax):
    xflat = x.ravel()
    minmask = xflat < xmin
    maxmask = xflat > xmax
    xflat[minmask] = xmin
    xflat[maxmask] = xmax
    return xflat.reshape(x.shape)


@jit(nopython=True)
def replace_val(x, mask, xnew):
    xflat = x.ravel()
    maskflat = mask.ravel()
    xflat[maskflat] = xnew
    return xflat.reshape(x.shape)


@jit(nopython=True)
def replace_vec(x, mask, xnew):
    xflat = x.ravel()
    maskflat = mask.ravel()
    xflat[maskflat] = xnew.ravel()[maskflat]
    return xflat.reshape(x.shape)


@jit(nopython=True)
def tcross(A, B):
    """
    Cross product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return A[:, :, 0] * B[:, :, 1] - A[:, :, 1] * B[:, :, 0]


@jit(nopython=True)
def tdot(A, B):
    """
    Dot product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return A[:, :, 0] * B[:, :, 0] + A[:, :, 1] * B[:, :, 1]


@jit(nopython=True)
def touter(A, B):
    """
    Outer product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return np.dstack(
        (
            np.dstack((A[:, :, 0] * B[:, :, 0], A[:, :, 1] * B[:, :, 0])),
            np.dstack((A[:, :, 0] * B[:, :, 1], A[:, :, 1] * B[:, :, 1])),
        )
    ).reshape(-1, 3, 2, 2)


@jit(nopython=True)
def tdet(A):
    a1, a2, a3 = A[:, :, 0]
    b1, b2, b3 = A[:, :, 1]
    c1, c2, c3 = A[:, :, 2]
    return (
        a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)
    )


@jit(nopython=True)
def tidentity(nv):
    """
    Generate an identity matrix for each element of a triangulation

    I ~ (nv x 3 x 2 x 2), where the last two dims are an identity matrix.
    :param nv:
    :return:
    """
    I = np.zeros((nv, 3, 2, 2))
    I[:, :, 0, 0] = 1
    I[:, :, 1, 1] = 1
    return I


@jit(nopython=True)
def tmatmul(A, B):
    """

    matrix multiplication of two triangulated matrices. Not in use atm.
    :param A:
    :param B:
    :return:
    """
    AT, BT = A.T, B.T
    return np.dstack(
        (
            (AT[0] * BT[0, 0] + AT[1] * BT[1, 0]).T,
            (AT[0] * BT[0, 1] + AT[1] * BT[1, 1]).T,
        )
    )


@jit(nopython=True)
def sum_tri(A):
    return A[:, 0] + A[:, 1] + A[:, 2]


@jit(nopython=True)
def prod_tri(A):
    return A[:, 0] * A[:, 1] * A[:, 2]


def assemble_tri(tval, tri):
    """
    Sum all components of a given cell property.
    I.e. (nv x 3) --> (nc x 1)
    :param tval:
    :param tri:
    :return:
    """
    vals = coo_matrix(
        (tval.ravel(), (tri.ravel(), np.zeros_like(tri.ravel()))),
        shape=(tri.max() + 1, 1),
    )
    return vals.toarray().ravel()


def assemble_tri3(tval, tri):
    """
    The same as above, but for vector quantities

    (nv x 3 x 2) --> (nc x 2)
    :param tval:
    :param tri:
    :return:
    """
    vals = coo_matrix(
        (tval.ravel(), (np.repeat(tri.ravel(), 2), np.tile((0, 1), tri.size))),
        shape=(tri.max() + 1, 2),
    )
    return vals.toarray()


@jit(nopython=True)
def find_neighbour_val(A, neighbours):
    """
    Check this
    :param A:
    :param neighbours:
    :return:
    """
    B = np.empty_like(A)
    for i, tneighbour in enumerate(neighbours):
        for j, neighbour in enumerate(tneighbour):
            B[i, j] = A[neighbour, j]
    return B


@jit(nopython=True)
def repeat_mat(A):
    return np.dstack((A, A, A, A)).reshape(-1, 3, 2, 2)


@jit(nopython=True)
def repeat_vec(A):
    return np.dstack((A, A))


@jit(nopython=True)
def circumcenter(C, L):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    r_mean = (ri + rj + rk) / 3
    disp = r_mean - L / 2
    ri, rj, rk = np.mod(ri - disp, L), np.mod(rj - disp, L), np.mod(rk - disp, L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    vs = np.empty((ax.size, 2), dtype=np.float64)
    vs[:, 0], vs[:, 1] = ux, uy
    vs = np.mod(vs + disp.T, L)
    return vs


def normalise(x):
    return (x - x.min()) / (x.max() - x.min())

## todo: set seed here
def hexagonal_lattice(rows=3, cols=3, noise=0.0005, A=None):
    """
    Assemble a hexagonal lattice
    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    points = []
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    if A is not None:
        points = points * np.sqrt(2 * np.sqrt(3) / 3) * np.sqrt(A)
    return points


@jit(nopython=True)
def get_neighbours(tri, neigh=None, Range=None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri, dtype=np.int32) * -1
    if Range is None:
        Range = np.arange(n_v)

    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in Range:  # range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                neighb, l = np.nonzero(
                    (tri_compare[:, :, 0] == tri_i[k, 0])
                    * (tri_compare[:, :, 1] == tri_i[k, 1])
                )
                neighb, l = neighb[0], l[0]
                neigh[j, k] = neighb
                neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh


@jit(nopython=True)
def tri_angles_periodic(x, tri, L):
    """
    Same as **tri_angles** apart from accounts for periodic triangulation (i.e. the **L**)

    Find angles that make up each triangle in the triangulation. By convention, column i defines the angle
    corresponding to cell centroid i

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param L: Domain size (np.float32)
    :return: tri_angles (n_v x 3) np.flaot32 array (in radians)
    """
    three = np.array([0, 1, 2])
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((tri.shape[0], 3, 2))
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (np.mod(C[:, i_b, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
        np.mod(C[:, i_b, 1] - C[:, i_c, 1] + L / 2, L) - L / 2
    ) ** 2
    b2 = (np.mod(C[:, :, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
        np.mod(C[:, :, 1] - C[:, i_c, 1] + L / 2, L) - L / 2
    ) ** 2
    c2 = (np.mod(C[:, i_b, 0] - C[:, :, 0] + L / 2, L) - L / 2) ** 2 + (
        np.mod(C[:, i_b, 1] - C[:, :, 1] + L / 2, L) - L / 2
    ) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)
    return Angles



# @cuda.jit
# def order_tris_cuda(tri, ordered_tri):
#     # Determine the thread's absolute position within the grid
#     # x = cuda.grid(1)
#     #
#     # # Check if the thread is within the bounds of the array
#     # if x < tri.shape[0]:
#     #     row = tri[x]
#     #     ordered_tri[x] = np.roll(row, -np.argmin(row))
#     x = cuda.grid(1)
#     if x < tri.shape[0]:
#         row = tri[x]
#         min_index = 0
#         min_value = row[0]
#         for i in range(1, len(row)):
#             if row[i] < min_value:
#                 min_value = row[i]
#                 min_index = i
#         # Implementing a manual roll operation
#         for i in range(len(row)):
#             ordered_tri[x, i] = row[(i + min_index) % len(row)]

# @cuda.jit
# def mark_duplicates_kernel(sorted_tri, duplicate_flags):
#     # x = cuda.grid(1)
#     # if x < sorted_tri.shape[0] - 1:
#     #     # Mark this row as a duplicate if it is the same as the next row
#     #     duplicate_flags[x] = np.all(sorted_tri[x] == sorted_tri[x + 1])
#     x = cuda.grid(1)
#     if x < sorted_tri.shape[0] - 1:
#         is_duplicate = True
#         for i in range(sorted_tri.shape[1]):
#             if sorted_tri[x, i] != sorted_tri[x + 1, i]:
#                 is_duplicate = False
#                 break
#         duplicate_flags[x] = is_duplicate
#
# # This function is a placeholder for a parallel sorting algorithm
# def parallel_sort(tri):
#     # Implement parallel sorting here
#     return np.sort(tri, axis=0)
#
# @cuda.jit
# def compact_array_kernel(sorted_tri, duplicate_flags, compacted_tri, new_indices):
#     x = cuda.grid(1)
#     if x < sorted_tri.shape[0] and not duplicate_flags[x]:
#         new_index = new_indices[x]
#         compacted_tri[new_index] = sorted_tri[x]
#
# def remove_repeats(tri, n_c):
#     n = tri.shape[0]
#     # ordered_tri = np.empty_like(tri)
#     tri_device = cuda.to_device(np.mod(tri, n_c))
#     ordered_tri_device = cuda.device_array(tri.shape)
#     # Define the number of threads per block and the number of blocks per grid
#     threads_per_block = 32  # This is a typical value, but it can be tuned based on your GPU architecture
#     blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
#
#     # Now launch the kernel with the specified configuration
#     order_tris_cuda[blocks_per_grid, threads_per_block](tri_device, ordered_tri_device)
#
#     # Sort the array using a parallel sort (to be implemented)
#     sorted_tri = parallel_sort(tri_device.copy_to_host())
#
#     # Allocate an array to mark duplicates
#     duplicate_flags = np.zeros(sorted_tri.shape[0], dtype=np.bool_)
#
#     # Calculate grid configuration
#     threads_per_block = 32
#     blocks_per_grid = (sorted_tri.shape[0] + (threads_per_block - 1)) // threads_per_block
#
#     # Mark duplicates
#     sorted_tri_device = cuda.to_device(sorted_tri)
#     mark_duplicates_kernel[blocks_per_grid, threads_per_block](sorted_tri_device, cuda.to_device(duplicate_flags))
#
#     new_indices_host = np.cumsum(~duplicate_flags) - 1
#     new_indices_device = cuda.to_device(new_indices_host)
#     compacted_tri_device = cuda.device_array((new_indices_host[-1] + 1, 3), dtype=np.int32)
#
#     # Calculate grid configuration
#     threads_per_block = 32
#     blocks_per_grid = (sorted_tri.shape[0] + (threads_per_block - 1)) // threads_per_block
#
#     # Compact the array
#     compact_array_kernel[blocks_per_grid, threads_per_block](sorted_tri, duplicate_flags, compacted_tri_device,
#                                                              new_indices_device)
#
#     return compacted_tri_device.copy_to_host()