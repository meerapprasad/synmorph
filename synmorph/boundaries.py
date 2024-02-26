from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import shapely.geometry as geometry
import math
from scipy.interpolate import splprep, splev
from shapely.geometry import Point
import matplotlib.pyplot as plt
## https://github.com/dwyerk/boundaries/blob/master/concave_hulls.ipynb ##
# todo: use this to detect boundaries
# points = np.random.rand(100, 2)
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([(point.x, point.y) for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def concave_hull_spline(points):
    # Perform cubic spline interpolation on the hull points
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)  # per=True for periodic boundary
    unew = np.linspace(0, 1, 100)
    spline_output = splev(unew, tck)

    return spline_output

def plot_concave_hull_spline(concave_hull, spline_output, points):
    plt.figure(figsize=(10, 6))
    plt.plot(spline_output[0], spline_output[1], label='Cubic Spline')
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='red', label='MEPs')
    x, y = concave_hull.exterior.xy
    plt.plot(x, y, '-', color='green', label='Concave Hull')
    plt.xlabel("spatial x")
    plt.ylabel("spatial y")
    plt.title("Cubic Spline on Concave Hull of Points")
    plt.legend()
    plt.show()


def compute_boundary(points, alpha=1.0):
    # points = np.random.rand(100, 2)
    tuple_list = [tuple(row) for row in points]
    # Assuming 'tuple_list' is your list of tuples
    shapely_points = [Point(x, y) for x, y in tuple_list]
    concave_hull, edge_points = alpha_shape(shapely_points, alpha=alpha)

    # Compute spline for the convex hull of the points
    x, y = concave_hull.exterior.xy
    spline_output = concave_hull_spline(np.stack([np.array(x), np.array(y)]).T)
    return np.stack(spline_output).T #, concave_hull, edge_points
    # plot_concave_hull_spline(concave_hull, spline_output, points)


def compute_min_distances_vectorized(array1, array2):
    # Expand dimensions of array1 and array2 for broadcasting
    # array1 shape becomes (81, 1, 2) and array2 shape becomes (1, 100, 2) for broadcasting
    array1_expanded = np.expand_dims(array1, axis=1)
    array2_expanded = np.expand_dims(array2, axis=0)

    # Compute squared distances between all pairs of points (broadcasting involved here)
    squared_distances = np.sum((array1_expanded - array2_expanded) ** 2, axis=2)

    # Find the minimum squared distance for each point in array1 to points in array2
    min_squared_distances = np.min(squared_distances, axis=1)

    # Return the square root of the minimum squared distances (Euclidean distances)
    return np.sqrt(min_squared_distances)

