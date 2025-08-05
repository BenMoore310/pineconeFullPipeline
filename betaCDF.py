import numpy as np
import random
import math
from scipy.stats import beta
import argparse


def offset_curve(coordinates, offset_distance):
    """
    Offset a curve by a given distance
    Args:
        coordinates (numpy array): Array of (x, y) points
        offset_distance (float): Distance to offset the curve
    Returns:
        offset_coordinates (numpy array): Array of offset (x, y) points
    """
    x = coordinates[:,0]
    y = coordinates[:,1]

    #calculate derivatives (ie tangent vector)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    #compute normal vectors (ie tangent vector rotated by 90 degrees)
    normal_x = -dy
    normal_y = dx

    #normalise normal vectors
    normal_magnitude = np.sqrt(normal_x**2 + normal_y**2)
    normal_x /= normal_magnitude
    normal_y /= normal_magnitude

    #ofset original curve
    offset_x = x + offset_distance * normal_x
    offset_y = y + offset_distance * normal_y

    offset_coordinates = np.column_stack((offset_x, offset_y))

    return offset_coordinates


def segment_to_segment_distance(p1, p2, q1, q2):
    """
    Returns the minimum distance between two line segments p1-p2 and q1-q2.
    """
    def clamp(v, min_val, max_val):
        return max(min_val, min(v, max_val))

    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denominator = a * c - b * b
    if denominator == 0:
        sc = 0.0
    else:
        sc = (b * e - c * d) / denominator
        sc = clamp(sc, 0.0, 1.0)

    tc = (a * e - b * d) / denominator if denominator != 0 else 0.0
    tc = clamp(tc, 0.0, 1.0)

    closest_point_on_p = p1 + sc * u
    closest_point_on_q = q1 + tc * v
    return np.linalg.norm(closest_point_on_p - closest_point_on_q), closest_point_on_p, closest_point_on_q



def generate_monotonic_beta_curve(start_pt, end_pt, num_basis, resolution, inputFile):
    """
    Generate a 2D monotonic curve from start_pt to end_pt using random Beta CDFs.

    Args:
        start_pt (tuple): (x_start, y_start)
        end_pt (tuple): (x_end, y_end)
        num_basis (int): Number of Beta CDF basis functions
        resolution (int): Number of points in the curve

    Returns:
        Q (numpy array): Array of (x, y) points
    """
    # Normalized x values
    x_norm = np.linspace(0, 1, resolution)

    # Random beta parameters and non-negative weights
    # beta_params = [(np.random.uniform(0, 15), np.random.uniform(0, 15)) for _ in range(num_basis)]

    inputParams = np.loadtxt(inputFile)



    beta_params = inputParams[:num_basis]
    print('beta params =', beta_params)
    # weights = np.random.rand(num_basis)
    # weights = np.ones(num_basis) / num_basis
    weights = inputParams[-1]

    print('weights', weights)

    # Build the curve in normalized space
    y_norm = np.zeros_like(x_norm)
    for (a, b), w in zip(beta_params, weights):
        y_norm += w * beta.cdf(x_norm, a, b)

    # Normalize y to [0,1]
    y_norm -= y_norm.min()
    y_norm /= y_norm.max()

    # Map x and y to target coordinates
    x_vals = start_pt[0] + (end_pt[0] - start_pt[0]) * x_norm
    y_vals = start_pt[1] + (end_pt[1] - start_pt[1]) * y_norm

    Q = np.stack((x_vals, y_vals), axis=1)
    return Q, beta_params, weights


def main(num_basis, resolution, inputFile):


    # Define start and end points
    start_point = (17.979, -14.515)
    end_point   = (6.596, -3.132)

    # Generate the curve
    curve, beta_params, weights = generate_monotonic_beta_curve(start_point, end_point, num_basis, resolution, inputFile)

    np.savetxt('spline.txt', curve)
    
    offset_coordinates = offset_curve(curve, 0.249)

    lowerCurveUpperSurface = np.copy(curve)
    lowerCurveUpperSurface[:,1] -= 3
    # print(curve.shape)
    # print(offset_coordinates.shape)
    # print(lowerCurveUpperSurface.shape)

    minimumSeparation = float('inf')
    for i in range(len(curve) - 1):
        p1 = offset_coordinates[i]
        p2 = offset_coordinates[i + 1]
        for j in range(len(lowerCurveUpperSurface) - 1):
            q1 = lowerCurveUpperSurface[j]
            q2 = lowerCurveUpperSurface[j + 1]
            dist, pt_p, pt_q = segment_to_segment_distance(p1, p2, q1, q2)
            if dist < minimumSeparation:
                minimumSeparation = dist
                min_pts = (pt_p, pt_q)
                min_indices = (i, j)

    print('Minimum separation between curves:', minimumSeparation)
    if minimumSeparation < 0.675:
        print('Minimum separation is less than 0.675", geometry is invalid.')
 

    to_delete = []

    for i in range(0, len(offset_coordinates)):

        if offset_coordinates[i][0] < 7.109 and offset_coordinates[i][1] > -3.999:
            to_delete.append(i)

    offset_coordinates = np.delete(offset_coordinates, to_delete, 0)

    second_delete = []

    for i in range(0, len(offset_coordinates)):

        if offset_coordinates[i][0] <= 6.596:
            second_delete.append(i)

    offset_coordinates = np.delete(offset_coordinates, second_delete, 0)

    np.savetxt('offset_spline.txt', np.flip(offset_coordinates, axis=0))


#eg: python3.11 betaCDF.py --num_basis 4 --resolution 50 --inputFile betaParams.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a monotonic beta curve.")
    parser.add_argument(
        "--num_basis",
        type=int,
        default=5,
        help="number of beta basis functions to use",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="number of points in the curve",
    )

    parser.add_argument(
        "--inputFile",
        type=str,
        default=None,
        help="input file to read the curve from",
    )
    args = parser.parse_args()

    main(args.num_basis, args.resolution, args.inputFile)