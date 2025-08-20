import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import beta
import json


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
    return (
        np.linalg.norm(closest_point_on_p - closest_point_on_q),
        closest_point_on_p,
        closest_point_on_q,
    )


def offset_curve(coordinates, offset_distance):
    """
    Offset a curve by a given distance
    """
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # calculate derivatives (ie tangent vector)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # compute normal vectors (ie tangent vector rotated by 90 degrees)
    normal_x = -dy
    normal_y = dx

    # normalise normal vectors
    normal_magnitude = np.sqrt(normal_x**2 + normal_y**2)
    normal_x /= normal_magnitude
    normal_y /= normal_magnitude

    # ofset original curve
    offset_x = x + offset_distance * normal_x
    offset_y = y + offset_distance * normal_y

    offset_coordinates = np.column_stack((offset_x, offset_y))

    return offset_coordinates


def generate_monotonic_beta_curve(
    start_pt,
    end_pt,
    paramsFile,
    weightingType,
    generationType,
    num_basis=2,
    resolution=50,
):
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

    # TODO set a seed for the random generation - make it deterministic for regeneration

    if generationType == "random":
        # Random beta parameters and non-negative weights
        beta_params = [
            (np.random.uniform(0, 15), np.random.uniform(0, 15))
            for _ in range(num_basis)
        ]
        if num_basis == 1:
            weights = np.array([1.0])
        elif weightingType == "random":
            weights = np.random.rand(num_basis)
        elif weightingType == "equal":
            weights = np.ones(num_basis) / num_basis
        else:
            raise ValueError("Invalid weighting type. Use 'random' or 'equal'.")

        # update the JSON file with the parameters

        with open(paramsFile, "r") as f:
            data = json.load(f)

        for i in range(len(beta_params)):
            if i >= len(data):
                data.append({})  # Add missing entry if needed
            data[i]["basis"] = list(beta_params[i])
            data[i]["weight"] = float(weights[i])

        # Save back
        with open(paramsFile, "w") as f:
            json.dump(data, f, indent=2)

    elif generationType == "preGenerated":

        with open(paramsFile, "r") as f:
            data = json.load(f)

            paramArray = []
            weightArray = []
            for entry in data:
                paramArray.extend(entry["basis"])
                weightArray.extend(entry["weight"])

        beta_params = np.array(paramArray).reshape(-1, 2)
        weights = np.array(weightArray)
    # print(paramArray)
    # print(weightArray)

    # Normalized x values
    x_norm = np.linspace(0, 1, resolution)

    # weights = np.random.rand(num_basis)
    # weights = np.ones(num_basis) / num_basis
    # print('weights', weights)

    # np.savetxt('betaParams.txt', np.vstack((beta_params, weights)))
    # print('stack',np.vstack((beta_params, weights))[:2])

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


def main(num_basis, paramsFile, weightingType="random", resolution=50):

    validGeom = False

    # Define start and end points
    start_point = (17.979, -14.515)
    end_point = (6.596, -3.132)

    # Generate the curve
    curve, beta_params, weights = generate_monotonic_beta_curve(
        start_point,
        end_point,
        paramsFile,
        "random",
        "preGenerated",
        num_basis=num_basis,
        resolution=50,
    )

    offset_coordinates = offset_curve(curve, 0.249)

    lowerCurveUpperSurface = np.copy(curve)
    lowerCurveUpperSurface[:, 1] -= 3
    # print(curve.shape)
    # print(offset_coordinates.shape)
    # print(lowerCurveUpperSurface.shape)

    # Main loop over segment pairs
    minimumSeparation = float("inf")
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

    print("Minimum separation between curves:", minimumSeparation)
    if minimumSeparation < 0.675:
        print('Minimum separation is less than 0.675", geometry is invalid.')
        validGeom = False
    else:
        validGeom = True

    while not validGeom:

        # Generate a new curve
        curve, beta_params, weights = generate_monotonic_beta_curve(
            start_point,
            end_point,
            paramsFile,
            "random",
            "random",
            num_basis=num_basis,
            resolution=resolution,
        )

        np.savetxt("runDirectory/spline.txt", curve)

        offset_coordinates = offset_curve(curve, 0.249)

        lowerCurveUpperSurface = np.copy(curve)
        lowerCurveUpperSurface[:, 1] -= 3

        minimumSeparation = float("inf")
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

        print("Minimum separation between curves:", minimumSeparation)
        if minimumSeparation < 0.675:
            print('Minimum separation is less than 0.675", geometry is invalid.')
            validGeom = False
        else:
            validGeom = True

    # function to check when to stop offset coordinates and connect to the main stack

    # print(len(offset_coordinates))

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

    np.savetxt("runDirectory/spline.txt", curve)

    np.savetxt("runDirectory/offset_spline.txt", np.flip(offset_coordinates, axis=0))

    print("Curve validity check complete.")

    # print(len(offset_coordinates))
