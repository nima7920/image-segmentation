import numpy as np
import matplotlib.pyplot as plt
import math


def readPoints(file):
    f = open(file)
    n = int(f.readline())
    points = np.zeros((2, 0))
    for line in f.readlines():
        point = line.split(" ")
        [x, y] = float(point[0]), float(point[1])
        points = np.c_[points, [x, y]]
    return n, points


def get_random_centers(points, k):
    mask = np.random.choice(np.arange(points.shape[1]), k, replace=False)
    return points[:, mask]


def cluster_points(points, selected_points, i):  # returns an array containing the i'th cluster
    result = np.zeros((points.shape[0], 0))
    for j in range(points.shape[1]):
        m = np.sum(np.square(points[:, j] - selected_points[:, i]))
        is_in_cluster = True
        for x in range(selected_points.shape[1]):
            n = np.sum(np.square(points[:, j] - selected_points[:, x]))
            if n < m:
                is_in_cluster = False
                break
        if is_in_cluster:
            result = np.c_[result, points[:, j]]

    return result


def get_mean_point(cluster):
    mean = np.average(cluster, axis=1)
    return mean


def k_means(points, k, threshold=0):
    selected_points = get_random_centers(points, k)
    diff = float('inf')
    while diff > threshold:
        mean_points = np.zeros((points.shape[0], k))
        for i in range(k):
            cluster_i = cluster_points(points, selected_points, i)
            mean_i = get_mean_point(cluster_i)
            mean_points[:, i] = mean_i
        diff = np.sum(np.square(selected_points - mean_points))
        selected_points = mean_points.copy()
    return mean_points


def draw_fig(points, cluster_centers, k, colors, output_path):
    for i in range(k):
        cluster_i = cluster_points(points, cluster_centers, i)
        for j in range(cluster_i.shape[1]):
            plt.plot(cluster_i[0, j], cluster_i[1, j], marker='.', color=colors[i])

    plt.savefig(output_path)
    plt.show()


''' ##### functions for points in polar coordinates #### '''

def get_polar_coordinates(points):
    polar_points = np.zeros(points.shape)
    for i in range(points.shape[1]):
        x, y = points[0, i], points[1, i]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = math.atan2(x, y)
        polar_points[:, i] = (r, theta)
    return polar_points


def cluster_points_polar(points, selected_points, i):
    result = np.zeros((points.shape[0], 0))
    for j in range(points.shape[1]):
        m = np.sum(np.square(points[0, j] - selected_points[0, i]))
        is_in_cluster = True
        for x in range(selected_points.shape[1]):
            n = np.sum(np.square(points[0, j] - selected_points[0, x]))
            if n < m:
                is_in_cluster = False
                break
        if is_in_cluster:
            result = np.c_[result, points[:, j]]

    return result


def k_means_polar(points, k, threshold=0):
    selected_points = get_random_centers(points, k)
    diff = float('inf')
    while diff > threshold:
        mean_points = np.zeros((points.shape[0], k))
        for i in range(k):
            cluster_i = cluster_points_polar(points, selected_points, i)
            mean_i = get_mean_point(cluster_i)
            mean_points[:, i] = mean_i
        diff = np.sum(np.square(selected_points[0:] - mean_points[0:]))
        selected_points = mean_points.copy()
    return mean_points

def draw_polar_fig(polar_points, polar_centers, k, colors, output_path):
    for i in range(k):
        cluster_i = cluster_points_polar(polar_points, polar_centers, i)
        for j in range(cluster_i.shape[1]):
            r, theta = cluster_i[0, j], cluster_i[1, j]
            x, y = r * np.sin(theta), r * np.cos(theta)
            plt.plot(x, y, marker='.', color=colors[i])

    plt.savefig(output_path)
    plt.show()
