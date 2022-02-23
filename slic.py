import numpy as np
import cv2
from scipy.signal import medfilt


def get_feature_vectors(img):
    lab_vectors = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    xy_vectors = np.zeros((img.shape[0], img.shape[1], 2))
    for i in range(img.shape[0]):
        xy_vectors[i, :, 0] = i
    for j in range(img.shape[1]):
        xy_vectors[:, j, 1] = j

    return lab_vectors, xy_vectors


def get_img_gradients(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_derivative = img_gray - np.roll(img_gray, 1, axis=0)
    y_derivative = img_gray - np.roll(img_gray, 1, axis=1)
    result = -np.sqrt(np.square(x_derivative) + np.square(y_derivative))
    return result


def perturb_centers(img, img_grads, centers, perturb_radius):
    result = centers.copy()
    for i in range(centers.shape[0]):
        for j in range(centers.shape[1]):
            x, y = centers[i, j, 0], centers[i, j, 1]
            area = img_grads[x - perturb_radius: x + perturb_radius, y - perturb_radius: y + perturb_radius]
            m = np.min(area)
            mask = np.where(area == m)
            result[i, j, 0], result[i, j, 1] = mask[0][0] + x - perturb_radius, mask[1][0] + y - perturb_radius
    return result


def initialize_cluster_centers(img, img_grads, num_of_centers, perturb_radius):
    h, w = img.shape[0], img.shape[1]
    s = int(np.sqrt(h * w / num_of_centers) / 2)
    m, n = int(np.round(h / (2 * s))), int(np.round(w / (2 * s)))
    centers = np.zeros((m, n, 2), dtype='int')
    for i in range(m):
        for j in range(n):
            centers[i, j, 0] = int(s * (2 * i + 1))
            centers[i, j, 1] = int(s * (2 * j + 1))

    return perturb_centers(img, img_grads, centers, perturb_radius)


def find_closest_center(i, j, centers, r, s, lab_vectors, xy_vectors, alpha):
    m, n = centers.shape[0], centers.shape[1]
    if r == 0 and s == 0:
        return r, s
    if r == 0 and s == n:
        return r, s - 1
    if r == m and s == 0:
        return r - 1, s
    if r == m and s == n:
        # print(r, s)
        return r - 1, s - 1

    d_lab = np.ones((3, 4))
    d_xy = np.ones((2, 4))
    d_lab_ij = lab_vectors[i, j, :].reshape((3, 1))
    d_xy_ij = xy_vectors[i, j].reshape((2, 1))
    if r == 0:
        result_centers = np.asarray([[r, s], [r, s - 1]])
        d_lab[:, ::2] = d_lab[:, ::2] * lab_vectors[centers[r, s, 0], centers[r, s, 1]].reshape((3, 1))
        d_lab[:, 1::2] = d_lab[:, 1::2] * lab_vectors[centers[r, s - 1, 0], centers[r, s - 1, 1]].reshape((3, 1))
        d_xy[:, ::2] = d_xy[:, ::2] * xy_vectors[centers[r, s, 0], centers[r, s, 1]].reshape((2, 1))
        d_xy[:, 1::2] = d_xy[:, 1::2] * xy_vectors[centers[r, s - 1, 0], centers[r, s - 1, 1]].reshape((2, 1))
        d1 = np.sum(np.square(d_lab - d_lab_ij), axis=0)
        d2 = np.multiply(np.sum(np.square(d_xy - d_xy_ij), axis=0), alpha)
        d = np.add(d1, d2)
        t = np.argmin(d)
        return result_centers[t]
    if r == m:
        result_centers = np.asarray([[r - 1, s], [r - 1, s - 1]])
        d_lab[:, ::2] = d_lab[:, ::2] * lab_vectors[centers[r - 1, s, 0], centers[r - 1, s, 1]].reshape((3, 1))
        d_lab[:, 1::2] = d_lab[:, 1::2] * lab_vectors[centers[r - 1, s - 1, 0], centers[r - 1, s - 1, 1]].reshape(
            (3, 1))
        d_xy[:, ::2] = d_xy[:, ::2] * xy_vectors[centers[r - 1, s, 0], centers[r - 1, s, 1]].reshape((2, 1))
        d_xy[:, 1::2] = d_xy[:, 1::2] * xy_vectors[centers[r - 1, s - 1, 0], centers[r - 1, s - 1, 1]].reshape((2, 1))
        d1 = np.sum(np.square(d_lab - d_lab_ij), axis=0)
        d2 = np.multiply(np.sum(np.square(d_xy - d_xy_ij), axis=0), alpha)
        d = np.add(d1, d2)
        t = np.argmin(d)
        return result_centers[t]
    if s == 0:
        result_centers = np.asarray([[r, s], [r - 1, s]])
        d_lab[:, ::2] = d_lab[:, ::2] * lab_vectors[centers[r, s, 0], centers[r, s, 1]].reshape((3, 1))
        d_lab[:, 1::2] = d_lab[:, 1::2] * lab_vectors[centers[r - 1, s, 0], centers[r - 1, s, 1]].reshape((3, 1))
        d_xy[:, ::2] = d_xy[:, ::2] * xy_vectors[centers[r, s, 0], centers[r, s, 1]].reshape((2, 1))
        d_xy[:, 1::2] = d_xy[:, 1::2] * xy_vectors[centers[r - 1, s, 0], centers[r - 1, s, 1]].reshape((2, 1))
        d1 = np.sum(np.square(d_lab - d_lab_ij), axis=0)
        d2 = np.multiply(np.sum(np.square(d_xy - d_xy_ij), axis=0), alpha)
        d = np.add(d1, d2)
        t = np.argmin(d)
        return result_centers[t]

    if s == n:
        result_centers = np.asarray([[r, s - 1], [r - 1, s - 1]])
        d_lab[:, ::2] = d_lab[:, ::2] * lab_vectors[centers[r, s - 1, 0], centers[r, s - 1, 1]].reshape((3, 1))
        d_lab[:, 1::2] = d_lab[:, 1::2] * lab_vectors[centers[r - 1, s - 1, 0], centers[r - 1, s - 1, 1]].reshape(
            (3, 1))
        d_xy[:, ::2] = d_xy[:, ::2] * xy_vectors[centers[r, s - 1, 0], centers[r, s - 1, 1]].reshape((2, 1))
        d_xy[:, 1::2] = d_xy[:, 1::2] * xy_vectors[centers[r - 1, s - 1, 0], centers[r - 1, s - 1, 1]].reshape((2, 1))
        d1 = np.sum(np.square(d_lab - d_lab_ij), axis=0)
        d2 = np.multiply(np.sum(np.square(d_xy - d_xy_ij), axis=0), alpha)
        d = np.add(d1, d2)
        t = np.argmin(d)
        return result_centers[t]
    result_centers = np.asarray([[r - 1, s - 1], [r - 1, s], [r, s], [r, s - 1]])
    d_lab[:, 0] = lab_vectors[centers[r - 1, s - 1, 0], centers[r - 1, s - 1, 1]]
    d_lab[:, 1] = lab_vectors[centers[r - 1, s, 0], centers[r - 1, s, 1]]
    d_lab[:, 2] = lab_vectors[centers[r, s, 0], centers[r, s, 1]]
    d_lab[:, 3] = lab_vectors[centers[r, s - 1, 0], centers[r, s - 1, 1]]
    d_xy[:, 0] = xy_vectors[centers[r - 1, s - 1, 0], centers[r - 1, s - 1, 1]]
    d_xy[:, 1] = xy_vectors[centers[r - 1, s, 0], centers[r - 1, s, 1]]
    d_xy[:, 2] = xy_vectors[centers[r, s, 0], centers[r, s, 1]]
    d_xy[:, 3] = xy_vectors[centers[r, s - 1, 0], centers[r, s - 1, 1]]
    d1 = np.sum(np.square(d_lab - d_lab_ij), axis=0)
    d2 = np.multiply(np.sum(np.square(d_xy - d_xy_ij), axis=0), alpha)
    d = np.add(d1, d2)
    t = np.argmin(d)
    return result_centers[t]

def cluster_pixels(img, centers, lab_vectors, xy_vectors, alpha, num_of_iterations=7):
    h, w = img.shape[0], img.shape[1]
    labels = np.zeros((h, w), dtype='int')
    a1, b1 = centers.shape[0], centers.shape[1]
    for t in range(num_of_iterations):
        r, s = 0, 0
        for i in range(h):
            for j in range(w):
                if s < centers.shape[1] and j > centers[
                    min(r, centers.shape[0] - 1), s, 1]:  # update close cluster point
                    s += 1
                a2, b2 = find_closest_center(i, j, centers, r, s, lab_vectors, xy_vectors, alpha)
                labels[i, j] = b1 * a2 + b2

            s = 0
            if r < centers.shape[0] and i > centers[r, s, 0]:
                r += 1
        centers2 = update_centers(labels, centers)
        threshold = np.sqrt(np.sum(np.square(centers2 - centers)))
        print(threshold)
        centers = centers2
    # plt.imshow(labels, cmap='gray')
    # plt.show()

    return labels


def update_centers(labels, centers):
    result_centers = np.zeros(centers.shape, dtype='int')
    t = 0
    for i in range(centers.shape[0]):
        for j in range(centers.shape[1]):
            mask = np.where(labels == t)
            if mask[0].size > 0:
                center = [int(np.average(mask[0])), int(np.average(mask[1]))]
                result_centers[i, j, 0] = center[0]
                result_centers[i, j, 1] = center[1]
            else:
                result_centers[i, j, 0] = centers[i, j, 0]
                result_centers[i, j, 1] = centers[i, j, 1]
            t += 1
    result_centers = np.asarray(result_centers)
    return result_centers


def remove_noise(labels, num_of_iterations):  # applies median filter to remove noise
    result = labels.copy().astype('int16')
    for i in range(num_of_iterations):
        result = medfilt(result, 3)
    return result


def generate_result_img(img, labels):
    x_segments = np.where((labels - np.roll(labels, 1, axis=0)) != 0)
    y_segments = np.where((labels - np.roll(labels, 1, axis=1)) != 0)
    result = img.copy()
    result[x_segments] = 0
    result[y_segments] = 0
    return result
