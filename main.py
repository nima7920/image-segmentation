import k_means
import matplotlib.pyplot as plt
import numpy as np
import cv2

''' k-means method '''
input_path = "inputs/Points.txt"
output_paths = ["outputs/res01.jpg", "outputs/res02.jpg", "outputs/res03.jpg", "outputs/res04.jpg"]

''' showing initial points '''

n, points = k_means.readPoints("inputs/Points.txt")
print(points)
for i in range(points[0].size):
    plt.plot(points[0, i], points[1, i], marker='.', color='blue')
plt.savefig(output_paths[0])
plt.show()

k = 2
colors = ["red", "green"]
for i in range(1, 3):
    selected_points = k_means.get_random_centers(points, k)
    centers = k_means.k_means(points, k)
    k_means.draw_fig(points, centers, k, colors, output_paths[i])

''' clustering in polar coordinates '''
polar_points = k_means.get_polar_coordinates(points)
polar_centers = k_means.k_means_polar(polar_points, k)
k_means.draw_polar_fig(polar_points, polar_centers, k, colors, output_paths[3])

''' SLIC method '''
import slic

input_path = "inputs/slic.jpg"
output_path = "outputs/slic-result.jpg"
img = cv2.imread(input_path)

shape = (img.shape[1], img.shape[0])
img_resized = cv2.resize(img, (1008, 776))
lab_vectors, xy_vectors = slic.get_feature_vectors(img_resized)
img_grads = slic.get_img_gradients(img_resized)

centers = slic.initialize_cluster_centers(img_resized, img_grads, 256, 5)
labels = slic.cluster_pixels(img_resized, centers, lab_vectors, xy_vectors, 0.5)
labels = slic.remove_noise(labels, 3)

labels = cv2.resize(labels, shape)
result = slic.generate_result_img(img, labels)
cv2.imwrite(output_path, result)
