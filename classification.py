import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def k_means(data, k, max_iterations):
    """
    :param data: list of numpy arrays
    :param k: number of clusters
    :param max_iterations: maximum number of iterations
    :return: list of cluster centers
    """
    cluster_centers = []
    clusters = []
    iterations = 0

    for i in range(k):
        cluster_centers.append(data[i])

    for i in range(k):
        clusters.append([])

    # assign each data point to the closest cluster center
    for i in range(len(data)):
        distances = []
        for j in range(k):
            distances.append(np.linalg.norm(data[i] - cluster_centers[j]))
        clusters[distances.index(min(distances))].append(data[i])

    # update cluster centers
    for i in range(k):
        cluster_centers[i] = np.mean(clusters[i], axis=0)

    # repeat until cluster centers do not change
    while True and iterations < max_iterations:
        changed = False
        # assign each data point to the closest cluster center
        for i in range(len(data)):
            distances = []
            for j in range(k):
                distances.append(np.linalg.norm(data[i] - cluster_centers[j]))
            clusters[distances.index(min(distances))].append(data[i])

        # update cluster centers
        for i in range(k):
            if not np.array_equal(cluster_centers[i], np.mean(clusters[i], axis=0)):
                changed = True
            cluster_centers[i] = np.mean(clusters[i], axis=0)

        if not changed:
            break

        iterations += 1

    return cluster_centers, clusters

def train_test(filename, patch_size):
    """
    :param filename: name of the image file
    :param patch_size: size of the patches
    :return: train and test data
    """
    sample_image = Image.open(filename)
    np_sample_img = np.array(sample_image)
    data = []

    for x in range(0, np_sample_img.shape[0] - patch_size, patch_size):
        for y in range(0, np_sample_img.shape[1] - patch_size, patch_size):
            patch = np_sample_img[x:x + patch_size, y:y + patch_size, :]
            data.append(patch)

    random.shuffle(data)
    half = int(len(data) / 2)
    train_data = data[:half]
    test_data = data[half:]

    """
    for i in range(len(train_data)):
        patch_img = Image.fromarray(train_data[i])
        patch_img.save("Patches/train{}_size64.png".format(i))

    for i in range(len(test_data)):
        patch_img = Image.fromarray(test_data[i])
        patch_img.save("Patches/test{}_size64.png".format(i))
    """

    return train_data, test_data


if __name__ == '__main__':
    #train_data, test_data = train_test("Colored_Brodatz/D1_COLORED.tif", 64)

    gmm2d = np.loadtxt("classif_data/gmm2d.asc")
    centers, clusters = k_means(gmm2d, 8, 100)
    cmap = get_cmap(len(clusters))

    fig = plt.figure()
    plt.title("GMM2D")
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(len(clusters)):
        x = []
        y = []
        for j in range(len(clusters[i])):
            x.append(clusters[i][j][0])
            y.append(clusters[i][j][1])
        plt.scatter(x, y, s=1, color=cmap(i))

    for i in range(len(centers)):
        plt.scatter(centers[i][0], centers[i][1], color='black', s=10)
    plt.show()

    #Save the image
    fig.savefig("GMM2D_K-Means.png")


    gmm3d = np.loadtxt("classif_data/gmm3d.asc")
    centers, clusters = k_means(gmm3d, 8, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("GMM3D")
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(len(clusters)):
        x = []
        y = []
        z = []
        for j in range(len(clusters[i])):
            x.append(clusters[i][j][0])
            y.append(clusters[i][j][1])
            z.append(clusters[i][j][2])
        ax.scatter(x, y, z, s=1)
    plt.show()
    fig.savefig("GMM3D_K-Means.png")


