import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def k_means(data, k, max_iterations):
    cluster_centers = []
    clusters = []
    iterations = 0

    for i in range(k):
        cluster_centers.append(data[i])

    for i in range(k):
        clusters.append([])

    for i in range(len(data)):
        distances = []
        for j in range(k):
            distances.append(np.linalg.norm(data[i] - cluster_centers[j]))
        clusters[distances.index(min(distances))].append(data[i])

    for i in range(k):
        cluster_centers[i] = np.mean(clusters[i], axis=0)

    while True and iterations < max_iterations:
        changed = False
        for i in range(len(data)):
            distances = []
            for j in range(k):
                distances.append(np.linalg.norm(data[i] - cluster_centers[j]))
            clusters[distances.index(min(distances))].append(data[i])

        for i in range(k):
            if not np.array_equal(cluster_centers[i], np.mean(clusters[i], axis=0)):
                changed = True
            cluster_centers[i] = np.mean(clusters[i], axis=0)

        if not changed:
            break

        iterations += 1

    return cluster_centers, clusters

def train_test(filenames, patch_size):
    train_data = []
    test_data = []
    for i in range(len(filenames)):
        img_data = []
        sample_image = Image.open(filenames[i])
        np_sample_img = np.array(sample_image)
        for x in range(0, np_sample_img.shape[0] - patch_size, patch_size):
            for y in range(0, np_sample_img.shape[1] - patch_size, patch_size):
                patch = np_sample_img[x:x + patch_size, y:y + patch_size, :]
                img_data.append(patch)

        random.shuffle(img_data)
        half = int(len(img_data) / 2)
        train_data.append(img_data[:half])
        test_data.append(img_data[half:])

    return train_data, test_data


if __name__ == '__main__':

    filenames = ["Colored_Brodatz/D1_COLORED.tif",
                 "Colored_Brodatz/D2_COLORED.tif",
                 "Colored_Brodatz/D3_COLORED.tif",
                 "Colored_Brodatz/D4_COLORED.tif",
                 "Colored_Brodatz/D5_COLORED.tif"]

    train_data, test_data = train_test(filenames, 64)


    for i in range(len(train_data[0])):
        patch_img = Image.fromarray(train_data[0][i])
        patch_img.save("Patches/train{}_size64.png".format(i))

    for i in range(len(test_data[0])):
        patch_img = Image.fromarray(test_data[0][i])
        patch_img.save("Patches/test{}_size64.png".format(i))


    # Labeliser nos données (faire des tuples <np_array, num_image> ?)
    # Faire l'histogramme des gradient sur toute nos données (gradient en noir et blanc)
        # Sobel en x et y sur le gradient
        # Tableaau de n valeurs pour les orientations
        # Pour chaque pixels la valeur est donnée par arc tan (y/x)
    # Entrainer notre k_means avec les données de train
    # Pour chaque cluster determiner le label dominant
    # Classifier nos données de tests avec k_means et vérifier si les prédictions sont corrects

    # Générer des textures et les classifier avec k_means

    """

    # Kmean GMM2D 
    gmm2d = np.loadtxt("classif_data/gmm2d.asc")
    centers, clusters = k_means(gmm2d, 5, 100)
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

    fig.savefig("GMM2D_K-Means_k5.png")

    # Kmean GMM3D
    gmm3d = np.loadtxt("classif_data/gmm3d.asc")
    centers, clusters = k_means(gmm3d, 5, 100)

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
    fig.savefig("GMM3D_K-Means_k5.png")
    """

