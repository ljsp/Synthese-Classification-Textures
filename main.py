import random
import time

import cv2
import numpy as np
from PIL import Image

def NumberOfNeighborsMask(number_of_neighbors_mask):
    start = time.time()
    for i in range(number_of_neighbors_mask.shape[0]):
        for j in range(number_of_neighbors_mask.shape[1]):
            if number_of_neighbors_mask[i, j] == -1:
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if 0 <= k < number_of_neighbors_mask.shape[0] and 0 <= l < number_of_neighbors_mask.shape[1]:
                            if number_of_neighbors_mask[k, l] != -1:
                                number_of_neighbors_mask[k, l] += 1

    end = time.time()
    #print("[INIT] Number of neighbor time: ", end - start)
    return number_of_neighbors_mask

def updateNumberOfNeighborsMask(number_of_neighbors_mask, pixel):
    #start = time.time()
    for i in range(pixel[0] - 1, pixel[0] + 2):
        for j in range(pixel[1] - 1, pixel[1] + 2):
            if 0 <= i < texture_size and 0 <= j < texture_size:
                if number_of_neighbors_mask[i, j] != -1:
                    number_of_neighbors_mask[i, j] += 1
    number_of_neighbors_mask[pixel[0], pixel[1]] = -1
    #end = time.time()
    #print("[UPDATE] Number of neighbor time: ", end - start)
    return number_of_neighbors_mask

def GetPixelWithMostColoredNeighbors(number_of_neighbors_mask):
    #start = time.time()
    max_neighbors = 0
    max_neighbors_pixel = (-1, -1)
    for i in range(number_of_neighbors_mask.shape[0]):
        for j in range(number_of_neighbors_mask.shape[1]):
            if number_of_neighbors_mask[i, j] > max_neighbors:
                max_neighbors = number_of_neighbors_mask[i, j]
                max_neighbors_pixel = (i, j)
    #end = time.time()
    #print("GET  Pixel with most neighbor time: ", end - start)
    return max_neighbors_pixel


def GetNeighborhoodWindow(pixel, image, half):
    y_start, y_end = max(pixel[0] - half, 0), min(pixel[0] + half + 1, image.shape[0])
    x_start, x_end = max(pixel[1] - half, 0), min(pixel[1] + half + 1, image.shape[1])

    window = image[y_start:y_end, x_start:x_end, :]

    mask = np.zeros((2 * half + 1, 2 * half + 1))
    mask[(half - (pixel[0] - y_start)):(half + (y_end - pixel[0])),
    (half - (pixel[1] - x_start)):(half + (x_end - pixel[1]))] = 1

    return window, mask


def SSD_Gaussian(template, sample, mask):
    sigma = 1
    difference = template - sample
    extended_mask = mask[:, :, np.newaxis]

    ssd = np.sum(extended_mask * (difference ** 2) * np.exp(-difference ** 2 / (2 * sigma ** 2)))
    return ssd


def FindMatches(template, sample_image, epsilon):
    h, w, _ = template.shape
    y_max, x_max, _ = sample_image.shape

    # Génération du masque
    mask = np.where(template[:, :, 0] != 0, 1, 0)

    pixel_list = []
    min_ssd = float('inf')

    for y in range(y_max - h):
        for x in range(x_max - w):
            sample_patch = sample_image[y:y + h, x:x + w, :]
            ssd_val = SSD_Gaussian(template, sample_patch, mask)
            if ssd_val < min_ssd:
                min_ssd = ssd_val

    threshold = min_ssd * (1 + epsilon)

    for y in range(y_max - h):
        for x in range(x_max - w):
            sample_patch = sample_image[y:y + h, x:x + w, :]
            if SSD_Gaussian(template, sample_patch, mask) <= threshold:
                pixel_list.append((y, x))

    return pixel_list


def efrosLeung(sample_image, texture_size, half_patch_size, epsilon):
    generated_texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    seed_x = np.random.randint(0, sample_image.shape[1] - half_patch_size * 2)
    seed_y = np.random.randint(0, sample_image.shape[0] - half_patch_size * 2)

    # Patch centered on the seed
    texture_center_x = texture_size // 2
    texture_center_y = texture_size // 2

    generated_texture[texture_center_y - half_patch_size:texture_center_y + half_patch_size,
                      texture_center_x - half_patch_size:texture_center_x + half_patch_size,
                      :] = sample_image[seed_y:seed_y + half_patch_size * 2, seed_x:seed_x + half_patch_size * 2, :]

    cv2.imwrite("generated_images/seed.png", generated_texture)

    # Set coordinates of the seed to -1
    number_of_neighbors_mask = np.zeros((texture_size, texture_size))
    number_of_neighbors_mask[texture_center_y - half_patch_size:texture_center_y + half_patch_size,
                        texture_center_x - half_patch_size:texture_center_x + half_patch_size] = -1

    number_of_neighbors_mask = NumberOfNeighborsMask(number_of_neighbors_mask)

    #cv2.imwrite("generated_images/number_of_neighbors_mask.png", number_of_neighbors_mask)

    while (number_of_neighbors_mask != -1).any():
        pixel = GetPixelWithMostColoredNeighbors(number_of_neighbors_mask)
        neighborhood_window, mask = GetNeighborhoodWindow(pixel, generated_texture, half_patch_size + 1)
        matches = FindMatches(neighborhood_window, sample_image, epsilon)
        if matches == []:
            print("No match found for pixel ({}, {})".format(pixel[0], pixel[1]))
            generated_texture[pixel[0], pixel[1], :] = (0, 0, 0)
            number_of_neighbors_mask = updateNumberOfNeighborsMask(number_of_neighbors_mask, pixel)
            continue
        match = random.choice(matches)
        generated_texture[pixel[0], pixel[1], :] = sample_image[match[0], match[1], :]
        number_of_neighbors_mask = updateNumberOfNeighborsMask(number_of_neighbors_mask, pixel)
        #Number of colored pixels
        colored_pixels = np.count_nonzero(generated_texture)
        total_pixels = generated_texture.shape[0] * generated_texture.shape[1] * generated_texture.shape[2]
        print("Percentage: {:.3%}".format(colored_pixels / total_pixels))

    return generated_texture

if __name__ == '__main__':
    image_number = 0
    sample_image = Image.open("textures_data/text{}.png".format(image_number))
    np_sample_img = np.array(sample_image)
    texture_size = 32
    seed_size = 20
    epsilon = 0.1
    half_patch_size = seed_size // 2
    np_texture = efrosLeung(np_sample_img, texture_size, half_patch_size, epsilon)
    texture = Image.fromarray(np_texture)
    texture.save("generated_images/text{}_size{}_seed{}_epsilon{}.png".format(image_number,texture_size, seed_size, epsilon))
    texture.show()
