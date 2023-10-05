import random
import time
import numpy as np
import cv2
from PIL import Image

def NumberOfNeighborsMask(number_of_neighbors_mask):
    for i in range(number_of_neighbors_mask.shape[0]):
        for j in range(number_of_neighbors_mask.shape[1]):
            if number_of_neighbors_mask[i, j] == -1:
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if 0 <= k < number_of_neighbors_mask.shape[0] and 0 <= l < number_of_neighbors_mask.shape[1]:
                            if number_of_neighbors_mask[k, l] != -1:
                                number_of_neighbors_mask[k, l] += 1
    return number_of_neighbors_mask

def updateNumberOfNeighborsMask(number_of_neighbors_mask, pixel):
    for i in range(pixel[0] - 1, pixel[0] + 2):
        for j in range(pixel[1] - 1, pixel[1] + 2):
            if 0 <= i < texture_size and 0 <= j < texture_size:
                if number_of_neighbors_mask[i, j] != -1:
                    number_of_neighbors_mask[i, j] += 1
    number_of_neighbors_mask[pixel[0], pixel[1]] = -1
    return number_of_neighbors_mask

def GetPixelWithMostColoredNeighbors(number_of_neighbors_mask):
    max_neighbors = 0
    max_neighbors_pixel = (-1, -1)
    for i in range(number_of_neighbors_mask.shape[0]):
        for j in range(number_of_neighbors_mask.shape[1]):
            if number_of_neighbors_mask[i, j] > max_neighbors:
                max_neighbors = number_of_neighbors_mask[i, j]
                max_neighbors_pixel = (i, j)
    return max_neighbors_pixel


def GetNeighborhoodWindow(pixel, image, half):
    y_start, y_end = max(pixel[0] - half, 0), min(pixel[0] + half, image.shape[0])
    x_start, x_end = max(pixel[1] - half, 0), min(pixel[1] + half, image.shape[1])
    window = image[y_start:y_end, x_start:x_end, :]
    return window

def SSD(template, sample):
    return np.sum((template - sample) ** 2)

def SSD_Gaussian(template, sample, mask):
    sigma = template.shape[0] / 6.4
    difference = template - sample
    extended_mask = mask[:, :, np.newaxis]
    ssd = np.sum(extended_mask * (difference ** 2) * np.exp(-difference ** 2 / (2 * sigma ** 2)))
    return ssd


def FindMatches(template, sample_image, half_patch_size, epsilon):
    y_max, x_max, _ = sample_image.shape

    pixel_list = []
    min_ssd = float('inf')

    for y in range(half_patch_size, y_max - half_patch_size):
        for x in range(half_patch_size, x_max - half_patch_size):
            sample_patch = sample_image[y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size, :]
            ssd_val = SSD(template, sample_patch)
            if ssd_val < min_ssd:
                min_ssd = ssd_val

    threshold = min_ssd * (1 + epsilon)

    for y in range(half_patch_size, y_max - half_patch_size):
        for x in range(half_patch_size, x_max - half_patch_size):
            sample_patch = sample_image[y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size, :]
            if SSD(template, sample_patch) <= threshold:
                pixel_list.append((y, x))

    return pixel_list


def efrosLeung(sample_image, texture_size, seed_size, half_patch_size, epsilon):
    half_seed_size = seed_size // 2
    seed_x = np.random.randint(0, sample_image.shape[1] - half_seed_size)
    seed_y = np.random.randint(0, sample_image.shape[0] - half_seed_size)

    texture_center_x = texture_size // 2
    texture_center_y = texture_size // 2
    generated_texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    generated_texture[texture_center_y - half_seed_size:texture_center_y + half_seed_size,
                      texture_center_x - half_seed_size:texture_center_x + half_seed_size,
                      :] = sample_image[seed_y - half_patch_size: seed_y + half_seed_size, seed_x - half_patch_size: seed_x + half_seed_size, :]

    number_of_neighbors_mask = np.zeros((texture_size, texture_size))
    number_of_neighbors_mask[texture_center_y - half_seed_size:texture_center_y + half_seed_size,
                        texture_center_x - half_seed_size:texture_center_x + half_seed_size] = -1

    number_of_neighbors_mask = NumberOfNeighborsMask(number_of_neighbors_mask)

    cv2.imshow('Generated Texture', generated_texture)
    cv2.imshow('Number of Neighbors Mask', number_of_neighbors_mask)
    cv2.waitKey(1)
    total_pixels = generated_texture.shape[0] * generated_texture.shape[1] * generated_texture.shape[2]

    while (number_of_neighbors_mask != -1).any():
        pixel = GetPixelWithMostColoredNeighbors(number_of_neighbors_mask)
        if pixel[0] < half_patch_size or pixel[0] > texture_size - half_patch_size or pixel[1] < half_patch_size or pixel[1] > texture_size - half_patch_size:
            number_of_neighbors_mask = updateNumberOfNeighborsMask(number_of_neighbors_mask, pixel)
            continue
        neighborhood_window = GetNeighborhoodWindow(pixel, generated_texture, half_patch_size)
        matches = FindMatches(neighborhood_window, sample_image, half_patch_size, epsilon)
        if matches == []:
            print("No match found for pixel ({}, {})".format(pixel[0], pixel[1]))
            break
        match = random.choice(matches)
        generated_texture[pixel[0], pixel[1], :] = sample_image[match[0], match[1], :]
        number_of_neighbors_mask = updateNumberOfNeighborsMask(number_of_neighbors_mask, pixel)

        cv2.imshow('Generated Texture', generated_texture)
        cv2.imshow('Number of Neighbors Mask', number_of_neighbors_mask)
        cv2.waitKey(1)  # Update display and wait for 1ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        colored_pixels = np.count_nonzero(generated_texture)
        #print("Percentage: {:.3%}".format(colored_pixels / total_pixels))

    return generated_texture

if __name__ == '__main__':
    start = time.time()
    image_number = 1
    sample_image = Image.open("textures_data/text{}.png".format(image_number))
    np_sample_img = np.array(sample_image)
    texture_size = 64
    seed_size = 24
    epsilon = 0.0
    half_patch_size = 12
    np_texture = efrosLeung(np_sample_img, texture_size, seed_size, half_patch_size, epsilon)
    texture = Image.fromarray(np_texture)
    texture.save("generated_images/text{}_size{}_seed{}_patch{}_epsilon{}.png".format(image_number,texture_size, seed_size, half_patch_size, epsilon))
    texture.show()
    end = time.time()
    print("Time: {:.2f} seconds".format(end - start))
