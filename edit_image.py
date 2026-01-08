from image_filters import *
import random as r

function_mapping = {
    1: convert_to_grayscale,
    2: mean_blur_image,
    3: gaussian_blur_image,
    4: change_brightness,
    5: crop_image,
    6: flip_horizontal,
    7: change_contrast_grayscale,
    8: gaussian_noise,
    9: rotate_image
}

image = Image.open("test_images/JustinBaratta3.jpg")
print(image.format)
print(image.size)
print(image.mode)

with open("operations.txt") as file:
    operations = file.readlines()

for operation in operations:
    pixel_array = np.asarray(image)
    operation = operation.strip()

    for char in operation:
        match char:
            case "1":
                pixel_array = convert_to_grayscale(pixel_array)
            case "2":
                pixel_array = mean_blur_image(pixel_array, n=r.choice([num for num in range(2, 6)]))
            case "3":
                pixel_array = gaussian_blur_image(pixel_array, sigma=r.choice([num for num in range(2, 3)]))
            case "4":
                n = r.choice([num*5 for num in range(-5, 5)])
                k = r.uniform(0.7, 2)
                type_of_change = r.choice([0, 1])
                if type_of_change == 0:
                    pixel_array = change_brightness(pixel_array, n=n, k=1)
                else:
                    pixel_array = change_brightness(pixel_array, n=0, k=k)
            case "5":
                pixel_array = crop_image(pixel_array)
            case "6":
                pixel_array = flip_horizontal(pixel_array)
            case "7":
                pixel_array = change_contrast_grayscale(pixel_array, alpha=r.uniform(0, 2.5))
            case "8":
                pixel_array = gaussian_noise(pixel_array, sigma=r.randint(2, 5), alpha=r.uniform(0, 2))
            case "9":
                pixel_array = rotate_image(pixel_array, angle=r.uniform(-8, 8))

    cv2.imwrite(f"edited_images/JustinBaratta/JustinBaratta3_{operation}.jpg", pixel_array)