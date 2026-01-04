from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter
import math

def convert_to_grayscale(rgb_arr:np.ndarray) -> np.ndarray:
    weights = np.array([0.2989, 0.5870, 0.1140])
    grayscale_array = np.tensordot(rgb_arr, weights, axes=([-1], [0]))

    grayscale_array = grayscale_array.astype(np.float32)

    grayscale_rgb = np.stack((grayscale_array,) * 3, axis=-1)
    final_arr = grayscale_rgb[:, :, 0]
    final_arr = final_arr[..., np.newaxis]

    return final_arr.astype(np.uint8)

def mean_blur_image(pixel_array:np.ndarray, n:int) -> np.ndarray:
    kernel = np.ones(shape=(n, n)) / (n**2)
    blurred_arr = convolve(pixel_array, kernel, mode='reflect')

    return blurred_arr.astype(np.uint8)

def gaussian_blur_image(pixel_array:np.ndarray, sigma:int) -> np.ndarray:
    return gaussian_filter(pixel_array, sigma=(sigma, sigma)).astype(np.uint8)

def change_brightness(rgb_array:np.ndarray, n=0, k=1) -> np.ndarray:
    #NOTE: To change brightness, change the value of n and keep k at 1
    #To change exposure, change k and keep n at 0
    brightness_changed_img = rgb_array.astype(np.int16)*k + n
    brightness_changed_img = np.clip(brightness_changed_img, 0, 255)
    return brightness_changed_img.astype(np.uint8)

def crop_image(grayscale_array:np.ndarray) -> np.ndarray:
    grayscale_array = np.squeeze(grayscale_array, axis=2)

    #Use Haar cascades to find the boundary box for a face
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        grayscale_array,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    padding = 0.2 #20% padding, or 20% more width and height than boundary box shows

    x_pad = int(w * padding)
    y_pad = int(h * padding)

    x1 = max(0, x - x_pad)
    y1 = max(0, y - y_pad)
    x2 = min(grayscale_array.shape[1], x + w + x_pad)
    y2 = min(grayscale_array.shape[0], y + h + y_pad)

    face_crop = grayscale_array[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (128, 128)) #Resize cropped image to easier and smaller format

    return face_resized.astype(np.uint8)

def flip_horizontal(pixel_array:np.ndarray) -> np.ndarray:
    return np.flip(pixel_array, 1).astype(np.uint8)

def change_contrast_grayscale(grayscale_array:np.ndarray, alpha:float) -> np.ndarray:
    grayscale_array = grayscale_array.astype(np.float32)

    mean_val = np.mean(grayscale_array)
    new_contrast_array = alpha * (grayscale_array - mean_val) + mean_val
    new_contrast_array = np.clip(new_contrast_array, 0, 255)

    return new_contrast_array.astype(np.uint8)

def gaussian_noise(grayscale_array:np.ndarray, sigma:float, alpha:float) -> np.ndarray:
    noise = np.random.normal(0, sigma, grayscale_array.shape)
    noisy_grayscale_array = grayscale_array + alpha*noise
    noisy_grayscale_array = np.clip(noisy_grayscale_array, 0, 255)

    return noisy_grayscale_array.astype(np.uint8)

def rotate_image(grayscale_array:np.ndarray, angle:float) -> np.ndarray:
    #rotated_array = rotate(grayscale_array, angle, reshape=False) code that rotated with black corners
    h, w = grayscale_array.shape
    center = (w / 2, h / 2)

    theta = math.radians(angle)

    abs_cos = abs(math.cos(theta))
    abs_sin = abs(math.sin(theta))
    scale = min(w / (w * abs_cos + h * abs_sin),
                h / (h * abs_cos + w * abs_sin))

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated_2d = cv2.warpAffine(grayscale_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    rotated_array = rotated_2d[:, :, np.newaxis]

    return rotated_array.astype(np.uint8)


"""XXX: Use this code to check format of the image (PNG, JPG, etc.), size, and mode (RGB, GBR, etc.)
print(img.format)
print(img.size)
print(img.mode)
"""
"""XXX: Use this code to save a grayscale numpy array of pixels to an image
grayscale_array = convert_to_grayscale(rgb_array)
cv2.imwrite('test_grayscale.jpg', grayscale_array)
"""
"""XXX: Use this code to resize an *RGB* image and save it to a file
resized_img_array = Image.fromarray(resize_image(NEW_HEIGHT, NEW_WIDTH, rgb_array))
resized_img_array.save("test_resized_img.jpg")
"""
"""XXX: Use this code to resize an image AND convert it to grayscale and save it to file
grayscale_arr = convert_to_grayscale(rgb_array)
resized_grayscale = resize_image(NEW_HEIGHT, NEW_WIDTH, grayscale_arr)
cv2.imwrite("test_resized_grayscale_img.jpg", resized_grayscale)
"""
"""XXX: Use this code to blur an img
blurred_arr = *blur type*_blur_image(n (blur level), rgb_array)
blurred_img = Image.fromarray(blurred_arr)
blurred_img.save("filename.jpg")
"""
"""XXX: To change brightness/exposure of an img
brightness_changed_img = Image.fromarray(change_brightness(rgb_array, n, k), mode="RGB")
brightness_changed_img.save("filename.jpg")
"""
"""XXX: To crop an image down to format (128, 128, 1)
cropped_img = Image.fromarray(crop_image(convert_to_grayscale(rgb_array).astype(np.uint8)))
cropped_img.save("AdamSandler_Cropped.jpg")
"""
"""XXX: Very simple code to horizontally reflect/flip the image
flipped_img = flip_horizontal(rgb_array)
Image.fromarray(flipped_img).save("edited_images/AdamSandler_HorizontallyFlipped.jpg")
"""
"""XXX: Use this code to change the contrast of a GRAYSCALE image
contrasted_img_array = change_contrast_grayscale(convert_to_grayscale(rgb_array), alpha)
cv2.imwrite('edited_images/AdamSandler_ChangedContrast.jpg', contrasted_img_array)
"""
"""XXX: Code to increase or decrease the noise of an image
noisy_grayscale_array = gaussian_noise(convert_to_grayscale(rgb_array), sigma, alpha)
cv2.imwrite("edited_images/AdamSandler_Noise.jpg", noisy_grayscale_array)
"""
"""XXX: Code to rotate and scale an image (needs to be original, full size image) without black spots
rotated_array = rotate_image(convert_to_grayscale(rgb_array), 5)
cv2.imwrite("edited_images/AdamSandler_Rotated5.jpg", rotated_array)
"""