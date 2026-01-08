from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import nn
from model import Model
from model import get_classes
import numpy as np
from image_filters import crop_image, convert_to_grayscale
import cv2

def predict_face(img):
    classes = get_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    img = Image.open(img)
    rgb_array = np.asarray(img)
    grayscale_cropped = crop_image(convert_to_grayscale(rgb_array))

    cv2.imwrite("image.jpg", grayscale_cropped)

    tensor = torch.from_numpy(grayscale_cropped).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        print(output)
        pred = torch.argmax(output, dim=1)
        pred = classes[pred.item()]

    return pred
