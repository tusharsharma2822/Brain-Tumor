from PIL import Image
import numpy as np

def preprocess_for_detection(image: Image.Image) -> np.ndarray:
    image = image.resize((150, 150))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

def preprocess_for_segmentation(image: Image.Image) -> np.ndarray:
    image = image.convert('L')
    image = image.resize((128, 128))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array
