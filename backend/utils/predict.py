import numpy as np

# Class labels
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_label(model, input_array: np.ndarray):
    prediction = model.predict(input_array)
    index = np.argmax(prediction)
    label = labels[index]
    has_tumor = label != 'notumor'
    return label, has_tumor

def predict_mask(model, input_array: np.ndarray):
    mask = model.predict(input_array)[0].squeeze()
    return mask
