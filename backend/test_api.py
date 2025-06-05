import requests

# Replace with your own image path
image_path = r"C:\Users\tusha\Desktop\Brain-Tumor-Detection-Segmentation\Backend\archive\Testing\meningioma\Te-me_0013.jpg"
url = "http://127.0.0.1:5000/predict"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

# Print response
if response.ok:
    data = response.json()
    print("Tumor Type Detected:", data["label"])
    print("Tumor Detected:", data["tumor_detected"])
    
    if data["tumor_detected"]:
        print("Segmentation mask (128x128) received.")
        # Optional: Convert to numpy array and visualize if needed
        import numpy as np
        import matplotlib.pyplot as plt

        mask = np.array(data["segmentation"])
        plt.imshow(mask, cmap="gray")
        plt.title("Predicted Tumor Segmentation Mask")
        plt.axis("off")
        plt.show()
else:
    print("Request failed:", response.status_code, response.text)
