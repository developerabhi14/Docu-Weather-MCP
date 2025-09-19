# import base64

# with open("image.jpg", "rb") as imageFile:
#     data = base64.b64encode(imageFile.read())
#     print(data)


import torch
from PIL import Image

# Load YOLOv5 model (downloads automatically if not present)
# You can use yolov5s, yolov5m, yolov5l, yolov5x depending on accuracy/speed tradeoff
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path):
    # Open image
    img = Image.open(image_path)

    # Inference
    results = model(img)

    # Extract detected class names
    detected_objects = results.pandas().xyxy[0]['name'].tolist()

    # Remove duplicates
    unique_objects = list(set(detected_objects))

    return unique_objects

if __name__ == "__main__":
    # Ask user for image path
    image_path = input("Enter the path to the image: ").strip()

    objects = detect_objects(image_path)

    print("\nObjects detected in the image:")
    for obj in objects:
        print("-", obj)
