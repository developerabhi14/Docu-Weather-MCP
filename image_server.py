from mcp.server.fastmcp import FastMCP
from PIL import Image
import torch


app=FastMCP('ImageServer')

# load yolo model  
model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)



@app.tool()
def analyze_image(image_path: str):
    """Detects objects in the image and returns a list of item names.
    Input: path to an image file (string)
    Output: list of strings (object names)
    """
    print("==========analyze image tool called==========")
    img = Image.open(image_path)
    # Inference
    results = model(img)
    # Extract detected class names
    detected_objects = results.pandas().xyxy[0]['name'].tolist()
    # Remove duplicates
    unique_objects = list(set(detected_objects))

    return unique_objects


if __name__ == "__main__":
    app.run(transport="stdio")