print("Script started!")
from ultralytics import YOLO

# Path to trained model weights
model_path = "C:\\Users\\Garima Bisht\\OneDrive\\Desktop\\The_Litter_Coach\\runs\\detect\\litter_coach_detection_v13\\weights\\best.pt"

# Path to images for inference (can be a single image or a folder)
source_path = "C:\\Users\\Garima Bisht\\OneDrive\\Desktop\\The_Litter_Coach\\test_inference_images\\"

# Load the trained model
model = YOLO(model_path)


results = model.predict(source=source_path, save=True, conf=0.25)

print("Inference completed! Check the 'runs/detect/predict/' folder for results.")
