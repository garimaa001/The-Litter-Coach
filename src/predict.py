print("Script started!")
from ultralytics import YOLO

# Path to your trained model weights
# IMPORTANT: Double-check that 'litter_coach_detection_v13' is the correct folder name.
# It might be 'train' or 'train1', 'train2' etc. Look in your runs/detect/ folder.
model_path = "C:\\Users\\Garima Bisht\\OneDrive\\Desktop\\The_Litter_Coach\\runs\\detect\\litter_coach_detection_v13\\weights\\best.pt"

# Path to your images for inference (can be a single image or a folder)
# IMPORTANT: Ensure 'test_inference_images' is the correct folder name.
source_path = "C:\\Users\\Garima Bisht\\OneDrive\\Desktop\\The_Litter_Coach\\test_inference_images\\"

# Load the trained model
model = YOLO(model_path)

# Perform inference
results = model.predict(source=source_path, save=True, conf=0.25)

print("Inference completed! Check the 'runs/detect/predict/' folder for results.")