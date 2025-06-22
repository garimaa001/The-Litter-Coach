import cv2
from ultralytics import YOLO

# --- Configuration ---
model_path = "C:\\Users\\Garima Bisht\\OneDrive\\Desktop\\The_Litter_Coach\\runs\\detect\\litter_coach_detection_v13\\weights\\best.pt"
conf_threshold = 0.50

# --- Biodegradability Mapping ---
# Make sure these class names exactly match the names in your model.names
biodegradability_map = {
    'banana_peel': 'Biodegradable',
    'plastic_bottle': 'Non-Biodegradable',
    'soda_can': 'Non-Biodegradable', # Ensure this is correct for your labels
    'notebook': 'Biodegradable',
    'glass': 'Non-Biodegradable',
    'apple': 'Biodegradable', # Assuming 'apple' is detected (whether fruit or peel)
    # Add all other classes detected by your model here
    'cardboard': 'Biodegradable', # Example
    'plastic_bag': 'Non-Biodegradable' # Example
}

# --- Load Model ---
try:
    model = YOLO(model_path)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please check if the model path is correct: {model_path}")
    exit()

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please ensure your webcam is connected and not in use by another application.")
    exit()

window_name = "The Litter Coach - Live Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Webcam opened. Starting real-time detection...")

# --- Real-time Detection Loop ---
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame, exiting...")
        break

    results = model(frame, stream=True, conf=conf_threshold)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # --- Get Biodegradability Status ---
            biodegradability_status = biodegradability_map.get(class_name, "Unknown")
            # Choose color based on biodegradability
            if biodegradability_status == 'Biodegradable':
                color = (0, 255, 0) # Green for biodegradable
            elif biodegradability_status == 'Non-Biodegradable':
                color = (0, 0, 255) # Red for non-biodegradable
            else:
                color = (255, 255, 0) # Yellow for unknown

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Update the label to include biodegradability
            label = f"{class_name} ({biodegradability_status}): {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" 'q' pressed, stopping detection.")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Detection stopped. Resources released.")