import cv2
from ultralytics import YOLO

# --- Configuration ---
model_path = "C:\\Users\\Garima Bisht\\OneDrive\\Desktop\\The_Litter_Coach\\runs\\detect\\litter_coach_detection_v13\\weights\\best.pt"
conf_threshold = 0.50

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

# --- Create a resizable window and maximize it ---
window_name = "The Litter Coach - Live Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create a resizable window
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Set to full screen

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

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(window_name, frame) # Display in the named window

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" 'q' pressed, stopping detection.")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Detection stopped. Resources released.")