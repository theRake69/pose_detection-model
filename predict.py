import cv2
import mysql.connector
from datetime import datetime
from ultralytics import YOLO

input_file = 'Traffic police Hand Signals.mp4'

# Define a function to load labels from a file
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

# Define a function to save results to MySQL
def save_results_to_mysql(results, filename, frame_time, video_timestamp, db_config, label_file, no_detection=False):
    # Load labels
    labels = load_labels(label_file)
    
    try:
        # Establish a connection to MySQL
        with mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            ) as conn:
            
            with conn.cursor() as cursor:
                # If no detection, insert an entry with 'No detection' label
                if no_detection:
                    frame_datetime = frame_time.strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute(
                        "INSERT INTO predictions (filename, label, confidence, date, video_timestamp) VALUES (%s, %s, %s, %s, %s)",
                        (filename, 'No detection', 0, frame_datetime, video_timestamp)
                    )
                else:
                    # Insert prediction results into the MySQL table
                    for result in results:
                        if len(result.boxes) == 0:
                            # No detection case
                            frame_datetime = frame_time.strftime('%Y-%m-%d %H:%M:%S')
                            cursor.execute(
                                "INSERT INTO predictions (filename, label, confidence, date, video_timestamp) VALUES (%s, %s, %s, %s, %s)",
                                (filename, 'No detection', 0, frame_datetime, video_timestamp)
                            )
                        else:
                            for box in result.boxes:
                                class_label = box.cls.item()  # Convert tensor to float or int
                                confidence = box.conf.item()  # Convert tensor to float or int
                                
                                # Attempt to extract bounding box coordinates
                                try:
                                    # Map class_label to label
                                    label = labels[int(class_label)]
                                    
                                    # Format the frame time as datetime
                                    frame_datetime = frame_time.strftime('%Y-%m-%d %H:%M:%S')
                                    
                                    # Insert into MySQL table (filename, label, confidence, date, video_timestamp)
                                    cursor.execute(
                                        "INSERT INTO predictions (filename, label, confidence, date, video_timestamp) VALUES (%s, %s, %s, %s, %s)",
                                        (filename, label, confidence, frame_datetime, video_timestamp)
                                    )
                                
                                except Exception as e:
                                    print(f"Error processing box: {e}")
                                    continue
            
            # Commit the transaction
            conn.commit()
            print("Saved results to MySQL successfully.")
    
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")

# Define your database configuration
db_config = {
    'host': 'localhost',
    'port': 8889,
    'user': 'root',
    'password': 'root',
    'database': 'pose_detection'
}

# Define the path to your label file
label_file = 'labels.txt'

# Load your previously trained model
model = YOLO('runs/pose/train15/weights/best.pt')

# Open the video file
cap = cv2.VideoCapture(input_file)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Target FPS for processing
target_fps = 5

# Calculate frame interval based on target FPS
frame_interval = int(fps / target_fps) if fps > 0 else 1  # Calculate the interval to achieve 5 fps

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process frames at the specified interval
    if frame_count % frame_interval == 0:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model.predict(source=frame_rgb, imgsz=288, task='pose', conf=0.1)
        
        # Get the timestamp of the current frame
        frame_time = datetime.now()  # This can be adjusted to use the video timestamp if available
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Video timestamp in seconds
        
        # Save results to MySQL, including handling no detections
        if results and len(results) > 0:
            save_results_to_mysql(results, input_file, frame_time, video_timestamp, db_config, label_file)
        else:
            save_results_to_mysql([], input_file, frame_time, video_timestamp, db_config, label_file, no_detection=True)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
