import cv2
from ultralytics import YOLO

# 1. open the video file
video_path = r"Technical Assessment\highway_cars.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error: Video file could not be read. Verify file path"

# grab video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# setup the output video file writer
video_writer = cv2.VideoWriter("output_car_counter.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 2. initialize the detector model
model = YOLO("yolov8n.pt")

# 3. memory structure
counted_ids = set()

# 4. main frame processing loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # track objects (2=car, 5=bus, 7=truck)
    results = model.track(frame, persist=True, classes=[2, 5, 7], verbose=False)

    # calculate detection zone limits
    current_h, current_w = frame.shape[:2]
    zone_top = int(current_h * 0.45)
    zone_bottom = int(current_h * 0.65)

    # hitbox over the highway lanes
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, zone_top), (current_w, zone_bottom), (255, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # extract target tracks if any vehicles are present
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # find the center vertical point of the vehicle bounding box
            cy = int((y1 + y2) / 2)

            # tracking boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicle ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # check if vehicle's center point sits inside target zone
            if zone_top <= cy <= zone_bottom:
                counted_ids.add(track_id)

    # 5. render the dashboard UI overlay
    total_vehicles = len(counted_ids)
    cv2.rectangle(frame, (15, 15), (280, 75), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Cars: {total_vehicles}", (30, 52), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # save processed frame
    video_writer.write(frame)

# 6. release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# final output summary
print("EVALUATION METRICS:")
print(f"Total Cars Counted: {total_vehicles}")