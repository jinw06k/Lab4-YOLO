"""
Part F: Object detection and tracking with YOLO  (YOLO26n via ONNX)

    Uses a small YOLO model (yolo26n) to detect real objects 
    and send driving command for your robot.

    Run a pre-exported yolo26n.onnx file (which you
    git-pulled). No need to pip install ultralytics.
    
    Ultralytics YOLO26 Overview: https://docs.ultralytics.com/models/yolo26#overview

    EECS 473 Lab 4 -- YOLO version
"""

# import the necessary packages
import time
import cv2
import numpy as np
import onnxruntime as ort

# ===========================================================================
# TODO: Step 1 - What object should the robot look for?
# (2 tasks)
# ===========================================================================
TARGET = ""                   # TODO: Change target to the object you want to follow
CONF = 0                      # TODO: Decide minimum confidence level (0~1)
IMGSZ = 320                   # provided yolo26n.onnx was exported at img size 320. Do not change

# COCO dataset that YOLO was trained on (see https://cocodataset.org/#home)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Make the TARGET is a known object
if TARGET not in COCO_CLASSES:
    print(f"'{TARGET}' is not a known object. Pick one of:\n{COCO_CLASSES}")
    raise SystemExit
target_id = COCO_CLASSES.index(TARGET)

# Load the YOLO model (the yolo26n.onnx file)
print("[INFO] loading YOLO model...")
session = ort.InferenceSession("yolo26n.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Open the camera. Force a small MJPG mode
# Is your Pi is constantly rebooting?
# --> Find a better power supply (5V 1.5A+)
print("[INFO] opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the next frame from the camera
    ok, frame = cap.read()
    if not ok:
        print("[INFO] camera read failed -- is the camera connected?")
        break
    h, w, _ = frame.shape

    # ---- Run YOLO on the frame -------------------------------------------
    # Pack the frame into the shape YOLO wants: resized to IMGSZ x IMGSZ,
    # colors scaled to 0-1, and switched from BGR to RGB.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)
    
    # One forward pass. YOLO26 returns up to 300 detections, already sorted
    # best-first, each row = [x1, y1, x2, y2, confidence, class_number].
    detections = session.run(None, {input_name: blob})[0][0]

    # ---- Find TARGET object among the detections ----------------------------
    center = None
    for x1, y1, x2, y2, conf, cls in detections:
        if int(cls) != target_id:
            continue                   # some other object -> skip it
        if conf < CONF:
            continue                   # low confidence -> skip it
            
        # Detected our TARGET object with CONF+ confidence level:
        # This box is in the IMGSZ x IMGSZ image,
        # so scale it back to the real frame size
        x1, x2 = x1 * w / IMGSZ, x2 * w / IMGSZ
        y1, y2 = y1 * h / IMGSZ, y2 * h / IMGSZ
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        break                          # take the best match and stop

    # ===========================================================================
    # TODO: Step 2 - Write your robot command
    # (3 tasks)
    # ===========================================================================
    if center is not None:             # = if TARGET object is found
        cx, cy = center
        print(f"[INFO] {TARGET} found at x={cx}, y={cy}")

        # TODO: Write your robot control logic. Be creative!
        # HINT: The frame is `w` pixels wide, so its middle is w // 2.
            # e.g. if {cx} is less than {w//2}, which side is the object on?
            
        if cx < w // 2 - 50:
            command = "LEFT"
        elif cx > w // 2 + 50:
            command = "RIGHT"
        else:
            command = "FORWARD"
        
        print("[INFO] robot command:", command)
        # TODO: Actually send `command` to your robot
        
        
    else:
        print(f"[INFO] {TARGET} not in view")
        # TODO: Decide what the robot should do when it can't see the object

    # show the frame to our screen
    cv2.imshow("frame", frame)

    # if [ESC] key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# cleanup
print("\n [INFO] Exiting Program and cleanup stuff \n")
cv2.destroyAllWindows()
cap.release()
