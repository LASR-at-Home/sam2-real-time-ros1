#!/usr/bin/env python3
import cv2
import rospy
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from lasr_vision_sam2.msg import (
    BboxWithFlag, 
    DetectionArray,
    Detection3DArray,
)

# === Config ===
use_3d = True
camera = "xtion"
yolo_model = YOLO("yolov8s.pt")

# === ROS setup ===
rospy.init_node("sam2_yolo_test")
bridge = CvBridge()

bbox_pub = rospy.Publisher("/sam2/bboxes", BboxWithFlag, queue_size=10)
track_flag_pub = rospy.Publisher("/sam2/track_flag", Bool, queue_size=1)

first_frame = None
prompts_sent = False
track_flag_sent = False
person_id = 1

# cv2.namedWindow("SAM2 Overlay", cv2.WINDOW_NORMAL)

def publish_yolo_prompts(frame):
    global person_id, prompts_sent
    results = yolo_model(frame)
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(float, box.xyxy.tolist()[0])
            msg = BboxWithFlag()
            msg.obj_id = person_id
            msg.reset = (i == 0)
            msg.clear_old_points = True
            msg.xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            bbox_pub.publish(msg)
            rospy.loginfo(f"Published BBox for ID {person_id}")
            person_id += 1
            rospy.sleep(0.05)
    rospy.sleep(0.5)
    prompts_sent = True

def publish_track_flag():
    global track_flag_sent
    for _ in range(3):
        track_flag_pub.publish(Bool(data=True))
        rospy.sleep(0.1)
    rospy.loginfo("Published /sam2/track_flag")
    track_flag_sent = True

def image_callback(msg):
    global first_frame, prompts_sent, track_flag_sent
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as e:
        rospy.logerr(f"Failed to decode image: {e}")
        return

    if first_frame is None:
        first_frame = frame.copy()
        publish_yolo_prompts(first_frame)

    if prompts_sent:
        publish_track_flag()

    try:
        overlay_msg = rospy.wait_for_message("/sam2/debug/mask_overlay", Image, timeout=1.0)
        overlay = bridge.imgmsg_to_cv2(overlay_msg, desired_encoding="bgr8")
    except:
        rospy.logwarn("No mask overlay received.")
        return

    try:
        if use_3d:
            detections = rospy.wait_for_message("/sam2/detections_3d", Detection3DArray, timeout=1.0)
        else:
            detections = rospy.wait_for_message("/sam2/detections", DetectionArray, timeout=1.0)
    except:
        rospy.logwarn("No detection array received.")
        return

    for det in detections.detections:
        pt = det.point
        label = f"ID {det.name}: ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})"
        if len(det.xywh) >= 2:
            x, y = det.xywh[0], det.xywh[1]
            cv2.putText(overlay, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # cv2.imshow("SAM2 Overlay", overlay)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     rospy.signal_shutdown("Exited by user.")

# === Subscribe ===
rospy.Subscriber(f"/{camera}/rgb/image_raw", Image, image_callback, queue_size=1)
rospy.loginfo("YOLO prompt sender + overlay visualizer started.")
rospy.spin()
cv2.destroyAllWindows()
