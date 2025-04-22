import cv2
import torch
import rospy
from ultralytics import YOLO
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from lasr_vision_sam2.msg import BboxWithFlag

# Initialize ROS node
rospy.init_node("yolov8_tracker_with_camera", anonymous=True)

# Initialize ROS publishers
bbox_pub = rospy.Publisher("/sum2/bboxes", BboxWithFlag, queue_size=10)
image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=1)
track_flag_pub = rospy.Publisher("/sum2/track_flag", Bool, queue_size=1)

# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")

# Initialize bridge
bridge = CvBridge()

# Flags
person_id = 1
prompts_sent = False
track_flag_sent = False
first_frame = None

# Use webcam (optional, for debugging)
# cap = cv2.VideoCapture(0)
# assert cap.isOpened(), "Failed to open video capture"

# === Image callback from ROS topic ===
def image_callback(msg):
    global first_frame, prompts_sent, track_flag_sent, person_id

    if first_frame is None:
        try:
            first_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rospy.loginfo("Received first frame from ROS topic.")
        except Exception as e:
            rospy.logerr(f"Failed to decode image: {e}")
            return

    if first_frame is not None and not prompts_sent:
        results = yolo_model(first_frame)
        found_any = False

        for idx, box in enumerate(results[0].boxes):
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(float, box.xyxy.tolist()[0])
                h, w = first_frame.shape[:2]
                # x1 /= w
                # x2 /= w
                # y1 /= h
                # y2 /= h

                bbox_msg = BboxWithFlag()
                bbox_msg.obj_id = person_id
                bbox_msg.reset = True if idx == 0 else False
                bbox_msg.clear_old_points = True
                bbox_msg.bbox_points = [
                    Point(x=x1, y=y1, z=0.0),
                    Point(x=x2, y=y2, z=0.0),
                ]

                bbox_pub.publish(bbox_msg)
                rospy.loginfo(f"Published BBox for person ID {person_id}")
                person_id += 1
                found_any = True
                rospy.sleep(0.05)

        if found_any:
            prompts_sent = True
            rospy.loginfo("Finished sending all BBox prompts.")

    elif prompts_sent and not track_flag_sent:
        for _ in range(4):  # repeat to ensure delivery
            track_flag_pub.publish(Bool(data=True))
            rospy.sleep(0.1)
        rospy.loginfo("Published /sum2/track_flag=True")
        track_flag_sent = True

# Subscribe to ROS image topic
rospy.Subscriber("/xtion/rgb/image_raw", Image, image_callback, queue_size=1)

# === Optional webcam loop for testing visualization ===
# while not rospy.is_shutdown():
#     ret, frame = cap.read()
#     if not ret:
#         rospy.logwarn("Failed to read frame from camera.")
#         continue
#
#     try:
#         ros_img = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
#         image_pub.publish(ros_img)
#     except Exception as e:
#         rospy.logerr(f"Failed to publish image: {e}")
#         continue
#
#     cv2.imshow("YOLOv8 Detection", frame)
#     if cv2.waitKey(1) == 27:
#         break
#
#     rospy.sleep(0.1)

# Keep node alive
rospy.spin()
# cap.release()
# cv2.destroyAllWindows()
