#!/usr/bin/env python3
import rospy
import importlib
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

# === Initialize ROS node first ===
rospy.init_node("sam2_yolo_test")

# === Parameters ===
using_lasr_msgs = rospy.get_param("using_lasr_msgs", False)
camera = rospy.get_param("camera", "xtion")
use_3d = rospy.get_param("use_3d", True)

# === Dynamic import message types ===
try:
    if using_lasr_msgs:
        rospy.loginfo("Importing LASR MSGs...")
    else:
        rospy.loginfo("Importing SAM2 MSGs...")
    msg_module = importlib.import_module(
        "lasr_vision_msgs.msg" if using_lasr_msgs else "lasr_vision_sam2.msg"
    )
    BboxWithFlag = getattr(msg_module, "Sam2BboxWithFlag")
    PromptArrays = getattr(msg_module, "Sam2PromptArrays")
    DetectionArray = getattr(msg_module, "DetectionArray")
    Detection3DArray = getattr(msg_module, "Detection3DArray")
except Exception as e:
    rospy.logerr(f"Failed to import messages: {e}")
    raise

# === ROS setup ===
bridge = CvBridge()
yolo_model = YOLO("yolov8s.pt")

prompt_pub = rospy.Publisher("/sam2/prompt_arrays", PromptArrays, queue_size=1)
track_flag_pub = rospy.Publisher("/sam2/track_flag", Bool, queue_size=1)

# === Internal states ===
state = {
    "first_frame": None,
    "prompts_sent": False,
    "track_flag_sent": False,
    "person_id": 1,
}

# === Functions ===


def publish_yolo_prompts(frame, state):
    """Run YOLO and publish prompt arrays."""
    results = yolo_model(frame)
    bbox_list = []

    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(float, box.xyxy.tolist()[0])
            bbox_msg = BboxWithFlag()
            bbox_msg.obj_id = state["person_id"]
            bbox_msg.reset = False
            bbox_msg.clear_old_points = True
            bbox_msg.xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            bbox_list.append(bbox_msg)
            rospy.loginfo(f"Prepared BBox for ID {state['person_id']}")
            state["person_id"] += 1

    if bbox_list:
        prompt_msg = PromptArrays()
        prompt_msg.bbox_array = bbox_list
        prompt_msg.point_array = []
        prompt_msg.reset = True  # full reset
        prompt_pub.publish(prompt_msg)
        rospy.loginfo(f"Published PromptArrays with {len(bbox_list)} BBoxes.")
        rospy.sleep(1.0)
        state["prompts_sent"] = True
    else:
        rospy.logwarn("No valid persons detected by YOLO.")


def publish_track_flag(state):
    """Publish /sam2/track_flag several times."""
    for _ in range(3):
        track_flag_pub.publish(Bool(data=True))
        rospy.sleep(0.1)
    rospy.loginfo("Published /sam2/track_flag.")
    state["track_flag_sent"] = True


def image_callback(msg):
    """Main image callback."""
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as e:
        rospy.logerr(f"Failed to decode image: {e}")
        return

    if state["first_frame"] is None:
        state["first_frame"] = frame.copy()
        publish_yolo_prompts(state["first_frame"], state)

    if state["prompts_sent"] and not state["track_flag_sent"]:
        publish_track_flag(state)

    try:
        overlay_msg = rospy.wait_for_message(
            "/sam2/debug/mask_overlay", Image, timeout=1.0
        )
        overlay = bridge.imgmsg_to_cv2(overlay_msg, desired_encoding="bgr8")
    except:
        rospy.logwarn("No mask overlay received.")
        return

    try:
        if use_3d:
            detections = rospy.wait_for_message(
                "/sam2/detections_3d", Detection3DArray, timeout=1.0
            )
        else:
            detections = rospy.wait_for_message(
                "/sam2/detections", DetectionArray, timeout=1.0
            )
    except:
        rospy.logwarn("No detection array received.")
        return

    for det in detections.detections:
        pt = det.point
        label = f"ID {det.name}: ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})"
        if len(det.xywh) >= 2:
            x, y = det.xywh[0], det.xywh[1]
            cv2.putText(
                overlay,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    # === Visualization (optional) ===
    # cv2.imshow("SAM2 Overlay", overlay)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     rospy.signal_shutdown("Exited by user.")


# === Subscribe ===
rospy.Subscriber(f"/{camera}/rgb/image_raw", Image, image_callback, queue_size=1)

rospy.loginfo("SAM2 YOLO prompt sender + overlay visualizer started.")
rospy.spin()
cv2.destroyAllWindows()
