# -----------------------------------------------------------------------------
# How to encode a 2D bounding box into a visualization_msgs/Marker message:
#
# Given a bounding box in the format:
#   bbox = [[x1, y1], [x2, y2]]
# where:
#   (x1, y1) = top-left corner
#   (x2, y2) = bottom-right corner
#
# Step 1: Compute center and size
#   cx = (x1 + x2) / 2.0
#   cy = (y1 + y2) / 2.0
#   w  = x2 - x1
#   h  = y2 - y1
#
# Step 2: Create a Marker message
#   marker = visualization_msgs.msg.Marker()
#   marker.type = Marker.CUBE
#   marker.pose.position.x = cx
#   marker.pose.position.y = cy
#   marker.pose.position.z = 0.0  # Set z = 0 for 2D
#   marker.scale.x = w
#   marker.scale.y = h
#   marker.scale.z = 0.01  # Small thickness for 2D display
#
# Optional:
#   marker.id = object_id       # Assign a unique ID per object
#   marker.ns = "bbox"          # Namespace to group markers
#   marker.color.r/g/b/a        # Set RGBA color
#   marker.header.frame_id      # Coordinate frame (e.g. "map", "camera")
#   marker.header.stamp         # Timestamp
#
# Step 3: Publish via rospy.Publisher("/bbox", Marker, queue_size=1)
# -----------------------------------------------------------------------------

import torch
import numpy as np
import cv2
import time
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from lasr_vision_sam2.msg import MaskWithID, MaskWithIDArray


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

import rospy
from sensor_msgs.msg import Image


class SAM2Node:
    def __init__(self):
        # set up predictor
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        self.cv_bridge = CvBridge()
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

        # set up ROS
        rospy.init_node("sam2_node")
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback, queue_size=1
        )
        self.bbox_sub = rospy.Subscriber(
            "/sum2/bbox", Marker, self.prompt_callback,
        )
        self.mask_pub = rospy.Publisher("/sum2/masks", MaskWithIDArray, queue_size=1)
        self.mask_overlay_pub = rospy.Publisher(
            "/camera/debug_mask_overlay", Image, queue_size=1
        )

        self.frame = None
        self.has_init = False

        self.visualize()

    def visualize(self):
        while not rospy.is_shutdown():
            if self.frame is None:
                rospy.sleep(0.1)
                continue

            if not self.has_init:
                self.predictor.load_first_frame(self.frame)
                obj_points = self.get_point_prompts(self.frame)

                for obj_id, (points, labels) in obj_points.items():
                    points = np.array(points, dtype=np.float32)
                    labels = np.array(labels, dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                        frame_idx=0, obj_id=obj_id, points=points, labels=labels
                    )

                self.has_init = True

            rospy.sleep(0.01)

    def prompt_callback(self, msg):
        if msg.type != Marker.CUBE:
            return
        if self.frame:
            self.predictor.load_first_frame(self.frame)
            cx = msg.pose.position.x
            cy = msg.pose.position.y
            w = msg.scale.x
            h = msg.scale.y

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            bbox = [[x1, y1], [x2, y2]]
            object_id = msg.id

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.has_init:  # only do prediction if the predictor is inited
            self.predictor.add_conditioning_frame(self.frame)
            out_obj_ids, out_mask_logits = self.predictor.track(self.frame)

            # make mask array message to publish
            mask_array_msg = MaskWithIDArray()
            
            # draw the overlay masks and ids
            mask_overlay = self.frame.copy()
            masks_np = out_mask_logits.cpu().numpy()
            for i, mask in enumerate(masks_np):
                # get mask with id messages and add to array
                mask_msg = MaskWithID()
                mask_msg.id = out_obj_ids[i]
                mask_msg.mask = self.cv_bridge.cv2_to_imgmsg(mask, encoding="mono8")
                mask_array_msg.masks.append(mask_msg)  # not sure if better to use binary

                # make coloured mask relay
                binary_mask = (mask[0] > 0).astype(np.uint8)
                colored = np.zeros_like(mask_overlay)
                colored[:, :, 2] = binary_mask * 255
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored, 0.5, 0)
                ys, xs = np.where(binary_mask)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(mask_overlay, f"ID {out_obj_ids[i]}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # publish both messages
            self.mask_pub.publish(mask_array_msg)
            self.mask_overlay_pub.publish(mask_overlay)
        else:
            rospy.sleep(0.01)

if __name__ == "__main__":
    node = SAM2Node()
