# -----------------------------------------------------------------------------
# IMPORTANT!!!!!! ALL THE COORDINATES MUST BE NORMALISED!!!!!!
# -----------------------------------------------------------------------------
# How to construct a BBoxPointsWithFlag message:
#
# This message represents a 2D bounding box with object ID and a reset flag.
# It uses two points to define the box:
#   - The top-left corner (x1, y1)
#   - The bottom-right corner (x2, y2)
#
# FIELD MEANING:
#   x1, y1: coordinates of the top-left corner of the bounding box
#   x2, y2: coordinates of the bottom-right corner of the bounding box
#   obj_id: integer ID assigned to the object (used for tracking or association)
#   reset: if True, indicates that previous information for this object should be cleared
#
# -----------------------------------------------------------------------------
# Step-by-step construction:
#
# 1. Import types:
#     from my_perception.msg import BBoxPointsWithFlag
#     from geometry_msgs.msg import Point
#
# 2. Create the message:
#     msg = BBoxPointsWithFlag()
#
# 3. Assign object ID and reset flag:
#     msg.obj_id = 42         # ID for this bounding box
#     msg.reset = False       # Set to True if this should reset existing data
#
# 4. Create the bounding box corner points:
#     p1 = Point()            # Top-left corner
#     p1.x = x1               # Minimum x-coordinate
#     p1.y = y1               # Minimum y-coordinate
#     p1.z = 0.0              # 2D, so z is 0
#
#     p2 = Point()            # Bottom-right corner
#     p2.x = x2               # Maximum x-coordinate
#     p2.y = y2               # Maximum y-coordinate
#     p2.z = 0.0
#
# 5. Assign to message:
#     msg.bbox_points = [p1, p2]
#
# 6. Publish:
#     bbox_pub.publish(msg)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# How to construct a PointsWithLabels message:
#
# This message contains:
#   - a list of 2D points with explicit x/y coordinates
#   - a label for each point
#   - an object identifier (obj_id)
#   - a reset flag indicating whether to clear previous point data
#
# FIELD MEANING:
#   point.x: horizontal coordinate (x-axis), increases to the right
#            x = 0 refers to the left edge of the image
#
#   point.y: vertical coordinate (y-axis), increases downward
#            y = 0 refers to the top edge of the image
#
#   point.z: always set to 0.0 (unused, for 2D purposes only)
#
#   label.data: integer label assigned to the point (e.g., 0, 1)
#   obj_id: integer identifier grouping points under the same object
#   reset: boolean flag; True clears all previous points for this object
#
# -----------------------------------------------------------------------------
# Step-by-step construction:
#
# 1. Import message types:
#     from my_perception.msg import PointsWithLabels
#     from geometry_msgs.msg import Point
#     from std_msgs.msg import Int32
#
# 2. Create the message:
#     msg = PointsWithLabels()
#     msg.obj_id = 1
#     msg.reset = False
#
# 3. Define points:
#     p1 = Point()
#     p1.x = x
#     p1.y = y
#     p1.z = 0.0
#
#     l1 = Int32()
#     l1.data = 1     # example label
#
# 4. Assign to message:
#     msg.points = [p1]
#     msg.labels = [l1]
#
# 5. Publish:
#     pub.publish(msg)
# -----------------------------------------------------------------------------

import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
from lasr_vision_sam2.msg import (
    MaskWithID,
    MaskWithIDArray,
    BboxWithFlag,
    PointsWithLabelsAndFlag,
)

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class SAM2Node:
    def __init__(self):

        # set up predictor
        # ToDo: need to deal with this stupid absolute path, potentialy add to ros param:
        ckpt_path = "/home/bentengma/work_space/robocup_ws/src/Base/common/third_party/sam2-real-time-ros1/checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        self.bridge = CvBridge()
        self.predictor = build_sam2_camera_predictor(model_cfg, ckpt_path)

        rospy.loginfo("SAM2 predictor initialized from checkpoint.")
        print("SAM2 predictor initialized from checkpoint.")

        # set up ROS
        rospy.init_node("lasr_vision_sam2_node")
        self.track_flag_sub = rospy.Subscriber(
            "/sum2/track_flag", Bool, self.track_flag_callback
        )
        self.condition_frame_flag_sub = rospy.Subscriber(
            "/sum2/add_conditioning_frame_flag",
            Bool,
            self.add_conditioning_frame_flag_callback,
        )
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback, queue_size=1
        )
        self.bbox_sub = rospy.Subscriber(
            "/sum2/bboxes",
            BboxWithFlag,
            self.bbox_prompt_callback,
        )
        self.point_sub = rospy.Subscriber(
            "/sum2/points",
            PointsWithLabelsAndFlag,
            self.points_prompt_callback,
        )
        self.mask_pub = rospy.Publisher("/camera/masks", MaskWithIDArray, queue_size=1)
        self.mask_overlay_pub = rospy.Publisher(
            "/camera/debug_mask_overlay", Image, queue_size=1
        )

        rospy.loginfo("SAM2Node ROS interfaces set up.")
        print("SAM2Node ROS interfaces set up.")

        self.add_conditioning_frame_flag = True
        self.frame = None
        self.has_first_frame = False
        self.track_flag = False

    def track_flag_callback(self, msg):
        self.track_flag = msg.data
        rospy.loginfo(f"Track flag set to {self.track_flag}")
        print(f"Track flag set to {self.track_flag}")

    def add_conditioning_frame_flag_callback(self, msg):
        self.add_conditioning_frame_flag = msg.data
        rospy.loginfo(
            f"Adding conditional frame flag set to {self.add_conditioning_frame_flag}"
        )
        print(
            f"Adding conditional frame flag set to {self.add_conditioning_frame_flag}"
        )

    def bbox_prompt_callback(self, msg):
        if len(msg.bbox_points) != 2:
            rospy.logwarn("Received BBox with invalid number of points.")
            return

        if self.frame is None:
            rospy.logwarn("Received BBox but no frame is available yet.")
            return

        reset_flag = msg.reset
        if reset_flag:
            rospy.loginfo(f"Resetting state for object ID {msg.obj_id}.")
            rospy.logwarn(f"Resetting state for object ID {msg.obj_id}.")
            rospy.logerr(f"Resetting state for object ID {msg.obj_id}.")
            if "point_inputs_per_obj" in self.predictor.condition_state:
                self.predictor.reset_state()
            self.has_first_frame = False
            self.track_flag = False

        if not self.has_first_frame:
            self.predictor.load_first_frame(self.frame)
            self.has_first_frame = True
            rospy.loginfo("Loaded first frame for BBox prompt.")

        x1, y1 = msg.bbox_points[0].x, msg.bbox_points[0].y
        x2, y2 = msg.bbox_points[1].x, msg.bbox_points[1].y
        bbox = [[x1, y1], [x2, y2]]
        obj_id = msg.obj_id

        frame_idx = self.predictor.condition_state.get("num_frames", 1) - 1
        frame_idx = max(0, frame_idx)
        rospy.loginfo(
            f"Adding BBox for object ID {obj_id}: [{x1:.1f}, {y1:.1f}], [{x2:.1f}, {y2:.1f}] to frame {frame_idx}."
        )

        self.predictor.add_new_prompt(
            frame_idx=frame_idx,
            obj_id=obj_id,
            bbox=bbox,
            points=None,
            labels=None,
            clear_old_points=msg.clear_old_points,
            normalize_coords=True,
        )

    def points_prompt_callback(self, msg):
        if len(msg.points) != len(msg.labels):
            rospy.logwarn("Received points and labels with mismatched lengths.")
            return

        if self.frame is None:
            rospy.logwarn("Received points prompt but no frame is available.")
            return

        reset_flag = msg.reset
        if reset_flag:
            rospy.loginfo(f"Resetting state for object ID {msg.obj_id}.")
            if "point_inputs_per_obj" in self.predictor.condition_state:
                self.predictor.reset_state()
            self.has_first_frame = False
            self.track_flag = False

        if not self.has_first_frame:
            self.predictor.load_first_frame(self.frame)
            self.has_first_frame = True
            rospy.loginfo("Loaded first frame for point prompt.")

        obj_id = msg.obj_id
        points = [[p.x, p.y] for p in msg.points]
        labels = [l.data for l in msg.labels]

        frame_idx = self.predictor.condition_state.get("num_frames", 1) - 1
        frame_idx = max(0, frame_idx)
        rospy.loginfo(
            f"Adding {len(points)} points for object ID {obj_id} to frame {frame_idx}."
        )

        self.predictor.add_new_points(
            frame_idx=frame_idx,
            obj_id=obj_id,
            bbox=None,
            points=points,
            labels=labels,
            clear_old_points=msg.clear_old_points,
            normalize_coords=True,
        )

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
        except Exception as e:
            rospy.logerr(f"Failed to decode image: {e}")
            return

        if not self.has_first_frame:
            self.track_flag = False
            return  # Waiting for bbox or points to start

        if not self.track_flag:
            return

        try:
            if self.add_conditioning_frame_flag:
                self.predictor.add_conditioning_frame(self.frame)
                rospy.loginfo("Conditioning frame added.")
            out_obj_ids, out_mask_logits = self.predictor.track(self.frame)
            rospy.loginfo(f"Tracking {len(out_obj_ids)} objects in current frame.")
        except Exception as e:
            rospy.logerr(f"Error during tracking: {e}")
            return

        # Create mask message array
        mask_array_msg = MaskWithIDArray()
        mask_overlay = self.frame.copy()
        masks_np = out_mask_logits.cpu().numpy()

        for i, mask in enumerate(masks_np):
            obj_id = out_obj_ids[i]

            try:
                # Print debug shape/type info
                rospy.loginfo(f"[Tracking] Processing object ID {obj_id}")
                rospy.loginfo(
                    f" - mask raw shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min():.4f}, max: {mask.max():.4f}"
                )

                # Ensure mono8 format
                mask_np = mask[0]  # Assume (1, H, W)
                mask_uint8 = (mask_np * 255).clip(0, 255).astype(np.uint8)

                mask_msg = MaskWithID()
                mask_msg.id = obj_id
                mask_msg.mask = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                mask_array_msg.masks.append(mask_msg)
                rospy.loginfo(f" - Successfully encoded mask for object ID {obj_id}")
            except Exception as e:
                rospy.logerr(
                    f"[Tracking] Failed to encode mask for object ID {obj_id}: {e}"
                )
                continue

            # Draw visualization overlay
            binary_mask = (mask_np > 0).astype(np.uint8)
            colored = np.zeros_like(mask_overlay)
            colored[:, :, 2] = binary_mask * 255
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored, 0.5, 0)
            ys, xs = np.where(binary_mask)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(
                    mask_overlay,
                    f"ID {obj_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                rospy.loginfo(f" - Middle point {(cx, cy)}")

        # Publish
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
            self.mask_pub.publish(mask_array_msg)
            self.mask_overlay_pub.publish(overlay_msg)
            rospy.loginfo("Published mask and overlay.")
        except Exception as e:
            rospy.logerr(f"Failed to publish output: {e}")


# if __name__ == "__main__":
node = SAM2Node()
rospy.spin()
