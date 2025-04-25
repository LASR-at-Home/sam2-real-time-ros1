import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

import rospy
import rospkg
import os

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


rospy.init_node("lasr_vision_sam2_node")
using_lasr_msgs = using_lasr_msgs = rospy.get_param("~using_lasr_msgs", False)

if using_lasr_msgs:
    from lasr_vision_msgs.msg import (
        MaskWithID,
        MaskWithIDArray,
        BboxWithFlag,
        PointsWithLabelsAndFlag,
        CentrePointWithIDArray,
        CentrePointWithID,
        Detection,
        DetectionArray,
        Detection3D,
        Detection3DArray,
    )
else:
    from lasr_vision_sam2.msg import (
        MaskWithID,
        MaskWithIDArray,
        BboxWithFlag,
        PointsWithLabelsAndFlag,
        CentrePointWithIDArray,
        CentrePointWithID,
        Detection,
        DetectionArray,
        Detection3D,
        Detection3DArray,
    )


class SAM2Node:
    def __init__(self):

        # set up predictor
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("lasr_vision_sam2")
        ckpt_path = os.path.join(pkg_path, "checkpoints", "sam2.1_hiera_small.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        self.bridge = CvBridge()
        self.predictor = build_sam2_camera_predictor(model_cfg, ckpt_path)

        self.camera = rospy.get_param("~camera", "xtion")
        self.use_3d = rospy.get_param("~use_3d", True)

        rospy.loginfo("SAM2 predictor initialized from checkpoint.")
        print("SAM2 predictor initialized from checkpoint.")

        # set up ROS
        self.track_flag_sub = rospy.Subscriber(
            "/sam2/track_flag", Bool, self.track_flag_callback
        )
        self.condition_frame_flag_sub = rospy.Subscriber(
            "/sam2/add_conditioning_frame_flag",
            Bool,
            self.add_conditioning_frame_flag_callback
        )
        if self.use_3d:
            self.image_sub = message_filters.Subscriber(
                f"/{self.camera}/rgb/image_raw", Image
            )
            self.depth_sub = message_filters.Subscriber(
                f"/{self.camera}/depth_registered/image_raw", Image
            )
            self.depth_info_sub = message_filters.Subscriber(
                f"/{self.camera}/depth_registered/camera_info", CameraInfo
            )
            ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub, self.depth_info_sub], queue_size=25, slop=0.1
            )
            ts.registerCallback(self.image_callback_3d)
            self.detection3d_pub = rospy.Publisher(
                "/sam2/detections_3d", Detection3DArray, queue_size=1
            )
            self.centre_marker_pub = rospy.Publisher(
                "/sam2/centre_marker", Marker, queue_size=1
            )

        else:
            self.image_sub = rospy.Subscriber(
                f"/{self.camera}/rgb/image_raw", Image, self.image_callback, queue_size=1
            )
            self.detection_pub = rospy.Publisher(
                "/sam2/detections", DetectionArray, queue_size=1
            )

        self.bbox_sub = rospy.Subscriber(
            "/sam2/bboxes",
            BboxWithFlag,
            self.bbox_prompt_callback,
        )
        self.point_sub = rospy.Subscriber(
            "/sam2/points",
            PointsWithLabelsAndFlag,
            self.points_prompt_callback,
        )
        self.mask_pub = rospy.Publisher(
            "/sam2/debug/masks", MaskWithIDArray, queue_size=1
        )
        self.mask_overlay_pub = rospy.Publisher(
            "/sam2/debug/mask_overlay", Image, queue_size=1
        )
        self.centre_pub = rospy.Publisher(
            "/sam2/debug/centre_points", CentrePointWithIDArray, queue_size=1
        )

        rospy.loginfo("SAM2Node ROS interfaces set up.")
        print("SAM2Node ROS interfaces set up.")

        self.add_conditioning_frame_flag = True
        self.frame = None
        self.has_first_frame = False
        self.track_flag = False

    def track_flag_callback(self, msg):
        if self.has_first_frame:
            self.track_flag = msg.data
            rospy.loginfo(f"Track flag set to {self.track_flag}")
            print(f"Track flag set to {self.track_flag}")
        else:
            self.track_flag = False
            rospy.loginfo(f"Track flag set to {self.track_flag} because there's no promt at all.")
            print(f"Track flag set to {self.track_flag} because there's no promt at all.")

    def add_conditioning_frame_flag_callback(self, msg):
        self.add_conditioning_frame_flag = msg.data
        rospy.loginfo(
            f"Adding conditional frame flag set to {self.add_conditioning_frame_flag}"
        )
        print(
            f"Adding conditional frame flag set to {self.add_conditioning_frame_flag}"
        )

    def bbox_prompt_callback(self, msg):
        rospy.loginfo("Received BBox Prompt!")
        if len(msg.xywh) != 4:
            rospy.logwarn("Received BBox with invalid number of values.")
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

        x, y, w, h = msg.xywh  # Unpack the 4-element int32[] array
        x1, y1 = x, y
        x2, y2 = x + w, y + h
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
            # the actual coords to give should not be normalised
            normalize_coords=True,
        )

    def points_prompt_callback(self, msg):
        rospy.loginfo("Received Point Prompt!")
        if len(msg.xy) % 2 != 0 or len(msg.xy) // 2 != len(msg.labels):
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

        # Reshape xy: [x1, y1, x2, y2, ...] -> [[x1, y1], [x2, y2], ...]
        points = [ [msg.xy[i], msg.xy[i+1]] for i in range(0, len(msg.xy), 2) ]
        labels = list(msg.labels)

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
            # the actual coords to give should not be normalised
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
        mask_centers_msg = CentrePointWithIDArray()
        detection_array_msg = Detection3DArray()
        detection_array_msg.detections = []

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
                # Bounding box in [x, y, w, h] format (int32)
                x_min, y_min = xs.min(), ys.min()
                x_max, y_max = xs.max(), ys.max()
                xywh = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

                cx, cy = int(xs.median()), int(ys.median())
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

                pt = Point()
                pt.x = float(cx)
                pt.y = float(cy)
                pt.z = 0.0

                center_msg = CentrePointWithID()
                center_msg.id = obj_id
                center_msg.centre_point = pt
                mask_centers_msg.centre_points.append(center_msg)

                det = Detection()
                det.name = str(obj_id)
                det.confidence = 1.0
                det.xywh = xywh
                det.xyseg = []
                detection_array_msg.detections.append(det)

        # Publish
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
            self.mask_pub.publish(mask_array_msg)
            self.mask_overlay_pub.publish(overlay_msg)
            self.centre_pub.publish(mask_centers_msg)
            rospy.loginfo("Published mask and overlay.")
            self.detection_pub.publish(detection_array_msg)
            rospy.loginfo("Published DetectionArray.")
        except Exception as e:
            rospy.logerr(f"Failed to publish output: {e}")


    def image_callback_3d(self, rgb_msg, depth_msg, cam_info_msg):
        # rospy.loginfo("Image received!")
        try:
            self.frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8").copy()
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").copy()
        except Exception as e:
            rospy.logerr(f"Failed to decode image or depth: {e}")
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
            rospy.logerr(f"Error during tracking: {e}, deactivated tracking.")
            # self.track_flag = False
            return

        # Create mask message array
        mask_array_msg = MaskWithIDArray()
        mask_centers_msg = CentrePointWithIDArray()
        detection_array_msg = Detection3DArray()
        detection_array_msg.detections = []

        mask_overlay = self.frame.copy()
        masks_np = out_mask_logits.cpu().numpy()

        fx = cam_info_msg.K[0]
        fy = cam_info_msg.K[4]
        cx = cam_info_msg.K[2]
        cy = cam_info_msg.K[5]

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

            # Draw visualization overlapty
            binary_mask = (mask_np > 0).astype(np.uint8)
            colored = np.zeros_like(mask_overlay)
            colored[:, :, 2] = binary_mask * 255
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored, 0.5, 0)
            
            ys, xs = np.where(binary_mask)
            if len(xs) > 0 and len(ys) > 0:
                # Bounding box in [x, y, w, h] format (int32)
                x_min, y_min = xs.min(), ys.min()
                x_max, y_max = xs.max(), ys.max()
                xywh = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                
                # Filter out all valid depth points in the mask
                depths = depth_image[ys, xs].astype(np.float32)
                valid = depths > 0

                if np.any(valid):
                    xs_valid = xs[valid]
                    ys_valid = ys[valid]
                    depths_valid = depths[valid]

                    # Compute the median of each coordinate
                    u_median = int(np.median(xs_valid))
                    v_median = int(np.median(ys_valid))
                    d_median = float(np.median(depths_valid))

                    pt = Point()
                    pt.x = (u_median - cx) * d_median / fx
                    pt.y = (v_median - cy) * d_median / fy
                    pt.z = d_median

                    cv2.putText(mask_overlay, f"ID {obj_id}", (u_median, v_median),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    rospy.loginfo(f" - 3D median center: ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})")
                else:
                    rospy.logwarn(f"No valid depth in mask for object ID {obj_id}, using 2D fallback")
                    pt = Point()
                    pt.x = float(np.median(xs))
                    pt.y = float(np.median(ys))
                    pt.z = 0.0

                center_msg = CentrePointWithID()
                center_msg.id = obj_id
                center_msg.centre_point = pt
                mask_centers_msg.centre_points.append(center_msg)

                det = Detection3D()
                det.name = str(obj_id)
                det.confidence = 1.0
                det.xywh = xywh
                det.xyseg = []
                det.point = pt
                detection_array_msg.detections.append(det)

                marker = Marker()
                marker.header.frame_id = depth_msg.header.frame_id
                marker.header.stamp = depth_msg.header.stamp
                marker.ns = "sam2_tracking"
                marker.id = obj_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = pt
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.lifetime = rospy.Duration(0.5)
                self.centre_marker_pub.publish(marker)

        # Publish
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
            self.mask_pub.publish(mask_array_msg)
            self.mask_overlay_pub.publish(overlay_msg)
            self.centre_pub.publish(mask_centers_msg)
            rospy.loginfo("Published mask, overlay, and 3D centers.")
            self.detection3d_pub.publish(detection_array_msg)
            rospy.loginfo("Published Detection3DArray.")
        except Exception as e:
            rospy.logerr(f"Failed to publish outputs: {e}")


# if __name__ == "__main__":
node = SAM2Node()
rospy.spin()
