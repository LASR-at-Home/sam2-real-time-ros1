"""
SAM2 node for ROS
Get image topic /camera/image_raw
Publish mask topic /camera/mask, /camera/mask_overlay
"""

import torch
import numpy as np
import cv2
import time
import threading


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

        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

        # set up ROS
        rospy.init_node("sam2_node")
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback, queue_size=1
        )
        self.mask_pub = rospy.Publisher("/camera/mask", Image, queue_size=1)
        self.mask_overlay_pub = rospy.Publisher(
            "/camera/mask_overlay", Image, queue_size=1
        )

        self.frame = None
        self.frame_vis = None
        self.has_init = False

        self.tracking_thread = threading.Thread(target=self.tracking_thread)
        self.tracking_thread.start()

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

            if self.frame_vis is not None:
                cv2.imshow("frame_vis", self.frame_vis)
                cv2.waitKey(1)
            else:
                pass

            rospy.sleep(0.01)

    def image_callback(self, msg):
        self.frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )

    def tracking_thread(self):
        while not rospy.is_shutdown():
            if not self.has_init or self.frame is None:
                rospy.sleep(0.1)
                continue

            stime = time.time()

            frame = self.frame.copy()

            height, width = self.frame.shape[:2]

            out_obj_ids, out_mask_logits = self.predictor.track(frame)
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)

            out_mask_logits = out_mask_logits.permute(0, 2, 3, 1).cpu().numpy()
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).astype(np.uint8) * 255

                all_mask = cv2.bitwise_or(all_mask, out_mask)

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            self.frame_vis = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
            self.mask = all_mask

            rospy.sleep(0.01)

    def get_point_prompts(self, frame):
        obj_points = {}
        ann_obj_id = 1
        cur_points = []
        cur_labels = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal cur_points, cur_labels
            if event == cv2.EVENT_LBUTTONDOWN:
                cur_points.append([x, y])
                cur_labels.append(1)
                print(f"Added positive click at ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                cur_points.append([x, y])
                cur_labels.append(0)
                print(f"Added negative click at ({x}, {y})")

        cv2.namedWindow("frame", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("frame", 1280, 720)
        cv2.setMouseCallback("frame", mouse_callback)

        while True:
            # overlay points on frame
            frame_vis = frame.copy()
            for point, label in zip(cur_points, cur_labels):
                cv2.circle(
                    frame_vis,
                    tuple(point),
                    5,
                    (0, 255, 0) if label == 1 else (0, 0, 255),
                    -1,
                )

            cv2.imshow("frame", frame_vis)
            k = cv2.waitKey(1)

            if k == ord("q"):
                break
            elif k == ord("n"):  # next object
                obj_points[ann_obj_id] = (cur_points, cur_labels)
                ann_obj_id += 1
                cur_points = []
                cur_labels = []
            elif k == ord("c"):  # clear points
                cur_points = []
                cur_labels = []

        cv2.destroyAllWindows()
        return obj_points


if __name__ == "__main__":
    node = SAM2Node()
