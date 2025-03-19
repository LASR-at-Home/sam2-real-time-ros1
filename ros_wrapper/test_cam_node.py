"""
This is a test node to test the camera node.
pub: /camera/image_raw
"""

import rospy
import cv2

# import cv_bridge

from sensor_msgs.msg import Image


class TestCamNode:
    def __init__(self):
        rospy.init_node("test_cam_node")
        self.image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=1)

        self.cap = cv2.VideoCapture("./notebooks/videos/aquarium/aquarium.mp4")
        self.rate = rospy.Rate(20)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                # break
                self.reset_cap()
                continue

            cv2.imshow("frame", frame)
            cv2.waitKey(1)

            # frame_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            frame_msg = Image()
            frame_msg.data = frame.tobytes()
            frame_msg.height = frame.shape[0]
            frame_msg.width = frame.shape[1]
            frame_msg.encoding = "rgb8"
            frame_msg.is_bigendian = False
            frame_msg.step = 3 * frame.shape[1]
            frame_msg.header.stamp = rospy.Time.now()

            self.image_pub.publish(frame_msg)

            self.rate.sleep()

    def reset_cap(self):
        self.cap = cv2.VideoCapture("./notebooks/videos/aquarium/aquarium.mp4")


if __name__ == "__main__":
    node = TestCamNode()
    node.run()
