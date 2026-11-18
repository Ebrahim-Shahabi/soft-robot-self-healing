import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# change to your detection message type if different
from apriltag_msgs.msg import AprilTagDetectionArray


class ImageOverlayNode(Node):

    def __init__(self):
        super().__init__('image_overlay_node')

        self.bridge = CvBridge()

        # Desired fixed point (RED) 176.95, 38.5
        self.target_x = 488#280
        self.target_y = 155#79
        self.base_depth = 0

        # Store latest detection
        self.detect_x = None
        self.detect_y = None

        # Subscribers
        self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            1
        )

    def depth_callback(self,img):
        bridge = CvBridge()
        # Use cv_bridge() to convert the ROS image to OpenCV format
        #Convert the depth image using the default passthrough encoding
        depth_image = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        # center_idx = np.array(depth_array.shape) / 2
        self.base_depth = depth_array[self.target_y, self.target_x]




    def detection_callback(self, msg):

        for detection in msg.detections:
            self.get_logger().info(
                f'[INFO] AprilTag detected id:{detection.id} '
                f'Position: [{detection.centre.x}, {detection.centre.y}]'
            )
            thr = 20
            if np.linalg.norm([detection.centre.x - self.target_x, detection.centre.y - self.target_y]) < thr:
                continue
            self.detect_x = int(detection.centre.x)
            self.detect_y = int(detection.centre.y)

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        cv2.circle(frame,
                   (self.target_x, self.target_y),
                   12,
                   (0, 0, 255),
                   3)
        
        cv2.circle(frame, (self.target_x,self.target_y), radius=4, color=(0, 0, 255), thickness=-1)
        cv2.line(frame, (self.target_x,self.target_y - 20), (self.target_x,self.target_y - 40), color=(0,0,255))


        if self.detect_x is not None and self.detect_y is not None:
            cv2.circle(frame,
                       (self.detect_x, self.detect_y),
                       12,
                       (0, 255, 0),
                       3)
            cv2.circle(frame, (self.detect_x,self.detect_y), radius=4, color=(255, 0, 0), thickness=-1)
            cv2.putText(frame, f"{np.rad2deg(np.arctan2(-(self.target_y - self.detect_y), self.target_x - self.detect_x)):2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # depth text
        cv2.putText(frame, f"{self.base_depth:.1f}".rstrip('0').rstrip('.') + "mm", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        scale_up = 2 
    
        # Resize the image (scaling up)
        # resized_up = cv2.resize(frame, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_LINEAR)
        resized_up = cv2.resize(frame, (500,500))
        # Display the images
        cv2.imshow("AprilTag Detection", resized_up)

        # cv2.imshow("Camera with markers", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageOverlayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()