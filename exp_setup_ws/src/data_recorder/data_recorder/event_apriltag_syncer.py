import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

class SyncedCameraBridge(Node):
    def __init__(self):
        super().__init__('synced_camera_bridge')

        self.image_sub = self.create_subscription(
            Image,
            '/event_camera/image_raw',
            self.image_cb,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/event_camera_synced/image_raw',
            10
        )

        self.info_pub = self.create_publisher(
            CameraInfo,
            '/event_camera_synced/camera_info',
            10
        )

    def image_cb(self, msg: Image):
        # Republish image with the same header
        out_img = Image()
        out_img.header = msg.header
        out_img.height = msg.height
        out_img.width = msg.width
        out_img.encoding = msg.encoding
        out_img.is_bigendian = msg.is_bigendian
        out_img.step = msg.step
        out_img.data = msg.data

        # Build matching camera_info
        info = CameraInfo()
        info.header.stamp = msg.header.stamp
        info.header.frame_id = msg.header.frame_id
        info.width = msg.width
        info.height = msg.height

        fx = 300.0
        fy = 300.0
        cx = msg.width / 2.0
        cy = msg.height / 2.0

        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.image_pub.publish(out_img)
        self.info_pub.publish(info)

def main():
    rclpy.init()
    node = SyncedCameraBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()