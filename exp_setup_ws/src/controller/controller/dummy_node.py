import rclpy
from rclpy.node import Node
import numpy as np
from controller.library import utils_dir as du
import pandas as pd
from pathlib import Path
import os
import time

# from apriltag_msgs.msg import AprilTagDetectionArray
from pneumatic_actuation_msgs.msg import FluidPressures
from std_msgs.msg import Float32
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool # for the init flag, to signal the start of sending random data.
from my_msgs.msg import RecordingMsg


DATA_DIR   = "./src/controller/controller/"

os.makedirs("./data/tracker_online/controller/online_controller/tracking", exist_ok=True)

path_trackingfile = "./data/tracker_online/controller/online_controller/tracking/tracking_output" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

file_track = open(path_trackingfile, 'w') # open a file to write the data.

class dummyNode(Node):

    def __init__(self):
        super().__init__('DummyNode')


        self.create_subscription(
            Bool,
            '/dummy_listener',
            self.reference_callback,
            10
        )
        self.create_subscription(
            RecordingMsg,
            "/data_recorder_online/info",
            self.recorder_callback,       # updates q_current AND buffer
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "/controller_output",
            self.controller_callback,       # updates q_current AND buffer
            10,
        )
        self.controller_publisher_ = self.create_publisher(Float32MultiArray, '/ref', 10)
        self.angle_publisher_ = self.create_publisher(Float32, '/ref_angle', 10)
        self.flag_init = False
        self.ramp_done = False
        self.get_logger().info("[INFO] DummyNode initialized, waiting for init flag...")
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # read data from the file
        # self.data = pd.read_csv(DATA_DIR + "online_output20260526-150423.csv")
        # self.states, self.targets = du.extract_states_targets(self.data)
        self.data = pd.read_csv(DATA_DIR + "new_trajectory.csv")
        self.states = self.data[["x"]].values
        self.targets = self.data[["y"]].values

        self.timer_paused = 0
        self.data_index = 0
        self.got_timer = False # toggle flag to record when a data is sent
        self.got_control = False # toggle flag to record when a control is received
        self.to_send = None
        self.t_system = time.time() # also have a timer for the whole system.

        file_track.write(
                "t_system[s], index,u[mBar], u_ff[mBar], u_fb[mBar],"
                "angle_current[rad], angle_ref[rad], x_current[m],"
                "x_ref[m], y_current[m], y_ref[m], dx_error[m], dy_error[m], angle_error[rad] \n"
            )

    def reference_callback(self, msg):
        if msg.data:
            self.get_logger().info("[INFO] Dummy node initialized.")
            self.get_logger().info("[INFO] Attempting to recreate trapezoid movement.")
            self.flag_init = True

        else:
            self.get_logger().info("[WARN] You sent false as init flag, like what?")

    def recorder_callback(self, msg):
        if self.flag_init:
            if self.got_timer and self.got_control:
                self.get_logger().info(f"[INFO] Recording data for index {self.data_index-1}")
                t_system = time.time() - self.t_system
                u = self.u[0]
                u_ff = self.u[1]
                u_fb = self.u[2]
                dx    = float(msg.xe - msg.x0)
                dy    = float(msg.ye - msg.y0)
                angle = float(msg.angle)
                
                dx_error = self.to_send[1] - dx
                dy_error = self.to_send[2] - dy
                angle_error = self.to_send[0] - angle

                file_track.write(
                    f"{t_system:.3f}, {self.data_index-1}, {u:.2f},{u_ff:.2f},{u_fb:.2f},"
                    f"{angle:.4f},{self.to_send[0]:.4f},{dx:.4f},{self.to_send[1]:.4f},{dy:.4f},{self.to_send[2]:.4f},{dx_error:.4f},{dy_error:.4f},{angle_error:.2f}\n"
                )
                self.got_timer = False
                self.got_control = False
            elif self.got_timer and not self.got_control:
                self.get_logger().info(f"[INFO] Waiting for controller output for index {self.data_index-1} to record data...")
            elif not self.got_timer and self.got_control:
                self.get_logger().info(f"[INFO] Waiting for timer callback for index {self.data_index-1} to record data...")
            else:
                self.get_logger().info("[INFO] Waiting for both timer and control to be received before recording data...")
        else:
            self.get_logger().info("[INFO] Waiting for init flag to start recording data...")
    
    def controller_callback(self, msg):
        if self.flag_init:
            if self.got_timer:
                self.get_logger().info(f"[INFO] Received controller output for index {self.data_index}")
                self.u = msg.data
                self.got_control = True
        else:
            self.get_logger().info("[INFO] Waiting for init flag to start sending pressures...")
            

    def timer_callback(self):
        if self.flag_init:
            msg = Float32MultiArray()
            # dx, dy, angle = self.states[self.data_index, :]
            angle = float(self.targets[self.data_index])
            dx, dy = float(0), float(0)
            msg.data = [angle, dx, dy]
            self.to_send = msg.data
            self.controller_publisher_.publish(msg)
            self.angle_publisher_.publish(Float32(data=np.rad2deg(angle)))
            self.data_index += 1
            self.got_timer = True
        else:
            self.get_logger().info("[INFO] Waiting for init flag to start sending reference...")

def main(args=None):
    rclpy.init(args=args)
    node = dummyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()