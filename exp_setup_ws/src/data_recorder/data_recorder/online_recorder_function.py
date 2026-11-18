import rclpy
import time
import numpy as np
from rclpy.node import Node

from std_msgs.msg import String
from pneumatic_actuation_msgs.msg import FluidPressures
from apriltag_msgs.msg import AprilTagDetectionArray
from my_msgs.msg import RecordingMsg
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

write_to_file = False
if write_to_file:
    os.makedirs("./data/tracker_online/controller/online_controller/pressure", exist_ok=True)
    os.makedirs("./data/tracker_online/controller/online_controller/aprilTag", exist_ok=True)
    os.makedirs("./data/tracker_online/controller/online_controller/online", exist_ok=True)
    os.makedirs("./data/tracker_online/controller/online_controller/video", exist_ok=True)

    path_of_pressure_file = "./data/tracker_online/controller/online_controller/pressure/pressure_output" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    path_of_aprilTag_file = "./data/tracker_online/controller/online_controller/aprilTag/aprilTag_output" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    path_of_online_file = "./data/tracker_online/controller/online_controller/online/online_output" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    path_of_video_file = "./data/tracker_online/controller/online_controller/video/video_output" + time.strftime("%Y%m%d-%H%M%S") + ".avi"

    file_pressure = open(path_of_pressure_file, 'w') # open a file to write the data.
    file_aprilTag = open(path_of_aprilTag_file, 'w') # open a file to write the data.
    file_online = open(path_of_online_file, 'w')
    file_video = open(path_of_video_file, 'w')

valve_idx = 0

thr = 50 # threshold for the detection positions of the aprilTags.

# Because messages might have different clock headers i will have the clock of this node only.
class DataRecorderOnline(Node):

    def __init__(self):
        super().__init__('data_recorder_online')
        self.t_system = time.time() # also have a timer for the whole system.
        self.t_init = self.t_system
        self.t_pressure = 0
        self.t_aprilTag = self.t_system
        self.subsctiption_aprilTag = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.aprilTag_callback,
            10)
        self.subscription_pressure = self.create_subscription(
            FluidPressures,
            '/vtem_control/output_pressures',
            self.pressure_callback,
            10
        )
        self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(RecordingMsg, '/data_recorder_online/info', 10)
        if write_to_file:
            file_pressure.write("t_system[s]," + "dt[s]," + "pressure[mBar]\n") # write the header of the file.
            file_aprilTag.write("t_system[s]," + "dt[s]," + "id," + "position_x[m]," + "position_y[m]\n") # write the header of the file.
            file_online.write(
                "t_system[s],pressure[mBar],"
                "x0[m],y0[m],xe[m],ye[m],">
            "dx[m],dy[m],angle[rad],dangle[rad/s],"
            "FOUND_INITIAL,FOUND_ENDPOINT,INITIAL_POINT\n"
            )
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v'
            self.file_video = cv2.VideoWriter(
                path_of_video_file,  # filename
                fourcc,
                20.0,          # FPS
                (640, 480)     # frame size
            )
        self.bridge = CvBridge()

        self.interval_pressures = []
        self.x0, self.y0 = 488, 155 #280, 79
        self.xe, self.ye = 0, 0
        self.dx, self.dy = 0, 0
        self.angle, self.dangle = 0, 0
        self.FOUND_INITIAL = False
        self.FOUND_ENDPOINT = False
        self.FOUND_INITIAL_now = False
        self.FOUND_ENDPOINT_now = False
        self.avg_pressure = 0.0
        self.initial_point_flag = True


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (640, 480))

        if write_to_file:
            self.file_video.write(frame)

        cv2.imshow("Camera with markers", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            if write_to_file:
                self.file_video.release()

    def pressure_callback(self, msg):
        self.t_system = time.time()
        self.get_logger().info(f'[INFO] Pressure message received. Received pressure: {msg.data[valve_idx].fluid_pressure/100:.6f} mBar')
        if write_to_file:
            file_pressure.write(f"{self.t_system:.6f}, {self.t_system - self.t_pressure:.6f}, {msg.data[valve_idx].fluid_pressure/100:.6f}\n")  # in mBar.
        self.t_pressure = self.t_system
        self.curr_pressure = np.float32(msg.data[valve_idx].fluid_pressure/100)
        self.interval_pressures.append(np.float32(msg.data[valve_idx].fluid_pressure/100))

    def aprilTag_callback(self, msg):
        self.FOUND_INITIAL_now = False
        self.FOUND_ENDPOINT_now = False
        self.t_system = time.time()
        if (self.initial_point_flag):
            self.FOUND_INITIAL, self.FOUND_INITIAL_now = True, True
            if (len(msg.detections) == 0):
                if self.FOUND_ENDPOINT:
                    self.get_logger().info(f'[WARN] No Endpoint AprilTags detected. Using previous position :[{self.xe}, {self.ye}]\n')
                    self.dx, self.dy, self.dangle = 0, 0, 0
                else:
                    self.get_logger().info('[WARN] No Endpoint AprilTags detected. Endpoint not initialized yet. \n')
            else:
                self.get_logger().info(f'[INFO] AprilTag message received. AprilTagID: {msg.detections[0].id}' )
                # save the detections.
                for i, detection in enumerate(msg.detections): # in this case there will only be one.
                    # small print verification of the detection.
                    self.get_logger().info('[INFO] AprilTag detected, id: "%i", Position: [%f, %f]' % ( \
                        detection.id, detection.centre.x, detection.centre.y))
                    if write_to_file:
                        file_aprilTag.write(f"{self.t_system:.6f}, {self.t_system - self.t_aprilTag:.6f}, {detection.id}, {detection.centre.x:.6f}, {detection.centre.y:.6f}\n")

                    if np.linalg.norm([detection.centre.x - self.x0, detection.centre.y - self.y0]) < 20:
                        continue
                    # dc = [np.linalg.norm((d.centre.x - self.x0, d.centre.y - self.y0)) for d in msg.detections]
                    cx = msg.detections[i].centre.x
                    cy = msg.detections[i].centre.y
                    dt = self.t_system - self.t_aprilTag
                    new_angle = np.arctan2(-(self.y0 - cy), self.x0 - cx)
                    # print()
                    if not self.FOUND_ENDPOINT:        
                        self.FOUND_ENDPOINT = True
                        self.FOUND_ENDPOINT_now = True
                        self.angle = new_angle
                        self.dangle, self.dx, self.dy = 0, 0, 0
                    else:
                        self.dx = (cx - self.xe) / dt
                        self.dy = (cy - self.ye) / dt
                        self.dangle = (new_angle - self.angle) / dt
                        self.angle = new_angle
                        self.FOUND_ENDPOINT_now = True
                        
                    self.get_logger().info(f'[INFO] Found Endpoint ApriTag in position : [{self.xe}, {self.ye}]. \n')
                    self.xe, self.ye = cx, cy

        else:
            if (len(msg.detections) == 0):
                if self.FOUND_INITIAL:
                    self.get_logger().info(f'[WARN] No Base AprilTags detected. Using previous position :[{self.x0}, {self.y0}]\n')
                else:
                    self.get_logger().info('[WARN] No Base AprilTags detected. Base not initialized yet. \n')

                if self.FOUND_ENDPOINT:
                    self.get_logger().info(f'[WARN] No Endpoint AprilTags detected. Using previous position :[{self.xe}, {self.ye}]\n')
                    self.dx, self.dy, self.dangle = 0, 0, 0
                else:
                    self.get_logger().info('[WARN] No Endpoint AprilTags detected. Endpoint not initialized yet. \n')
            else :
                self.get_logger().info(f'[INFO] AprilTag message received. AprilTagID: {msg.detections[0].id}' )
                # save the detections.
                for detection in msg.detections:
                    # small print verification of the detection.
                    self.get_logger().info('[INFO] AprilTag detected, id: "%i", Position: [%f, %f]' % ( \
                        detection.id, detection.centre.x, detection.centre.y))
                    file_aprilTag.write(f"{self.t_system:.6f}, {self.t_system - self.t_aprilTag:.6f}, {detection.id}, {detection.centre.x:.6f}, {detection.centre.y:.6f}\n")

            

                distances = np.zeros(len(msg.detections),dtype=np.float32)
                centers = []
                for i, d in enumerate(msg.detections):
                    centers.append((d.centre.x, d.centre.y))
                    distances[i] = np.linalg.norm((self.x0 - d.centre.x, self.y0 - d.centre.y))

                # ----------------------------
                # BASE
                # ----------------------------
                base_idx = None
                if not self.FOUND_INITIAL:
                    self.x0, self.y0 = centers[0]
                    self.FOUND_INITIAL = True
                    self.FOUND_INITIAL_now = True
                    self.get_logger().info('[INFO] Found Base AprilTag in initial position: [%f, %f]' % (self.x0, self.y0))
                    base_idx = 0
                elif distances.min() < thr:
                    base_idx = np.argmin(distances)
                    self.x0, self.y0 = centers[base_idx]
                    self.FOUND_INITIAL_now = True
                    self.get_logger().info('[INFO] Found Base AprilTag in position: [%f, %f]' % (self.x0, self.y0))

                # ----------------------------
                # ENDPOINT
                # ----------------------------
                ep_canditates = [] 
                for i, d in enumerate(distances):
                    if i == base_idx :
                        continue
                    if d > thr:
                        ep_canditates.append(centers[i])
                dt = self.t_system - self.t_aprilTag
                if len(ep_canditates) == 0:
                    if self.FOUND_ENDPOINT:
                        self.get_logger().info(f'[WARN] No Endpoint AprilTags detected. Using previous position :[{self.xe}, {self.ye}]\n')
                        # in the beggining you do not have time initialization.
                        new_angle = np.arctan2(-(self.y0 - self.ye), self.x0 - self.xe) # could have found new base position. ( Just to be sure. )
                        
                        self.dx = 0
                        self.dy = 0
                        self.dangle = (new_angle - self.angle) / dt
                        self.angle = new_angle
                    else:
                        self.get_logger().info(f'[WARN] No Endpoint AprilTags detected. Endpoint not initialized yet. \n')
                else:
                    dc = [np.linalg.norm((c[0] - self.x0, c[1] - self.y0)) for c in ep_canditates]
                    cx, cy = ep_canditates[int(np.argmax(dc))] # canditate further from the base.
                    new_angle = np.arctan2(-(self.y0 - cy),self.x0 - cx)
                    if not self.FOUND_ENDPOINT:        
                        self.FOUND_ENDPOINT = True
                        self.FOUND_ENDPOINT_now = True
                        self.angle = new_angle
                        self.dangle, self.dx, self.dy = 0, 0, 0
                    else:
                        self.dx = (cx - self.xe) / dt
                        self.dy = (cy - self.ye) / dt
                        self.dangle = (new_angle - self.angle) / dt
                        self.angle = new_angle
                        self.FOUND_ENDPOINT_now = True
                        
                    self.get_logger().info(f'[INFO] Found Endpoint ApriTag in position : [{self.xe}, {self.ye}]. \n')
                    self.xe, self.ye = cx, cy

        # ----------------------------
        # LOGGING (one line per frame)
        # ----------------------------
        # first get pressures that were measured in the interval and it there are more than one take the average.
        if self.interval_pressures:
            self.avg_pressure = np.mean(self.interval_pressures)
            self.get_logger().info(f'[INFO] Average Pressure within the AprilTag detection interval: {self.avg_pressure:.6f}. \n')
        else:
            self.get_logger().info(f'[ERROR] No pressure was sent within the time interval of the detections. \n')

        self.interval_pressures.clear()

        if write_to_file:
            file_online.write(
                f"{self.t_system - self.t_init:.6f}, "
                f"{self.avg_pressure:.6f}, "
                f"{self.x0:.6e}, {self.y0:.6e}, "
                f"{self.xe:.6e}, {self.ye:.6e}, "
            f"{self.dx:.6e}, {self.dy:.6e}, "
            f"{self.angle:.6e}, {self.dangle:.6e}, "
            f"{self.FOUND_INITIAL_now}, {self.FOUND_ENDPOINT_now}, {self.initial_point_flag}\n"
        )
        msg = RecordingMsg(
            time=float(self.t_system),
            avg_pressure=float(self.curr_pressure),

            x0=float(self.x0),
            y0=float(self.y0),

            xe=float(self.xe),
            ye=float(self.ye),

            dx=float(self.dx),
            dy=float(self.dy),

            angle=float(self.angle),
            dangle=float(self.dangle),

            found_initial=bool(self.FOUND_INITIAL_now),
            found_endpoint=bool(self.FOUND_ENDPOINT_now),
            initial_point_flag=bool(self.initial_point_flag)
        )

        self.publisher_.publish(msg)
        self.t_aprilTag = self.t_system


        



def main(args=None):
    rclpy.init(args=args)

    data_recorder = DataRecorderOnline()

    rclpy.spin(data_recorder)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    data_recorder.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()