import rclpy
import time
import numpy as np
from rclpy.node import Node
from event_camera_msgs.msg import EventPacket
from event_camera_py import Decoder
import dv_processing as dv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import timedelta

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import cv2 as cv
import os


# os.makedirs("./data/event_tracker/aprilTag", exist_ok=True)
# os.makedirs("./data/event_tracker/events", exist_ok=True)

# path_of_aprilTag_file = "./data/event_tracker/aprilTag/aprilTag_output" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
# path_of_events_file = "./data/event_tracker/events/events_output" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
# file_aprilTag = open(path_of_aprilTag_file, 'w') # open a file to write the data.
# file_events = open(path_of_events_file, 'w')

thr = 50 # threshold for the detection positions of the aprilTags.

# Because messages might have different clock headers i will have the clock of this node only.
class EventRecorderOnline(Node):

    def __init__(self):
        super().__init__('event_recorder')
        self.t_system = time.time() # also have a timer for the whole system.
        self.t_init = self.t_system
        self.t_pressure = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(
            EventPacket,
            '/event_camera/events',
            self.events_callback,
            qos
        )

        self.create_subscription(
            Image,
            '/event_camera/image_raw',
            self.image_callback,
            1
        )

        self.decoder = Decoder()
        self.bridge = CvBridge()
        self.get_logger().info('[INFO] Event recorder initialized.\n')
        self.slicer = dv.EventStreamSlicer()
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=1), self.print_events)
        self.filter = dv.noise.BackgroundActivityNoiseFilter((346, 260), backgroundActivityDuration=timedelta(milliseconds=1))
        self.events, self.filtered = None, None



    def print_events(self, events: dv.EventStore):
        if len(events) == 0:
            return

        print(f"Number of events in the last 1ms: {len(events)}")
        self.events = events
        self.filter.accept(events)
        self.filtered = self.filter.generateEvents()
        # Print out the reduction factor, which indicates the percentage of discarded events
        print(f"Filter reduced number of events by a factor of {self.filter.getReductionFactor()}")


    def events_callback(self, msg):
        self.get_logger().info('[INFO] Events detected.\n')
        self.decoder.decode(msg)
        cd_events = self.decoder.get_cd_events()
        # cd_events.sort(order="t")
        self.conversion_layer(cd_events)

    def conversion_layer(self, cd_events):
        store = dv.EventStore()
        for event in cd_events:
            store.push_back(self.decoder.get_start_time() + event['t'], event['x'], event['y'], event['p'])
        # first_ts = store[0].timestamp()
        # last_ts = store[len(store) - 1].timestamp()
        # print("slice:", first_ts, last_ts)
        self.slicer.accept(store)

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.events is not None:
            for event in self.events:
                    cv.circle(frame, (event.x(), event.y()), radius=2, color=(0, 0, 255), thickness=-1)
        if self.filtered is not None:
            for event in self.filtered:
                cv.circle(frame, (event.x(), event.y()), radius=2, color=(255, 255, 0), thickness=-1)
        res = cv.resize(frame, (600,600))
        cv.imshow("Camera with markers", res)
        cv.waitKey(1)



def main(args=None):
    rclpy.init(args=args)

    event_recorder = EventRecorderOnline()

    rclpy.spin(event_recorder)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    event_recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()