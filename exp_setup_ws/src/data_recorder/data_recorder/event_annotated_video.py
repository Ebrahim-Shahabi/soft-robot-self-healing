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
from apriltag_msgs.msg import AprilTagDetectionArray
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
thr_event = 20
# Because messages might have different clock headers i will have the clock of this node only.
class EventVideoOnline(Node):

    def __init__(self):
        super().__init__('event_video')
        self.t_system = time.time() # also have a timer for the whole system.
        self.width = 346
        self.height = 260
        self.resolution = (self.width, self.height)

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

        self.subsctiption_aprilTag = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.aprilTag_callback,
            10)

        self.create_subscription(
            Image,
            '/event_camera_synced/image_raw',
            self.image_callback,
            1
        )

        
        self.decoder = Decoder() # necessary to decode messages from ros2 to usable events.
        self.bridge = CvBridge() # just for the video.

        # EventStream slicer
        self.slicer = dv.EventStreamSlicer()
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=5), self.print_events) # keep only events in the last 1ms.

        # filters 
        self.filter = dv.EventFilterChain()

        # remove background noise that may occur due to analog circuit noise.
        self.background_filter = dv.noise.BackgroundActivityNoiseFilter(self.resolution, backgroundActivityDuration=timedelta(milliseconds=5))
        self.filter.addFilter(self.background_filter)


        # keep only a rectangle of the image since everything is going to happen there.
        self.mask_filter = dv.EventRegionFilter((0, 0, self.width//2, self.height))
        self.filter.addFilter(self.mask_filter)

        self.polarity_filter = dv.EventPolarityFilter(False)
        self.filter.addFilter(self.polarity_filter)

        # tracker
        bandwidth = 200 # this is pixel search radius around a point

        time_window = timedelta(milliseconds=1) # window of time used to compute the time surface used for the tracking update

        self.mean_shift = dv.features.MeanShiftTracker(self.resolution, bandwidth, time_window)


        self.events, self.background_filtered, self.mask_filtered, self.mean_shift_tracks, self.polarity_filtered = None, None, None, None, None
        self.filtered_events = None
        self.center = None
        self.april = None
        self.toggle = False
        # :)
        self.get_logger().info('[INFO] Event recorder initialized.\n')


    def print_events(self, events: dv.EventStore):
        if len(events) == 0:
            return

        print(f"Number of events in the last 1ms: {len(events)}")
        self.events = events
        # self.background_filter.accept(events) # background noise filter
        # self.background_filtered = self.background_filter.generateEvents() 


        # self.mask_filter.accept(events) # mask filter
        # self.mask_filtered = self.mask_filter.generateEvents() 

        # self.polarity_filter.accept(events) # polarity filter
        # self.polarity_filtered = self.polarity_filter.generateEvents()

        self.filter.accept(events) # filter chain
        self.filtered_events = self.filter.generateEvents()

        mean_shift = dv.cluster.mean_shift.MeanShiftEventStoreAdaptor(self.filtered_events, 100, 0.01, 1000, 20)
        

        # Perform the mean-shift, the algorithm returns a tuple of center coordinates, labels, count of event in the
        # cluster, and variances of the cluster
        centers, _, counts, _ = mean_shift.fit()
        # Let's assign the cluster size to the response value of the center keypoint
        for i in range(len(centers)):
            centers[i].response = counts[i]
        # Sort the estimated centers by the number of events in cluster, the values are sorted in descending order
        centers.sort(key=lambda a: a.response, reverse=True)

        best_center = None

        if len(centers) > 0:
            if self.center is not None:
                dists = [
                    np.hypot(c.pt[0] - self.center.pt[0], c.pt[1] - self.center.pt[1])
                    for c in centers
                ]
                best_idx = int(np.argmin(dists))
                best_dist = dists[best_idx]

                if best_dist < thr_event:
                    best_center = centers[best_idx]
                else:
                    best_center = centers[best_idx]
            else:
                # no previous center, initialize immediately
                best_center = centers[0]

        if self.toggle:
            if best_center is not None:
                dx = best_center.pt[0] - float(self.april.x)
                dy = best_center.pt[1] - float(self.april.y)

                if np.hypot(dx, dy) < 20.0:
                    best_center.pt = (float(self.april.x), float(self.april.y))
                    self.center = best_center
            else:
                best_center = dv.TimedKeyPoint(
                    pt=(float(self.april.x), float(self.april.y)),
                    size=1.0,
                    angle=0.0,
                    response=1.0,
                    octave=0,
                    class_id=0,
                    timestamp=0
                )
            self.toggle = False

        if best_center is not None:
            self.center = best_center
        

        # self.mean_shift.accept(self.filtered_events) # mean shift tracker
        # self.mean_shift_tracks = self.mean_shift.runTracking()

        # Print out the reduction factor, which indicates the percentage of discarded events
        print(f"Filter reduced number of events by a factor of {self.filter.getReductionFactor()}")


    def aprilTag_callback(self, msg):
        self.t_system = time.time()
        if (len(msg.detections) == 0):
            self.get_logger().info('[WARN] No AprilTags detected.')
            self.toggle = False
        else :
            self.get_logger().info(f'[INFO] AprilTag message received. AprilTagID: {msg.detections[0].id}' )
            # n = len(msg.detections)
            for detection in msg.detections:
                # small print verification of the detection.
                self.get_logger().info('[INFO] AprilTag detected, id: "%i", Position: [%f, %f]' % ( \
                    detection.id, detection.centre.x, detection.centre.y))
            self.april = msg.detections[0].centre
            self.toggle = True

        self.t_aprilTag = self.t_system

    def events_callback(self, msg):
        self.get_logger().info('[INFO] Events detected.\n')
        self.decoder.decode(msg) # decode the messages
        cd_events = self.decoder.get_cd_events() # get the events in a usable format.
        cd_events.sort(order="t")

        # convert to event format and pass them to the slicer to keep only events in the last 1ms.
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
        # filtered = frame.copy()
        # mask_filtered = frame.copy()
        # mean_shift_tracking = frame.copy()
        # frame_polarity = frame.copy()
        frame_filtered = frame.copy()

        if self.events is not None:
            for event in self.events:
                    cv.circle(frame, (event.x(), event.y()), radius=2, color=(0, 0, 255), thickness=-1)

        # if self.background_filtered is not None:
        #     for event in self.background_filtered:
        #         cv.circle(filtered, (event.x(), event.y()), radius=2, color=(255, 255, 0), thickness=-1)

        # if self.mask_filtered is not None:
        #     for event in self.mask_filtered:
        #         cv.circle(mask_filtered, (event.x(), event.y()), radius=2, color=(0, 255, 0), thickness=-1)

        # if self.polarity_filtered is not None:
        #     for event in self.polarity_filtered:
        #         cv.circle(frame_polarity, (event.x(), event.y()), radius=2, color=(255, 0, 255), thickness=-1)

        if self.filtered_events is not None:
            for event in self.filtered_events:
                cv.circle(frame_filtered, (event.x(), event.y()), radius=2, color=(255, 255, 255), thickness=-1)
        # Draw markers on each of the track coordinates
        # if (self.mean_shift_tracks is not None) and (len(self.mean_shift_tracks.keypoints) > 0):
        #     for index in range(len(self.mean_shift_tracks.keypoints)):
        #         track = self.mean_shift_tracks.keypoints[index]
        #         cv.drawMarker(mean_shift_tracking, (int(track.pt[0]), int(track.pt[1])), dv.visualization.colors.red(), cv.MARKER_CROSS,
        #                     20, 2)
        #         cv.circle(mean_shift_tracking, (int(track.pt[0]), int(track.pt[1])), radius=100, color=dv.visualization.colors.red(), thickness=2)
            
        if self.center is not None:
            cv.drawMarker(frame_filtered, (int(self.center.pt[0]), int(self.center.pt[1])), dv.visualization.colors.green(), cv.MARKER_CROSS,
                            20, 2)
            cv.circle(frame_filtered, (int(self.center.pt[0]), int(self.center.pt[1])), radius=100, color=dv.visualization.colors.green(), thickness=2)
        
        res = cv.resize(frame, (400,400))
        # res1 = cv.resize(filtered, (400,400))
        # res2 = cv.resize(mask_filtered, (400,400))
        # res4 = cv.resize(frame_polarity, (400,400))
        res5 = cv.resize(frame_filtered, (400,400))
        # res3 = cv.resize(mean_shift_tracking, (400,400))


        cv.imshow("Original", res)
        # cv.imshow("Filtered events", res1)
        # cv.imshow("Mask filtered events", res2)
        # cv.imshow("Mean shift tracking", res3)
        # cv.imshow("Polarity filtered events", res4)
        cv.imshow("Tracking", res5)
        cv.waitKey(1)



def main(args=None):
    rclpy.init(args=args)

    event_video = EventVideoOnline()

    rclpy.spin(event_video)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    event_video.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()