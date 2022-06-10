import os
import argparse
import rosbag
from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2



def mkdir_p(path):
  try:
    os.makedirs(path)
  except:
    pass


parser = argparse.ArgumentParser("Extract camera and IMU data from a rosbag into a folder of files")
parser.add_argument("--bag", type=str, help="rosbag containing data")
parser.add_argument("--cam_topic", type=str, help="RGB camera topic name", default="/cam0")
parser.add_argument("--imu_topic", type=str, help="IMU topic name", default="/imu0")
parser.add_argument("--sequence_name", type=str, help="name of the sequence being extracted; will also be name of output folder")
args = parser.parse_args()

# CV bridge
bridge = CvBridge()

# Open rosbag file
bagfile = rosbag.Bag(args.bag, mode='r')
sequence_name = args.sequence_name
# make sure output directories exist
imu_dir = os.path.join(sequence_name, "imu0")
rgb_dir = os.path.join(sequence_name, "cam0")
mkdir_p(imu_dir)
mkdir_p(rgb_dir)

# Open IMU data file and write header line
imu_filename = os.path.join(imu_dir, 'data.csv')
imu_file = open(imu_filename, 'w')
imu_file.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")


# read IMU messages
for _,msg,t in bagfile.read_messages(topics=args.imu_topic):
  imu_file.write("{},{},{},{},{},{},{}\n".format(
    t,
    msg.angular_velocity.x,
    msg.angular_velocity.y,
    msg.angular_velocity.z,
    msg.linear_acceleration.x,
    msg.linear_acceleration.y,
    msg.linear_acceleration.z
  ))
imu_file.close()

# read RGB images
for _,msg,t in bagfile.read_messages(topics=args.cam_topic):
  try:
    opencv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
  except CvBridgeError as e:
    print(e)
  img_filename = os.path.join(rgb_dir, "{}.png".format(t))
  cv2.imwrite(img_filename, opencv_img)

