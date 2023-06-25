#!/usr/bin/env python3
import os
import sys
import cv2
import math
import time
import queue
import random
import threading
import numpy as np
import rospy
import hiwonder
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse
from jetmax_control.msg import SetServo
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "alphabetically"
IMAGE_SIZE = 640, 480  # Enter image size

CHARACTERS_ENGINE_PATH = os.path.join(sys.path[0], 'characters_v5_160.trt')  # letter model path
CHARACTER_LABELS = tuple([i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])  # alphabetic taxonomy name
CHARACTER_NUM = 26

NUMBERS_ENGINE_PATH = os.path.join(sys.path[0], 'numbers_v5_160.trt')  # digital model path
NUMBERS_LABELS = tuple([i for i in '0123456789+-*/='])  # name of numeric categorical value
NUMBERS_NUM = 15  # Total number of categories

TRT_INPUT_SIZE = 160  # trt input size
# Random color, used for result boxing
COLORS = tuple([tuple([random.randint(10, 255) for j in range(3)]) for i in range(CHARACTER_NUM + NUMBERS_NUM)])

GOAL_POSITIONS = (
    (-230, -70, 55, 17), (-230, -25, 55, 7), (-230, 20, 55, -3), (-230, 65, 55, -13),
    (-185, -70, 55, 20), (-185, -25, 55, 10), (-185, 20, 55, -3), (-185, 65, 55, -13),
    (-140, -70, 55, 30), (-140, -25, 55, 15), (-140, 20, 55, -5), (-140, 65, 55, -18),
)


def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invR
        worldPtPlane = np.dot(inv_r, worldPtCam)  # 3*3 dot 3*1 = 3*1
        # zc
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


class Alphabetically:
    def __init__(self):
        self.running_mode = 0
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.moving_count = 0
        self.count = 0
        self.lock = threading.RLock()
        self.fps_t0 = time.time()
        self.fps = 0
        self.camera_params = None
        self.enable_moving = False
        self.K = None
        self.R = None
        self.T = None

    """
    Reset required variables
    """

    def reset(self):
        self.running_mode = 0
        self.enable_moving = False
        self.moving_box = None
        self.moving_count = 0
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.fps_t0 = time.time()
        self.fps = 0

    """
    Read camera internal reference
    """

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/card_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)


"""
moving card
"""


def moving():
    try:
        c_x, c_y, cls_id, cls_conf = state.moving_box  # Get the position of the card to be moved on the graph
        cur_x, cur_y, cur_z = jetmax.position  # The current coordinates of the nozzle

        # Calculate the coordinates of the nozzle center of the card in the real world (the nozzle is calibrated when the external parameters are calibrated)
        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]

        t = math.sqrt(x * x + y * y + 120 * 120) / 120  # Calculate the distance of the card position, by distance/speed=time, calculate how long it takes to reach the card position
        new_x, new_y = cur_x + x, cur_y + y + 15  # Calculate the coordinates of the card position relative to the base of the robot arm
        nn_new_x = new_x - 25
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi  # Calculate the deflection angle of the mechanical arm relative to the central axis after reaching the new position. Later, when we place the card, we will rotate the small servo to deflect it back
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass
        # The robotic arm moves to the card position step by step
        jetmax.set_position((nn_new_x, new_y, 70), t)
        rospy.sleep(t)
        jetmax.set_position((new_x, new_y, 70), 0.3)
        rospy.sleep(t + 0.6)

        # drop, suck
        sucker.set_state(True)
        jetmax.set_position((new_x, new_y, 50 - 4), 0.8)  # -5
        rospy.sleep(0.85)

        # Look up the table according to the order to get where to put it, and raise the robotic arm
        x, y, z, angle = GOAL_POSITIONS[state.moving_count]
        cur_x, cur_y, cur_z = jetmax.position
        jetmax.set_position((cur_x, cur_y, 140), 0.8)  # 100
        rospy.sleep(0.8)

        # Control the small servo to rotate the card so that the angle of the card can be the same as the angle of placement before suction
        hiwonder.pwm_servo1.set_position(angle + arm_angle, 0.1)  # 90 + angle
        cur_x, cur_y, cur_z = jetmax.position
        # Calculate the distance from the current position to the target position to control the speed
        t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 160  # 150
        # Control the robotic arm to reach the target
        jetmax.set_position((x, y, 120), t)   # z + 30
        rospy.sleep(t)

        # Control the robotic arm to reach the target position
        jetmax.set_position((x, y, z), 0.8)
        rospy.sleep(0.8)

        # freed
        sucker.release(3)
        # lift up
        jetmax.set_position((x, y, z + 50), 0.8)  # 30
        rospy.sleep(0.1)
        # small servo recovery
        hiwonder.pwm_servo1.set_position(90, 0.5)  # 0.4
        rospy.sleep(0.8)

    finally:
        sucker.release(3)
        cur_x, cur_y, cur_z = jetmax.position
        # Calculate the distance between the current position and the default position to control the speed
        t = math.sqrt((cur_x - jetmax.ORIGIN[0]) ** 2 + (cur_y - jetmax.ORIGIN[1]) ** 2) / 120
        # back to default position
        jetmax.go_home(t)
        # Small servo recovery, multiple recovery is because the above recovery may not be executed due to exception
        hiwonder.pwm_servo1.set_position(90, 0.2)
        rospy.sleep(t + 0.2)
        with state.lock:  # Clear the running logo and let the program continue to the next operation
            state.moving_box = None
            state.moving_count += 1  # Accumulate the number of placements to record where to put them
            if state.moving_count >= len(GOAL_POSITIONS):
                state.moving_count = 0
            state.runner = None
        print("FINISHED")


"""
Alphabet Image Processing
"""


def image_proc_chars(img_in):
    if state.runner is not None:  # No recognition if there is a moving action in progress
        return img_in
    result_image = img_in
    outputs = yolov5_chars.detect(np.copy(img_in))  # Call yolov5 for recognition
    boxes, confs, classes = yolov5_chars.post_process(img_in, outputs, 0.65)  # Post-process the network output of yolov5 to get the final result
    cards = []
    width, height = IMAGE_SIZE
    # Iterate through the identified results
    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        # The recognized card is framed in the screen and displayed
        cv2.putText(img_in, CHARACTER_LABELS[cls_id] + " " + str(float(cls_conf))[:4],
                    (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls_id], 2)
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls_id], 3)

    if len(cards) == 0:  # not recognized
        state.count = 0
        state.moving_box = None
    else:  # recognized
        if state.moving_box is None:  # The previous frame was not recognized, and multiple recognitions are required to prevent false triggering
            moving_box = min(cards, key=lambda x: x[4])  # The smallest id among all recognized cards
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2  # center of box
            state.moving_box = c_x, c_y, cls_id, cls_conf  # Record the location of this card and other parameters
            cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)  # draw the center
            state.count = 0
        else:  # The previous frame was recognized
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            # Iterate through all newly recognized cards
            cards = [((x1 + x2) / 2,
                      (y1 + y2) / 2,
                      cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in cards]
            # Find the closest card to the target card recorded in the previous frame
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in cards]
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            # draw the center twice
            cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id and state.enable_moving:  # If the id is the same twice, it is correctly identified
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 20:  # After 20 consecutive frames are recognized, the card will be moved
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.runner.start()
            else:
                state.moving_box = None
    return result_image


"""
Digital Card Identification
The overall process is the same as the letter card, but the model used is different
"""


def image_proc_nums(img_in):
    if state.runner is not None:
        return img_in
    result_image = img_in
    outputs = yolov5_nums.detect(np.copy(img_in))
    boxes, confs, classes = yolov5_nums.post_process(img_in, outputs, 0.65)
    cards = []
    width, height = IMAGE_SIZE

    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        result_image = cv2.putText(img_in, NUMBERS_LABELS[cls_id] + " " + str(float(cls_conf))[:4],
                                   (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                   COLORS[CHARACTER_NUM + cls_id], 2)
        result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)),
                                     COLORS[CHARACTER_NUM + cls_id], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = min(cards, key=lambda x: x[4])
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, cls_id, cls_conf
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            state.count = 0
        else:
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            cards = [((x1 + x2) / 2,
                      (y1 + y2) / 2,
                      cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in cards]
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in cards]
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            result_image = cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id and state.enable_moving:
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 20:
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.runner.start()
            else:
                state.moving_box = None
    return result_image


def show_fps(img, fps):
    """display fps"""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


"""
General Image Processing Functions
It will decide whether to call digital card recognition or letter card recognition or no recognition according to the setting state
This function will be called in the while loop instead of being triggered directly by the callback of the ros topic because pycuda requires that the related calls must be in the same thread
So the ros topic callback will only put the received image into a queue and will not process the image directly. image_proc will cycle through the queue
"""


def image_proc():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    with state.lock:
        if state.running_mode == 1:
            result_img = image_proc_chars(image)
        elif state.running_mode == 2:
            result_img = image_proc_nums(image)
        else:
            result_img = image
    # fps cal
    fps_t1 = time.time()
    fps_cur = (1.0 / (fps_t1 - state.fps_t0))
    state.fps = fps_cur if state.fps == 0.0 else (state.fps * 0.8 + fps_cur * 0.2)
    state.fps_t0 = fps_t1
    # show_fps(result_img, state.fps)
    #
    rgb_image = result_img.tostring()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)


"""
The callback of the camera screen topic
Only received frames will be pushed into the queue
"""


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass


"""
start service
The program enters the ready state, ready to run, but does not recognize

"""


def enter_func(msg):
    rospy.loginfo("enter")
    exit_func(msg)
    jetmax.go_home()
    state.reset()
    state.load_camera_params()
    state.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    return TriggerResponse(success=True)


"""
quit service
The program unsubscribes from the camera topic
"""


def exit_func(msg):
    rospy.loginfo("exit")
    try:
        state.heartbeat_timer.cancel()
    except:
        pass
    with state.lock:
        state.running_mode = 0
        # unsubscribe
        if isinstance(state.image_sub, rospy.Subscriber):
            rospy.loginfo('unregister image')
            state.image_sub.unregister()
            state.image_sub = None
    # Wait for the running move to complete
    if isinstance(state.runner, threading.Thread):
        state.runner.join()
    # Call the service to return the robotic arm to its initial position
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    return TriggerResponse(success=True)


def heartbeat_timeout_cb():
    rospy.loginfo("heartbeat timeout. exiting...")
    rospy.ServiceProxy('/%s/exit' % ROS_NODE_NAME, Trigger)


def heartbeat_srv_cb(msg: SetBoolRequest):
    try:
        state.heartbeat_timer.cancel()
    except:
        pass
    rospy.logdebug("Heartbeat")
    if msg.data:
        state.heartbeat_timer = threading.Timer(5, heartbeat_timeout_cb)
        state.heartbeat_timer.start()
    return SetBoolResponse(success=msg.data)


"""
Begin Alphabet Card Recognition
"""


def set_char_running_cb(msg):
    with state.lock:
        if msg.data:
            # If the current mode is not letter recognition mode (1), it is set to letter recognition mode
            if state.running_mode != 1:
                state.running_mode = 1
                state.moving_count = 0
                state.enable_moving = False
            # If it is already in the letter recognition mode, switch the recognition switch
            else:
                if state.enable_moving:
                    state.enable_moving = False
                else:
                    state.enable_moving = True
        else:
            if state.running_mode == 1:
                state.running_mode = 0
                state.enable_moving = False
    return [True, '']


"""
Start to recognize digital cards
"""


def set_num_running_cb(msg):
    with state.lock:
        if msg.data:
            # If the current operating mode is not the recognition number, change to the recognition number
            if state.running_mode != 2:
                state.running_mode = 2
                state.moving_count = 0
                state.enable_moving = False
            # If it is an identification number, control the switch identification
            else:
                if state.enable_moving:
                    state.enable_moving = False
                else:
                    state.enable_moving = True
        else:
            if state.running_mode == 2:
                state.running_mode = 0
                state.enable_moving = False
    return [True, '']


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)  # initialize node
    state = Alphabetically()  # Initialize related parameters
    state.load_camera_params()  # Read camera internal reference
    if state.camera_params is None:
        rospy.logerr("Can not load camera parameters")
        sys.exit(-1)
    # Build letter and number recognizer
    yolov5_chars = Yolov5TensorRT(CHARACTERS_ENGINE_PATH, TRT_INPUT_SIZE, CHARACTER_NUM)
    yolov5_nums = Yolov5TensorRT(NUMBERS_ENGINE_PATH, TRT_INPUT_SIZE, NUMBERS_NUM)
    # image queue
    image_queue = queue.Queue(maxsize=1)
    # robot control interface
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()

    # Subscription and publication registration of related topics
    image_pub = rospy.Publisher('/%s/image_result' % ROS_NODE_NAME, Image, queue_size=1)  # register result image pub
    enter_srv = rospy.Service('/%s/enter' % ROS_NODE_NAME, Trigger, enter_func)
    exit_srv = rospy.Service('/%s/exit' % ROS_NODE_NAME, Trigger, exit_func)
    char_running_srv = rospy.Service('/%s/set_running_char' % ROS_NODE_NAME, SetBool, set_char_running_cb)
    num_running_srv = rospy.Service('/%s/set_running_num' % ROS_NODE_NAME, SetBool, set_num_running_cb)
    heartbeat_srv = rospy.Service('/%s/heartbeat' % ROS_NODE_NAME, SetBool, heartbeat_srv_cb)

    while True:
        try:
            image_proc()  # Cycle through images
            if rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            rospy.signal_shutdown("custom shutdown")
            break
