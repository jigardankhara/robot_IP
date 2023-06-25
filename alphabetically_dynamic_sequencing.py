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
from sensor_msgs.msg import Image
import hiwonder
from yolov5_tensorrt import Yolov5TensorRT

WORD_WANT = tuple([i for i in "HEYGOVIND"])  # word we want to write from robot

ROS_NODE_NAME = "hiwonder_jetmax_aph"
IMAGE_SIZE = 640, 480

CHARACTERS_ENGINE_PATH = os.path.join(sys.path[0], 'models/characters_v5_160.trt')
CHARACTER_LABELS = tuple([i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
CHARACTER_NUM = 26

TRT_INPUT_SIZE = 160
COLORS = tuple([tuple([random.randint(10, 255) for j in range(3)]) for i in range(CHARACTER_NUM)])

TARGET_POSITION = (
    (-230, -115, 55, 27), (-230, -70, 55, 17), (-230, -25, 55, 7),
    (-185, -160, 55, -13), (-185, -115, 55, 20), (-185, -70, 55, 20), (-185, -25, 55, 10), (-185, 20, 55, -3),
    (-185, 65, 55, -13))

yolov5_chars = Yolov5TensorRT(CHARACTERS_ENGINE_PATH, TRT_INPUT_SIZE, CHARACTER_NUM)


class Alphabetically:
    def __init__(self):
        self.lock = threading.RLock()
        self.moving_box = None
        self.index = 0
        self.pos_add = 0
        self.last_target = TARGET_POSITION

        self.runner = None
        self.moving_count = 0
        self.count = 0
        self.fps_t0 = time.time()
        self.fps = 0
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/card_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)


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


def moving():
    c_x, c_y, cls_id, cls_conf = state.moving_box  # Get the position of the card to be moved on the graph
    cur_x, cur_y, cur_z = jetmax.position  # The current coordinates of the nozzle
    print(f'cur_x={cur_x}, cur_y={cur_y}, cur_z={cur_z}')
    # Calculate the coordinates of the nozzle center of the card in the real world (the nozzle is calibrated when the external parameters are calibrated)
    x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
    print(f'camera to world :- x={x}, y={y}, _={_}')
    t = math.sqrt(
        x * x + y * y + 120 * 120) / 120  # Calculate the distance of the card position, by distance/speed=time, calculate how long it takes to reach the card position
    new_x, new_y = cur_x + x, cur_y + y + 15  # Calculate the coordinates of the card position relative to the base of the robot arm
    print(f'new_x={new_x}, new_y={new_y}')
    nn_new_x = new_x - 25
    print(f'nn_new_x={nn_new_x}')
    arm_angle = math.atan(
        new_y / new_x) * 180 / math.pi  # Calculate the deflection angle of the mechanical rm relative to the central axis after reaching the new position. Later, when we place the card, we will rotate the small servo to deflect it back
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
    jetmax.set_position((new_x, new_y, 50 - 5), 0.8)
    rospy.sleep(0.85)

    # Look up the table according to the order to get where to put it, and raise the robotic arm
    x, y, z, angle = TARGET_POSITION[state.moving_count]
    cur_x, cur_y, cur_z = jetmax.position
    jetmax.set_position((cur_x, cur_y, 100), 0.8)
    rospy.sleep(0.8)

    # Control the small servo to rotate the card so that the angle of the card can be the same as the angle of placement before suction
    hiwonder.pwm_servo1.set_position(90 + angle + arm_angle, 0.1)
    cur_x, cur_y, cur_z = jetmax.position
    # Calculate the distance from the current position to the target position to control the speed
    t = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2) / 150
    # Control the robotic arm to reach the target
    jetmax.set_position((x, y, z + 30), t)
    rospy.sleep(t)

    # Control the robotic arm to reach the target position
    jetmax.set_position((x, y, z), 0.8)
    rospy.sleep(0.8)

    # freed
    sucker.release(3)
    # lift up
    jetmax.set_position((x, y, z + 30), 0.8)
    rospy.sleep(0.1)
    # small servo recovery
    hiwonder.pwm_servo1.set_position(90, 0.4)
    rospy.sleep(0.8)

    # sucker.release(3)
    # cur_x, cur_y, cur_z = jetmax.position
    # # Calculate the distance between the current position and the default position to control the speed
    # t = math.sqrt((cur_x - jetmax.ORIGIN[0]) ** 2 + (cur_y - jetmax.ORIGIN[1]) ** 2) / 120
    # back to default position
    jetmax.go_home(t)
    # Small servo recovery, multiple recovery is because the above recovery may not be executed due to exception
    hiwonder.pwm_servo1.set_position(90, 0.2)
    rospy.sleep(t + 0.2)
    with state.lock:  # Clear the running logo and let the program continue to the next operation
        state.moving_box = None
        state.moving_count += 1  # Accumulate the number of placements to record where to put them
        state.index += 1
        if state.moving_count >= len(WORD_WANT):
            state.moving_count = 0
        state.runner = None
    print("FINISHED")


def image_proc(img_in):
    result_image = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)

    if state.runner is not None:
        return result_image

    outputs = yolov5_chars.detect(np.copy(img_in))
    boxes, confs, classes = yolov5_chars.post_process(img_in, outputs, 0.65)

    cards = []
    width, height = IMAGE_SIZE
    if state.index >= len(WORD_WANT):
        state.index = 0
        state.last_target = TARGET_POSITION

    if WORD_WANT[state.index] == '\n':
        state.index += 1
        x, y, z = state.last_target
        y -= 50
        x = TARGET_POSITION[0] - 5 - 40
        state.last_target = x, y, z

    for box, cls_id, cls_conf in zip(boxes, classes, confs):
        x1 = box[0] / TRT_INPUT_SIZE * width
        y1 = box[1] / TRT_INPUT_SIZE * height
        x2 = box[2] / TRT_INPUT_SIZE * width
        y2 = box[3] / TRT_INPUT_SIZE * height
        char = CHARACTER_LABELS[cls_id]
        if char == WORD_WANT[state.index]:
            cards.append((x1, y1, x2, y2, cls_id, cls_conf))
        cv2.putText(result_image, char + " " + str(float(cls_conf))[:4], (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls_id], 2)
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls_id], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
        rospy.logwarn("can not find card '{}'".format(WORD_WANT[state.index]))
    else:
        if state.moving_box is None:
            moving_box = max(cards, key=lambda x: x[
                -1])  # The one with the highest probability among all identified cards
            x1, y1, x2, y2, cls_id, cls_conf = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, cls_id, cls_conf  # save
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            state.count = 0
        else:
            l_c_x, l_c_y, l_cls_id, _ = state.moving_box
            cards = [((x1 + x2) / 2, (y1 + y2) / 2, cls_id, cls_conf) for x1, y1, x2, y2, cls_id, cls_conf in
                     cards]  # Calculate center coordinates
            distances = [math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) for c_x, c_y, _, _ in
                         cards]  # Calculate the center coordinate distance twice
            new_moving_box = min(zip(distances, cards), key=lambda x: x[0])  # Find the smallest distance
            _, (c_x, c_y, cls_id, cls_conf) = new_moving_box
            result_image = cv2.circle(result_image, (int(l_c_x), int(l_c_y)), 1, (0, 255, 0), 30)
            result_image = cv2.circle(result_image, (int(c_x), int(c_y)), 1, (255, 0, 0), 30)
            if cls_id == l_cls_id:  # If the ids recognized twice before and after are the same, the handling classification will be performed. Re-identify if different
                state.moving_box = c_x, c_y, cls_id, cls_conf
                state.count += 1
                if state.count > 5:
                    print("MOVE")
                    state.count = 0
                    state.runner = threading.Thread(target=moving, daemon=True)
                    state.runner.start()
            else:
                state.moving_box = None
    return result_image


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def image_proc_b():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    text_detect=text_process(image)
    # result_img = image_proc(image)
    # fps cal
    fps_t1 = time.time()
    fps_cur = (1.0 / (fps_t1 - state.fps_t0))
    state.fps = fps_cur if state.fps == 0.0 else (state.fps * 0.8 + fps_cur * 0.2)
    state.fps_t0 = fps_t1
    # show_fps(result_img, state.fps)
    cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imshow(ROS_NODE_NAME, result_img)
    cv2.waitKey(1)

def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass


if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    state = Alphabetically()
    image_queue = queue.Queue(maxsize=1)
    state.load_camera_params()
    if state.camera_params is None:
        rospy.logerr("Can not load camera parameters!")
    jetmax = hiwonder.JetMax()
    #add chart location function
    #detect camere
    
    jetmax.go_home()
    sucker = hiwonder.Sucker()
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)  # Subscribe to webcam footage
    while True:
        try:
            image_proc_b()
            if rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            break
