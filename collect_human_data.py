import argparse
import os
import time
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
from tele_vision import OpenTeleVision
import ros_numpy
import cv2
import threading
from camera_utils import list_video_devices, find_device_path_by_name
from multiprocessing import shared_memory
import sys
import signal
import h5py
from datetime import datetime
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class HumanDataCollector:
    def __init__(self, args):
        rospy.init_node('avp_teleoperation')
        self.args = args
        self.checkpt = 0
        self.manipulate_eef_idx = [0]
        self.update_manipulate_eef_idx()
        
        self.init_cameras()
        if self.args.head_camera_type == 0:
            img_shape = (self.head_frame_res[0], self.head_frame_res[1], 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name="pAlad1n3traiT")
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.tele_vision = OpenTeleVision(self.head_frame_res, self.shm.name, False)
        elif self.args.head_camera_type == 1:
            img_shape = (self.head_frame_res[0], 2 * self.head_frame_res[1], 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name="SaHlofoLinA")
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.tele_vision = OpenTeleVision(self.head_frame_res, self.shm.name, True)
        else:
            raise NotImplementedError("Not supported camera.")
        
        # command buffer
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()
        # eef 6d pose in two types of defined unified frame
        self.eef_uni_command = np.zeros((2, 6))
        
        self.fsm_state_msg = Int32()
        self.fsm_state_msg.data = 0

        # initial values
        self.initial_receive = True
        self.init_body_pos = np.zeros(3)
        self.init_body_rot = np.eye(3)
        self.init_eef_pos = np.zeros((2, 3))
        self.init_eef_rot = np.array([np.eye(3), np.eye(3)])
        self.init_gripper_angles = np.zeros(2)

        self.is_changing_reveive_status = False
        self.rate = rospy.Rate(200)
        
        self.on_reset = False
        self.on_collect = False
        self.on_save_data = False
        
        # self.pinch_gripper_angle_scale = 10.0
        self.pinch_dist_gripper_full_close = 0.02
        self.pinch_dist_gripper_full_open = 0.15
        # self.gripper_full_close_angle = 0.33
        self.gripper_full_close_angle = 0.01
        self.gripper_full_open_angle = 1.8
        self.eef_xyz_scale = 1.0
        
        # to get the ratation matrix of the world frame built by the apple vision pro, relative to the world frame of IssacGym or the real world used by LocoMan
        # first rotate along x axis with 90 degrees, then rotate along z axis with -90 degrees, in the frame of IsaacGym
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        
        self.last_reset_time = 0.0
        self.last_calib_time = 0.0
        
        self.pause_commands = False

        self.record_episode_counter = 0
        
        self.reset_trajectories()
        self.init_embodiment_states()

        # data collection
        if self.args.collect_data:
            self.human_data_folder = 'demonstrations'
            self.exp_name = self.args.exp_name
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_data_folder = self.human_data_folder + '/' + self.exp_name + '/human/' + current_time
            if not os.path.exists(self.exp_data_folder):
                os.makedirs(self.exp_data_folder)

    def reset_pov_transform(self):
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])

    def get_gripper_angle_from_pinch_dist(self, pinch_dist):
        scale = (self.gripper_full_open_angle - self.gripper_full_close_angle) / (self.pinch_dist_gripper_full_open - self.pinch_dist_gripper_full_close)
        angle = (pinch_dist - self.pinch_dist_gripper_full_close) * scale + self.gripper_full_close_angle
        return np.clip(angle, self.gripper_full_close_angle, self.gripper_full_open_angle)
    
    def update_manipulate_eef_idx(self):
        if self.args.manipulate_mode == 1:
            # right manipulation
            self.manipulate_eef_idx = [0]
        elif self.args.manipulate_mode == 2:
            # left manipulation
            self.manipulate_eef_idx = [1]
        elif self.args.manipulate_mode == 3:
            # left manipulation
            self.manipulate_eef_idx = [0, 1]

    def sim_view_callback(self, msg):
        self.sim_view_frame = ros_numpy.numpify(msg)
        np.copyto(self.img_array, self.sim_view_frame)
        # self.tele_vision.modify_shared_image(sim_view_image)

    def init_cameras(self):
        import pyrealsense2 as rs
        # initialize all cameras
        self.desired_stream_fps = self.args.desired_stream_fps
        # initialize head camera
        # realsense as head camera
        if self.args.head_camera_type == 0:
            self.head_camera_resolution = (480, 640)
            self.head_frame_res = (480, 640)
            self.head_color_frame = np.zeros((self.head_frame_res[0], self.head_frame_res[1], 3), dtype=np.uint8)
            self.head_cam_pipeline = rs.pipeline()
            self.head_cam_config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.head_cam_pipeline)
            pipeline_profile = self.head_cam_config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("a head camera is required")
                exit(0)
            self.head_cam_config.enable_stream(rs.stream.color, self.head_camera_resolution[1], self.head_camera_resolution[0], rs.format.bgr8, 30)
            # start streaming head cam
            self.head_cam_pipeline.start(self.head_cam_config)
        # stereo rgb camera (dual lens) as the head camera 
        elif self.args.head_camera_type == 1:
            # self.head_camera_resolution: the supported resoltion of the camera, to get the original frame without cropping
            self.head_camera_resolution = (720, 1280) # 720p, the original resolution of the stereo camera is actually 1080p (1080x1920)
            # self.head_view_resolution: the resolution of the images that are seen and recorded
            self.head_view_resolution = (480, 640) # 480p
            self.crop_size_w = 0
            self.crop_size_h = 0
            self.head_frame_res = (self.head_view_resolution[0] - self.crop_size_h, self.head_view_resolution[1] - 2 * self.crop_size_w)
            self.head_color_frame = np.zeros((self.head_frame_res[0], 2 * self.head_frame_res[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            head_camera_name = "3D USB Camera"
            device_path = find_device_path_by_name(device_map, head_camera_name)
            self.head_cap = cv2.VideoCapture(device_path)
            self.head_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.head_camera_resolution[1])
            self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.head_camera_resolution[0])
            self.head_cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        else:
            raise NotImplementedError("Not supported camera.")
        
        # initialize wrist camera/cameras
        if self.args.use_wrist_camera:
            self.wrist_camera_resolution = (720, 1280)
            self.wrist_view_resolution = (480, 640)
            self.wrist_color_frame = np.zeros((self.wrist_view_resolution[0], self.wrist_view_resolution[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            wrist_camera_name = "Global Shutter Camera"
            device_path = find_device_path_by_name(device_map, wrist_camera_name)
            self.wrist_cap1 = cv2.VideoCapture(device_path)
            self.wrist_cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.wrist_camera_resolution[1])
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wrist_camera_resolution[0])
            self.wrist_cap1.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)

    def prompt_rendering(self):
        viewer_img = self.head_color_frame
        height = viewer_img.shape[0]
        width = viewer_img.shape[1]
        if self.on_save_data:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=53)
            drawer.text((width / 8 - 25, (height - 80) / 2), "SAVING DATA", font=font, fill=(10, 255, 10))
            viewer_img = np.array(im)
        elif self.on_reset:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=53)
            drawer.text((width / 8 - 35, (height - 80) / 2), "PINCH to START", font=font, fill=(255, 63, 63))
            # drawer.text((767, 200), "PINCH to START", font=font, fill=(255, 63, 63))
            viewer_img = np.array(im)
        elif self.on_collect:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=20)
            if len(self.manipulate_eef_idx) == 1:
                drawer.text((width / 16 - 10, (height - 80) / 4), '{:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+1],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+2], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+3],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+4], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+5]
                        ), font=font, fill=(255, 63, 63))
            elif len(self.manipulate_eef_idx) == 2:
                drawer.text((width / 16 - 10, (height - 80) / 4), '{:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+1],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+2], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]+1], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]+2]
                        ), font=font, fill=(255, 63, 63))
            viewer_img = np.array(im)
        return viewer_img

    def head_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        if self.args.head_camera_type == 0:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    # handle head camera (realsense) streaming 
                    frames = self.head_cam_pipeline.wait_for_frames()
                    # depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        return
                    head_color_frame = np.asanyarray(color_frame.get_data())
                    head_color_frame = cv2.resize(head_color_frame, (self.head_frame_res[1], self.head_frame_res[0]))
                    self.head_color_frame = cv2.cvtColor(head_color_frame, cv2.COLOR_BGR2RGB)
                    np.copyto(self.img_array, self.head_color_frame)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
            finally:
                self.head_cam_pipeline.stop()
        elif self.args.head_camera_type == 1:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    ret, frame = self.head_cap.read()
                    frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                    image_left = frame[:, :self.head_frame_res[1], :]
                    image_right = frame[:, self.head_frame_res[1]:, :]
                    if self.crop_size_w != 0:
                        bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                        image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                    else:
                        bgr = np.hstack((image_left[self.crop_size_h:, :],
                                        image_right[self.crop_size_h:, :]))

                    self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    viewer_img = self.prompt_rendering()
                    np.copyto(self.img_array, viewer_img)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                self.head_cap.release()
        else:
            raise NotImplementedError('Not supported camera.')
        
    def wrist_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                ret, frame = self.wrist_cap1.read()
                wrist_color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.wrist_color_frame = cv2.resize(wrist_color_frame, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # print(1/(time.time() - start_time))
        finally:
            self.head_cap.release()

        
    def collect_data(self, event=None):
        # publish teleop commands at a fixed rate 
        # need to check the duration to finish one execution and if a separate thread is needed: takes 0.0002-0.0003s
        self.update_manipulate_eef_idx()
        # left_hand_data: [4, 4]
        left_hand_data = self.tele_vision.left_hand
        # right_hand_data: [4, 4]
        right_hand_data = self.tele_vision.right_hand
        # left_landmarks_data: [25, 3]
        left_landmarks_data = self.tele_vision.left_landmarks
        # right_landmarks_data: [25, 3]
        right_landmarks_data = self.tele_vision.right_landmarks
        # head_data: [4, 4]
        head_data = self.tele_vision.head_matrix
        
        left_wrist_pos = left_hand_data[:3, 3]
        left_wrist_rot = left_hand_data[:3, :3]
        right_wrist_pos = right_hand_data[:3, 3]
        right_wrist_rot = right_hand_data[:3, :3]
        head_pos = head_data[:3, 3]
        head_rot = head_data[:3, :3]        
        
        ### define new head rot and wrist local frame at reset to align with the unified frame which refers to the issacgym world frame
        ### issacgym world frame: x forward, y leftward, z upward
        ### unified frame: z upward (x-y plane horizontal to the ground), x aligns with the forward direction of human head at reset (rotate around z axis using the head yaw)
        head_rot_new = np.zeros_like(head_rot)
        head_rot_new[:, 0] = -head_rot[:, 2]
        head_rot_new[:, 1] = -head_rot[:, 0]
        head_rot_new[:, 2] = head_rot[:, 1]
        # keep the old one to extract head yaw at reset, then rotate the unified frame using the head yaw
        head_rot_old = head_rot
        head_rot = head_rot_new
        
        left_wrist_rot_new = np.zeros_like(left_wrist_rot)
        left_wrist_rot_new[:, 0] = -left_wrist_rot[:, 2]
        left_wrist_rot_new[:, 1] = -left_wrist_rot[:, 0]
        left_wrist_rot_new[:, 2] = left_wrist_rot[:, 1]
        left_wrist_rot = left_wrist_rot_new
        
        right_wrist_rot_new = np.zeros_like(right_wrist_rot)
        right_wrist_rot_new[:, 0] = -right_wrist_rot[:, 2]
        right_wrist_rot_new[:, 1] = -right_wrist_rot[:, 0]
        right_wrist_rot_new[:, 2] = right_wrist_rot[:, 1]
        right_wrist_rot = right_wrist_rot_new

        # thumb to index finger
        left_pinch_dist0 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[9]) 
        right_pinch_dist0 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[9]) 
        # thumb to middle finger
        left_pinch_dist1 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[14]) 
        right_pinch_dist1 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[14]) 
        # thumb to ring finger
        left_pinch_dist2 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[19])
        right_pinch_dist2 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[19])
        # thumb to little finger
        left_pinch_dist3 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[24])
        right_pinch_dist3 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[24])
        
        # reset
        if left_pinch_dist1 < 0.008 and left_pinch_dist0 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05 and time.time() - self.last_reset_time > 3.0:
            self.on_reset = True
            self.on_collect = False
            self.last_reset_time = time.time()
            self.initial_receive = True
            self.reset_pov_transform()
            self.flush_trajectory()
            print("reset")
        self.command[:] = 0
        # initialize and calibrate
        if self.on_reset:
            # the initialization only begins at a certain signal (e.g., a gesture)
            # the gesture is temporarily designed for single hand manipulation
            if left_pinch_dist0 < 0.008 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05:
                self.on_reset = False
                self.on_collect = True
                if self.args.collect_data:
                    self.reset_embodiment_states()
                    # extract the head rotation around y axis in the world frame of apple vision pro (head yaw)
                    head_rpy = self.rot_mat_to_rpy_zxy(head_rot_old)
                    # use the head yaw to build a new frame to map the 6d poses of the human hand and head
                    # the new operator_pov_transform represents the rotation matrix of the apple vision pro frame relative to the unified frame (IsaacGym/world frame)
                    # unified frame: z upward (x-y plane horizontal to the ground), x aligns with the forward direction of human head at reset (rotate around z axis using the head yaw)
                    z_axis_rot = self.rpy_to_rot_mat(np.array([0, 0, -head_rpy[1]]))
                    self.operator_pov_transform = z_axis_rot @ self.operator_pov_transform
                    self.init_body_pos[:] = self.operator_pov_transform @ head_pos
                    self.init_body_rot[:] = self.operator_pov_transform @ head_rot
                    # use the head position at reset as the origin of the frame
                    # body (head) pose [6]
                    body_pos_uni = self.operator_pov_transform @ head_pos - self.init_body_pos
                    body_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ head_rot)
                    self.body_pose_uni = np.concatenate((body_pos_uni, body_rot_uni))
                    self.delta_body_pose_uni = np.zeros_like(self.body_pose_uni)
                    # eef pose [6 * 2]
                    right_eef_pos_uni = self.operator_pov_transform @ right_wrist_pos - self.init_body_pos
                    right_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ right_wrist_rot)
                    left_eef_pos_uni = self.operator_pov_transform @ left_wrist_pos - self.init_body_pos
                    left_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ left_wrist_rot)
                    self.eef_pose_uni = np.concatenate([right_eef_pos_uni, right_eef_rot_uni, left_eef_pos_uni, left_eef_rot_uni])
                    self.delta_eef_pose_uni = np.zeros_like(self.eef_pose_uni)
                    # eef relative to body [6 * 2]
                    right_eef_to_body_pos = right_eef_pos_uni - body_pos_uni
                    right_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(right_eef_rot_uni))
                    left_eef_to_body_pos = left_eef_pos_uni - body_pos_uni
                    left_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(left_eef_rot_uni))
                    self.eef_to_body_pose = np.concatenate([right_eef_to_body_pos, right_eef_to_body_rot, left_eef_to_body_pos, left_eef_to_body_rot])
                    # simulated gripper (hand as gripper) [1 * 2]
                    right_gripper_angle = self.get_gripper_angle_from_pinch_dist(right_pinch_dist0)
                    left_gripper_angle = self.get_gripper_angle_from_pinch_dist(left_pinch_dist0)
                    self.gripper_angle = np.array([right_gripper_angle, left_gripper_angle])
                    self.delta_gripper_angle = np.zeros_like(self.gripper_angle)
                    # hand joints
                    self.right_hand_joints = right_landmarks_data.flatten()
                    self.left_hand_joints = left_landmarks_data.flatten()
                    # main camera pose relative to its initial pose in the unified frame (we assume that the main camera is fixed on the head)
                    head_camera_to_init_rot = self.rot_mat_to_rpy(self.init_body_rot.T @ (self.operator_pov_transform @ head_rot))
                    self.head_camera_to_init_pose_uni = np.concatenate((body_pos_uni, head_camera_to_init_rot))
                    self.update_trajectories()
            else:
                return
        # start collect
        elif self.on_collect:
            if left_pinch_dist3 < 0.008 and left_pinch_dist0 > 0.05 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05:
                self.pause_commands = True
                print("Pause sending commands")
                return
            if self.pause_commands and left_pinch_dist0 < 0.008 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05:
                print("Restart sending commands")
                self.pause_commands = False
            if self.pause_commands:
                return

            if self.args.collect_data:
                self.reset_embodiment_states()
                # use the head position (3D) at reset as the origin of the frame
                # body (head) pose [6]
                body_pos_uni = self.operator_pov_transform @ head_pos - self.init_body_pos
                body_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ head_rot)
                body_pose_uni = np.concatenate((body_pos_uni, body_rot_uni))
                self.delta_body_pose_uni = body_pose_uni - self.body_pose_uni
                self.body_pose_uni = body_pose_uni
                # eef pose [6 * 2]
                right_eef_pos_uni = self.operator_pov_transform @ right_wrist_pos - self.init_body_pos
                right_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ right_wrist_rot)
                left_eef_pos_uni = self.operator_pov_transform @ left_wrist_pos - self.init_body_pos
                left_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ left_wrist_rot)
                eef_pose_uni = np.concatenate([right_eef_pos_uni, right_eef_rot_uni, left_eef_pos_uni, left_eef_rot_uni])
                self.delta_eef_pose_uni = eef_pose_uni - self.eef_pose_uni
                self.eef_pose_uni = eef_pose_uni
                # eef relative to body [6 * 2]
                right_eef_to_body_pos = right_eef_pos_uni - body_pos_uni
                right_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(right_eef_rot_uni))
                left_eef_to_body_pos = left_eef_pos_uni - body_pos_uni
                left_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(left_eef_rot_uni))
                self.eef_to_body_pose = np.concatenate([right_eef_to_body_pos, right_eef_to_body_rot, left_eef_to_body_pos, left_eef_to_body_rot])
                # simulated gripper (hand as gripper) [1 * 2]
                right_gripper_angle = self.get_gripper_angle_from_pinch_dist(right_pinch_dist0)
                left_gripper_angle = self.get_gripper_angle_from_pinch_dist(left_pinch_dist0)
                gripper_angle = np.array([right_gripper_angle, left_gripper_angle])
                self.delta_gripper_angle = gripper_angle - self.gripper_angle
                self.gripper_angle = gripper_angle
                # hand joints
                self.right_hand_joints = right_landmarks_data.flatten()
                self.left_hand_joints = left_landmarks_data.flatten()
                # main camera pose relative to its initial pose in the unified frame (we assume that the main camera is fixed on the head)
                head_camera_to_init_rot = self.rot_mat_to_rpy(self.init_body_rot.T @ (self.operator_pov_transform @ head_rot))
                self.head_camera_to_init_pose_uni = np.concatenate((body_pos_uni, head_camera_to_init_rot))
                self.update_trajectories()
    
    def init_embodiment_states(self):
        self.body_pose_uni = np.zeros(6)
        self.delta_body_pose_uni = np.zeros(6)
        self.eef_pose_uni = np.zeros(12)
        self.delta_eef_pose_uni = np.zeros(12)
        self.eef_to_body_pose = np.zeros(12)
        self.gripper_angle = np.zeros(2)
        self.delta_gripper_angle = np.zeros(2)
        self.right_hand_joints = np.zeros(75) # (25, 3)
        self.left_hand_joints = np.zeros(75) # (25, 3)
        # camera pose relative to its initial pose in the unified frame
        self.head_camera_to_init_pose_uni = np.zeros(6)

    def reset_embodiment_states(self):
        # need to reset: make sure add the updated state to the trajectory meanwhile keep previous ones unchanged
        # not reset to zeros: copy the value to compute delta
        self.body_pose_uni = self.body_pose_uni.copy()
        self.delta_body_pose_uni = self.delta_body_pose_uni.copy()
        self.eef_pose_uni = self.eef_pose_uni.copy()
        self.delta_eef_pose_uni = self.delta_eef_pose_uni.copy()
        self.eef_to_body_pose = self.eef_to_body_pose.copy()
        self.gripper_angle = self.gripper_angle.copy()
        self.delta_gripper_angle = self.delta_gripper_angle.copy()
        self.right_hand_joints = self.right_hand_joints.copy()
        self.left_hand_joints = self.left_hand_joints.copy()
        # camera pose relative to its initial pose in the unified frame
        self.head_camera_to_init_pose_uni = self.head_camera_to_init_pose_uni.copy()

    def reset_trajectories(self):
        self.main_cam_image_history = []
        self.wrist_cam_image_history = []
        self.body_pose_history = []
        self.delta_body_pose_history = []
        self.eef_pose_history = []
        self.delta_eef_pose_history = []
        self.eef_to_body_pose_history = []
        self.gripper_angle_history = []
        self.delta_gripper_angle_history = []
        self.right_hand_joints_history = []
        self.left_hand_joints_history = []
        self.head_camera_to_init_pose_history = []

    def update_trajectories(self):
        self.main_cam_image_history.append(self.head_color_frame)
        if self.args.use_wrist_camera:
            self.wrist_cam_image_history.append(self.wrist_color_frame) 
        self.body_pose_history.append(self.body_pose_uni)
        self.delta_body_pose_history.append(self.delta_body_pose_uni)
        self.eef_pose_history.append(self.eef_pose_uni)
        self.delta_eef_pose_history.append(self.delta_eef_pose_uni)
        self.eef_to_body_pose_history.append(self.eef_to_body_pose)
        self.gripper_angle_history.append(self.gripper_angle)
        self.delta_gripper_angle_history.append(self.delta_gripper_angle)
        self.right_hand_joints_history.append(self.right_hand_joints)
        self.left_hand_joints_history.append(self.left_hand_joints) 
        self.head_camera_to_init_pose_history.append(self.head_camera_to_init_pose_uni)

    def get_embodiment_masks(self):
        self.img_main_mask = np.array([True])
        self.img_wrist_mask = np.array([False, False])
        if self.args.use_wrist_camera:
            for eef_idx in self.manipulate_eef_idx:
                self.img_wrist_mask[self.manipulate_eef_idx] = True
        self.proprio_body_mask = np.array([True, True, True, True, True, True])
        self.proprio_eef_mask = np.array([False] * 12)
        self.proprio_gripper_mask = np.array([False, False])
        self.proprio_other_mask = np.array([False, False])            
        self.act_body_mask = np.array([True, True, True, True, True, True])
        self.act_eef_mask = np.array([False] * 12)
        self.act_gripper_mask = np.array([False, False])
        for eef_idx in self.manipulate_eef_idx:
            self.proprio_eef_mask[6*eef_idx:6+6*eef_idx] = True
            self.proprio_gripper_mask[eef_idx] = True
            # unsure if we should use the hand joints as priprio
            self.proprio_other_mask[eef_idx] = True
            self.act_eef_mask[6*eef_idx:6+6*eef_idx] = True
            self.act_gripper_mask[eef_idx] = True

    def flush_trajectory(self):
        if self.args.collect_data and len(self.eef_pose_history) > 0:
            self.on_save_data = True
            self.get_embodiment_masks()
            save_path = self.exp_data_folder
            episode_num = self.increment_counter
            with h5py.File(os.path.join(save_path, f'episode_{episode_num}.hdf5'), 'a') as f:
                # add the embodiment tag
                f.attrs['embodiment'] = 'human'
                # save observations
                obs_group = f.create_group('observations')
                image_group = obs_group.create_group('images')
                image_group.create_dataset('main', data=self.main_cam_image_history)
                image_group.create_dataset('wrist', data=self.wrist_cam_image_history)
                proprio_group = obs_group.create_group('proprioceptions')
                proprio_group.create_dataset('body', data=self.body_pose_history)
                proprio_group.create_dataset('eef', data=self.eef_pose_history)
                proprio_group.create_dataset('eef_to_body', data=self.eef_to_body_pose_history)
                proprio_group.create_dataset('gripper', data=self.gripper_angle_history)
                proprio_other_group = proprio_group.create_group('other')
                proprio_other_group.create_dataset('right_hand_joints', data=self.right_hand_joints_history)
                proprio_other_group.create_dataset('left_hand_joints', data=self.left_hand_joints_history)
                # save actions
                action_group = f.create_group('actions')
                action_group.create_dataset('body', data=self.body_pose_history)
                action_group.create_dataset('delta_body', data=self.delta_body_pose_history)
                action_group.create_dataset('eef', data=self.eef_pose_history)
                action_group.create_dataset('delta_eef', data=self.delta_eef_pose_history)
                action_group.create_dataset('gripper', data=self.gripper_angle_history)
                action_group.create_dataset('delta_gripper', data=self.delta_gripper_angle_history)
                # save masks
                mask_group = f.create_group('masks')
                mask_group.create_dataset('img_main', data=self.img_main_mask)
                mask_group.create_dataset('img_wrist', data=self.img_wrist_mask)
                mask_group.create_dataset('proprio_body', data=self.proprio_body_mask)
                mask_group.create_dataset('proprio_eef', data=self.proprio_eef_mask)
                mask_group.create_dataset('proprio_gripper', data=self.proprio_gripper_mask)
                mask_group.create_dataset('proprio_other', data=self.proprio_other_mask)
                mask_group.create_dataset('act_body', data=self.act_body_mask)
                mask_group.create_dataset('act_eef', data=self.act_eef_mask)
                mask_group.create_dataset('act_gripper', data=self.act_gripper_mask)
                # save camera poses
                camera_group = f.create_group('camera_poses')
                camera_group.create_dataset('head_camera_to_init', data=self.head_camera_to_init_pose_history)
            # save videos
            if self.args.save_video:
                h, w, _ = self.main_cam_image_history[0].shape
                freq = self.args.control_freq
                main_cam_video = cv2.VideoWriter(os.path.join(save_path, f'episode_{episode_num}_main_cam_video.mp4'), 
                                                 cv2.VideoWriter_fourcc(*'mp4v'), freq, (w, h))
                
                for image in self.main_cam_image_history:
                    # swap back to bgr for opencv
                    image = image[:, :, [2, 1, 0]] 
                    main_cam_video.write(image)
                main_cam_video.release()
                if self.args.use_wrist_camera:
                    h, w, _ = self.wrist_cam_image_history[0].shape
                    freq = self.args.control_freq
                    wrist_cam_video = cv2.VideoWriter(os.path.join(save_path, f'episode_{episode_num}_wrist_cam_video.mp4'), 
                                                    cv2.VideoWriter_fourcc(*'mp4v'), freq, (w, h))
                    for image in self.wrist_cam_image_history:
                        # swap back to bgr for opencv
                        image = image[:, :, [2, 1, 0]] 
                        wrist_cam_video.write(image)
                    wrist_cam_video.release()
            self.reset_trajectories()
            self.on_save_data = False
    
    @property
    def increment_counter(self):
        self.record_episode_counter += 1
        return self.record_episode_counter
    
    def run(self):
        head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
        head_camera_streaming_thread.start()
        if self.args.use_wrist_camera:
            wrist_camera_streaming_thread = threading.Thread(target=self.wrist_camera_stream_thread, daemon=True)
            wrist_camera_streaming_thread.start()
        rospy.Timer(rospy.Duration(1.0 / self.args.control_freq), self.collect_data)
        rospy.spin()

    def rot_mat_to_rpy(self, R):
        """
        Convert a rotation matrix to Euler angles (roll, pitch, yaw) with ZYX order.
        This method is more numerically stable, especially near singularities.
        """
        sy = -R[2, 0]
        singular_threshold = 1e-6  # Threshold to check for singularity
        cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        # Handle singularity
        if cy < singular_threshold:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(sy, cy)
            z = 0  # Yaw is indeterminate in this case
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(sy, cy)
            z = np.arctan2(R[1, 0], R[0, 0])

        return np.array([x, y, z])

    def rpy_to_rot_mat(self, rpy):
        """
        Convert Euler angles (roll, pitch, yaw) with ZYX order to a rotation matrix.
        """
        x, y, z = rpy
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)

        R = np.array([
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy]
        ])
        return R

    def rot_mat_to_rpy_zxy(self, R):
        """
        Convert a rotation matrix (RzRxRy) to Euler angles with ZXY order (first rotate with y, then x, then z).
        This method is more numerically stable, especially near singularities.
        """
        sx = R[2, 1]
        singular_threshold = 1e-6
        cx = np.sqrt(R[2, 0]**2 + R[2, 2]**2)

        if cx < singular_threshold:
            x = np.arctan2(sx, cx)
            y = np.arctan2(R[0, 2], R[0, 0])
            z = 0
        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([x, y, z])

    def close(self):
        self.shm.close()
        self.shm.unlink()       
        print('clean up shared memory')

def signal_handler(sig, frame):
    print('pressed Ctrl+C! exiting...')
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(prog="teleoperation with apple vision pro and vuer")
    parser.add_argument("--head_camera_type", type=int, default=1, help="0=realsense, 1=stereo rgb camera")
    parser.add_argument("--use_wrist_camera", type=str2bool, default=False, help="whether to use wrist camera")
    parser.add_argument("--desired_stream_fps", type=int, default=60, help="desired camera streaming fps to vuer")
    parser.add_argument("--control_freq", type=int, default=60, help="frequency to record human data")
    parser.add_argument("--collect_data", type=str2bool, default=True, help="whether to collect data")
    parser.add_argument("--manipulate_mode", type=int, default=1, help="1: right eef; 2: left eef; 3: bimanual")
    parser.add_argument('--save_video', type=str2bool, default=True, help="whether to collect save videos of camera views when storing the data")
    parser.add_argument("--exp_name", type=str, default='test')
    # exp_name

    args = parser.parse_args()
    
    avp_teleoperator = HumanDataCollector(args)
    try:
        avp_teleoperator.run()
    finally:
        avp_teleoperator.close()









