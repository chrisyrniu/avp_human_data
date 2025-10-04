<h1 align="center">
  Human Data Collection with Apple Vision Pro
</h1>


## Installation 
1. Create a new virtual environment.
    ```bash
    conda create -n avp_human_data python=3.8
    conda activate avp_human_data
    ```
2. Install dependencies:
    ```bash
    pip install -e .
    ```
3. Install ROS:
   * Follow the steps [here](https://wiki.ros.org/noetic/Installation/Ubuntu) to install ROS Noetic.
   * If you are using a system newer than Ubuntu 20.04 (e.g., Ubuntu 22.04), we recommand installing [ROS Noetic from RoboStack with Pixi](https://robostack.github.io/GettingStarted.html) to launch the ROS master, and then installing the following ROS Noetic packages within your `Conda` environment:
        ```bash
        conda install -c robostack -c conda-forge ros-noetic-rospy ros-noetic-sensor-msgs ros-noetic-nav-msgs
        ```
4. Install `ros_numpy`:
   ```bash
   git clone https://github.com/eric-wieser/ros_numpy.git
   cd ros_numpy
   pip install -e .
   ```
5. Following the instructions in [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) to install [mkcert](https://github.com/FiloSottile/mkcert) and ca-certificats on Vision Pro.

## Hardware Setup

### Hardware
Pleasd refer to [hardware.md](doc/hardware.md) for the hardware guidance.


## Human Data Collection
<p align="center">
  <img src="doc/figures/human_dump_teaser.gif" width="60%" />
</p>
<p align="center">An example of two consecutive episodes of human data collection.</p>

### Implementation Steps
1. Connect the dual-lens camera mounted on the Apple Vision Pro to the server.
2. Run the following command to collect human data of the bimanual mode (`--manipulate_mode=3`) with a frequency of 30 (`--control_freq=30`), without saving the video from the main camera view (`--save_video=False`):
    ```bash
    python collect_human_data.py --control_freq=30 --head_camera_res=720p --collect_data=True --manipulate_mode=3 --save_video=False --exp_name=test 
    ```
3. Open the Safari browser on the Apple Vision Pro, and go to the `Vuer` webpage: `https://your.shared.local.address:8012?ws=wss://your.shared.local.address:8012`, and then enter the VR session.
4. Pinch your left thumb and middle finger to reset/initialize the data recording, untill you see "PINCH to START" in red to show up.
5. Pinch your left thumb and index finger to start recording.
When you complete an episode, pinch your left thumb and middle finger to end the recording and reset the session.
Wait while "SAVING DATA" appears on the screen. The system will be ready for the next take once "PINCH TO START" is displayed again.


## Data Storage Format
Each trajectory is stored in an hdf5 file with the following structure:

### HDF5 File Structure
```
root
├── observations
│   ├── head_cam_timestamp # the system timestamp of each head camera frame
│   ├── images
│   │   ├─ main          # images from main camera: [h * w * c] * traj_length
│   │   └─ wrist         # images from wrist cameras: [h * w * c] * traj_length
│   └── proprioceptions       
│       ├─ body          # 6d pose of the rigid body where the main camera is mounted: [6] * traj_length
│       ├─ eef           # 6d pose of the end effectors (at most two, right first): [12] * traj_length
│       ├─ relative      # relative 6d pose of the end effector to the rigid body where the main camera is mounted (at most two, right first): [12] * traj_length
│       ├─ gripper       # gripper angle (at most two, right first): [2] * traj_length
│       └─ other         # other prorioceptive state, e.g., robot joint positions, robot joint velocities, human hand joint poses, ...
├── actions       
│   ├── body             # 6d pose of the rigid body where the main camera is mounted: [6] * traj_length
│   ├── delta_body       # delta 6d pose of the rigid body where the main camera is mounted: [6] * traj_length
│   ├── eef              # 6d pose of the end effectors (at most two, right first): [12] * traj_length
│   ├── delta_eef        # delta 6d pose of the end effectors (at most two, right first): [12] * traj_length
│   ├── gripper          # gripper angle (at most two, right first): [2] * traj_length
│   └── delta_gripper    # delta gripper angle (at most two, right first): [2] * traj_length
├── masks                # embodiment-specific masks to mask out observations and actions for training and inference (the mask for observations and actions of the same modality could be different)
│   ├── img_main         # mask for the main camera image input: [1]
│   ├── img_wrist        # mask for the wrist camera image input: [2]
│   ├── proprio_body     # mask for the body 6d pose input: [6]
│   ├── proprio_eef      # mask for the eef 6d pose input: [12]
│   ├── proprio_gripper  # mask for the gripper angle input: [2]
│   ├── proprio_other    # mask for other proprioception input (n components): [n]
│   ├── act_body         # mask for the body 6d pose output: [6]
│   ├── act_eef          # mask for the eef 6d pose output: [12]
│   └── act_gripper      # mask for the gripper angle output: [2]
└── camera_poses
    └── head_camera_to_init   # relative 6d head camera pose to its original one in the unifed frame: [6] * traj_length
```