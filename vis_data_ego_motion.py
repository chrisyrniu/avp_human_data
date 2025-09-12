import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import re

# ---------- config ----------
file_dir = '/home/yaru/research/bosch_data_collect/avp_human_data/demonstrations/test/human/20250911_202103/episode_1.hdf5'
output_path = 'motion_with_ego.mp4'
fps = 30
step = 1
frame_axis_len = 0.15
trail_len = 50

# ---------- helpers ----------
def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def make_frame_lines(ax, color_x='r', color_y='g', color_z='b'):
    lx, = ax.plot([], [], [], color=color_x, linewidth=2)
    ly, = ax.plot([], [], [], color=color_y, linewidth=2)
    lz, = ax.plot([], [], [], color=color_z, linewidth=2)
    return lx, ly, lz

def update_frame_lines(lines, origin, R, length):
    for line, v in zip(lines, [R[:,0], R[:,1], R[:,2]]):
        p0, p1 = origin, origin + length * v
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

def try_writer(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mp4':
        try:
            return FFMpegWriter(fps=fps, bitrate=6000)
        except Exception:
            print("[warn] FFmpeg not available; falling back to GIF.")
            return PillowWriter(fps=min(15, fps))
    else:
        return PillowWriter(fps=min(15, fps))

def natural_ints(s):
    # extract first integer in a string (e.g., '00012' -> 12)
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.inf

# ---------- open file ----------
f = h5py.File(file_dir, 'r')
print("embodiment:", f.attrs.get('embodiment', 'unknown'))

body_pose_proprio = np.array(f['observations/proprioceptions/body'])          # (T,6)
eef_pose_proprio  = np.array(f['observations/proprioceptions/eef'])           # (T,12)
right_hand_joints = np.array(f['observations/proprioceptions/other/right_hand_joints'])  # (T,75)
left_hand_joints  = np.array(f['observations/proprioceptions/other/left_hand_joints'])   # (T,75)

# ---------- images: dataset OR group ----------
images_node = f['observations/images/main']
timestamps  = np.array(f['observations/head_cam_timestamp'])

if isinstance(images_node, h5py.Dataset):
    # Layout A: a single dataset shaped (T, H, W, 3)
    images_len = images_node.shape[0]
    def get_image(k):
        return images_node[k]  # (H, W, 3) uint8
else:
    # Layout B: a group with one dataset per frame (keys '0','1',...)
    frame_keys = sorted(list(images_node.keys()), key=natural_ints)
    images_len = len(frame_keys)
    def get_image(k):
        # Each child is a dataset; [()] reads the whole dataset into a numpy array
        return images_node[frame_keys[k]][()]

# ---------- align lengths ----------
T = body_pose_proprio.shape[0]
T = min(T, images_len)  # ensure we don't index past images
assert eef_pose_proprio.shape[0] >= T and right_hand_joints.shape[0] >= T and left_hand_joints.shape[0] >= T

eef_r = eef_pose_proprio[:T, 0:6]
eef_l = eef_pose_proprio[:T, 6:12]
rh = right_hand_joints[:T].reshape(T, 25, 3)
lh = left_hand_joints[:T].reshape(T, 25, 3)
body_xyz = body_pose_proprio[:T, :3]
eefr_xyz = eef_r[:, :3]
eefl_xyz = eef_l[:, :3]

# downsample indices
idx = np.arange(0, T, step)
T_show = len(idx)

# limits for 3D
pts = np.concatenate([body_xyz, eefr_xyz, eefl_xyz, rh.reshape(-1,3), lh.reshape(-1,3)], axis=0)
mins, maxs = pts.min(0), pts.max(0)
center, span = (mins+maxs)/2, (maxs-mins)
max_span = float(span.max()); pad = 0.1*max_span if max_span>0 else 1.0
lims = np.vstack([center-(max_span/2+pad), center+(max_span/2+pad)])

# ---------- figure ----------
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
ax_img = fig.add_subplot(gs[0, 0])
ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
fig.subplots_adjust(wspace=0.10)

first_img = get_image(idx[0])
im_artist = ax_img.imshow(first_img)
ax_img.axis('off')

if len(timestamps) > 0:
    ax_img.set_title(f"Egocentric View  –  t = {timestamps[0]:.3f} s")
else:
    ax_img.set_title("Egocentric View")

ax = ax_3d
ax.set_box_aspect([1,1,1])
ax.set_xlim(lims[0,0], lims[1,0]); ax.set_ylim(lims[0,1], lims[1,1]); ax.set_zlim(lims[0,2], lims[1,2])
ax.set_xlabel('X', fontsize=8); ax.set_ylabel('Y', fontsize=8); ax.set_zlabel('Z', fontsize=8)
ax.set_title('Head / EEF / Hand Joints', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=8)

# Set custom view angle so X axis is on the right, Y axis on the left
# elev: elevation angle (vertical rotation), azim: azimuth angle (horizontal rotation)
ax.view_init(elev=30, azim=225)

# Define colors for right and left sides
head_color = 'lightcoral'
right_color = 'purple'
left_color = 'orange'

body_scatter = ax.scatter([], [], [], s=60, depthshade=True, label='head', color=head_color)
eefr_scatter = ax.scatter([], [], [], s=50, depthshade=True, label='right eef', color=right_color)
eefl_scatter = ax.scatter([], [], [], s=50, depthshade=True, label='left eef', color=left_color)
rh_scatter = ax.scatter([], [], [], s=10, alpha=0.8, depthshade=True, label='right hand joints', color=right_color)
lh_scatter = ax.scatter([], [], [], s=10, alpha=0.8, depthshade=True, label='left hand joints', color=left_color)
body_trail, = ax.plot([], [], [], linewidth=1.5, alpha=0.7, label='head trail', color=head_color)
eefr_trail, = ax.plot([], [], [], linewidth=1.0, alpha=0.7, label='right eef trail', color=right_color)
eefl_trail, = ax.plot([], [], [], linewidth=1.0, alpha=0.7, label='left eef trail', color=left_color)

body_axes = make_frame_lines(ax)
eefr_axes = make_frame_lines(ax)
eefl_axes = make_frame_lines(ax)
ax.legend(loc='upper right', fontsize=8)

def set_scatter_xyz(scatter, xyz):
    scatter._offsets3d = (np.array([xyz[0]]), np.array([xyz[1]]), np.array([xyz[2]]))

def set_scatter_points(scatter, P):
    scatter._offsets3d = (P[:,0], P[:,1], P[:,2])

# ---------- animation ----------
def animate(i):
    j = idx[i]

    # image
    frame_img = get_image(j)
    im_artist.set_data(frame_img)

    # timestamp
    if j < len(timestamps):
        ax_img.set_title(f"Egocentric View  –  t = {timestamps[j]:.3f} s")
    else:
        ax_img.set_title("Egocentric View")

    # body + frames
    bpos = body_pose_proprio[j, :3]; bR = rpy_to_R(*body_pose_proprio[j, 3:6])
    set_scatter_xyz(body_scatter, bpos); update_frame_lines(body_axes, bpos, bR, frame_axis_len)

    rpos = eef_r[j, :3]; rR = rpy_to_R(*eef_r[j, 3:6])
    set_scatter_xyz(eefr_scatter, rpos); update_frame_lines(eefr_axes, rpos, rR, frame_axis_len*0.8)

    lpos = eef_l[j, :3]; lR = rpy_to_R(*eef_l[j, 3:6])
    set_scatter_xyz(eefl_scatter, lpos); update_frame_lines(eefl_axes, lpos, lR, frame_axis_len*0.8)

    set_scatter_points(rh_scatter, rh[j]); set_scatter_points(lh_scatter, lh[j])

    if trail_len > 0:
        s = max(0, j - trail_len)
        body_trail.set_data(body_xyz[s:j+1,0], body_xyz[s:j+1,1]); body_trail.set_3d_properties(body_xyz[s:j+1,2])
        eefr_trail.set_data(eefr_xyz[s:j+1,0], eefr_xyz[s:j+1,1]); eefr_trail.set_3d_properties(eefr_xyz[s:j+1,2])
        eefl_trail.set_data(eefl_xyz[s:j+1,0], eefl_xyz[s:j+1,1]); eefl_trail.set_3d_properties(eefl_xyz[s:j+1,2])

    ax_3d.set_title(f'Head / EEF / Hand Joints  –  frame {i+1}/{T_show}')
    return [im_artist]

anim = FuncAnimation(fig, animate, frames=T_show, interval=1000.0/fps, blit=False)
writer = try_writer(output_path)
anim.save(output_path, writer=writer, dpi=120)
print(f"Saved to {output_path}")

plt.show()
