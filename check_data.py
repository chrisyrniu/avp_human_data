import h5py
import numpy as np
import matplotlib.pyplot as plt

file_dir = '/home/yaru/research/bosch_data_collect/VibeMesh/BoschTripScripts/h2l/demonstrations/test/human/20250906_175506/episode_1.hdf5'

with h5py.File(file_dir, 'r') as f:
    print(f.attrs['embodiment'])
    main_images = np.array(f['observations/images/main'])
    timestamp = np.array(f['observations/head_cam_timestamp'])

print("main_images:", main_images.shape, "timestamps:", timestamp.shape)

# --- Interactive viewer ---
class ImageViewer:
    def __init__(self, images, timestamps, fps=10):
        self.images = images
        self.timestamps = timestamps
        self.N = images.shape[0]
        self.i = 0
        self.play = False
        self.delay = 1.0 / fps

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_axis_off()
        self.im = self.ax.imshow(self.images[0])
        self.title = self.ax.set_title(self._title_text())

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _title_text(self):
        return f"Frame {self.i+1}/{self.N}  |  t = {self.timestamps[self.i]:.3f}s"

    def show_frame(self):
        self.im.set_data(self.images[self.i])
        self.title.set_text(self._title_text())
        self.fig.canvas.draw_idle()

    def step(self, delta):
        self.i = (self.i + delta) % self.N
        self.show_frame()

    def run(self):
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number):
            if self.play:
                self.step(+1)
                plt.pause(self.delay)
            else:
                plt.pause(0.05)

    def on_key(self, e):
        if e.key == "right":
            self.step(+1)
        elif e.key == "left":
            self.step(-1)
        elif e.key == " ":
            self.play = not self.play
        elif e.key in ("q", "escape"):
            plt.close(self.fig)

viewer = ImageViewer(main_images, timestamp, fps=10)
viewer.run()
