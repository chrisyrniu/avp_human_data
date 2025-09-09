#!/usr/bin/env python3
import h5py
import numpy as np
import re

FILE_PATH = '/home/yaru/research/bosch_data_collect/VibeMesh/BoschTripScripts/h2l/demonstrations/test/human/20250906_175506/episode_1.hdf5'

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def fmt_shape(shape):
    try:
        return "(" + ", ".join(str(int(x)) for x in shape) + ")"
    except Exception:
        return str(shape)

def dataset_info(dset: h5py.Dataset) -> str:
    return f"shape={fmt_shape(dset.shape)}, dtype={dset.dtype}"

def get_images_len_and_shape(node):
    """Supports both dataset layout (T,H,W,C) and group-of-datasets layout ('0','1',...)."""
    if isinstance(node, h5py.Dataset):
        shp = node.shape
        per = tuple(shp[1:4]) if len(shp) >= 4 else None
        return shp[0], per
    elif isinstance(node, h5py.Group):
        keys = sorted(node.keys(), key=natural_key)
        if not keys:
            return 0, None
        first = node[keys[0]]
        return len(keys), tuple(first.shape) if isinstance(first, h5py.Dataset) else None
    else:
        return 0, None

def walk_and_print(f: h5py.File):
    print("=" * 80)
    print(f"File: {f.filename}")
    if f.attrs:
        print("Attributes:")
        for k, v in f.attrs.items():
            print(f"  - {k}: {v}")
    print("-" * 80)

    # Pretty tree with shapes
    def visit(path, obj):
        indent = "  " * path.count("/")
        base = path.split("/")[-1]
        if isinstance(obj, h5py.Group):
            print(f"{indent}{base}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{base}: {dataset_info(obj)}")

    f.visititems(visit)
    print("-" * 80)

    # Explicit, human-friendly summaries for common elements
    obs = f.get("observations")
    if obs is not None:
        # images
        if "images" in obs:
            imgs = obs["images"]
            if "main" in imgs:
                T, per = get_images_len_and_shape(imgs["main"])
                print(f"observations/images/main -> T={T}, per-frame={per}")
            if "wrist" in imgs:
                T, per = get_images_len_and_shape(imgs["wrist"])
                print(f"observations/images/wrist -> T={T}, per-frame={per}")

        # timestamps
        if "head_cam_timestamp" in obs:
            d = obs["head_cam_timestamp"]
            print(f"observations/head_cam_timestamp -> {dataset_info(d)}")

        # proprioceptions
        if "proprioceptions" in obs:
            prop = obs["proprioceptions"]
            for key in ["body", "eef", "eef_to_body", "gripper"]:
                if key in prop:
                    d = prop[key]
                    print(f"observations/proprioceptions/{key} -> {dataset_info(d)}")
            if "other" in prop:
                oth = prop["other"]
                for key in ["right_hand_joints", "left_hand_joints"]:
                    if key in oth:
                        d = oth[key]
                        print(f"observations/proprioceptions/other/{key} -> {dataset_info(d)}")

    # actions
    acts = f.get("actions")
    if acts is not None:
        for key in ["body", "delta_body", "eef", "delta_eef", "gripper", "delta_gripper"]:
            if key in acts:
                d = acts[key]
                print(f"actions/{key} -> {dataset_info(d)}")

    # masks
    masks = f.get("masks")
    if masks is not None:
        for key in masks.keys():
            d = masks[key]
            print(f"masks/{key} -> {dataset_info(d)}")

    # camera poses
    cams = f.get("camera_poses")
    if cams is not None and "head_camera_to_init" in cams:
        d = cams["head_camera_to_init"]
        print(f"camera_poses/head_camera_to_init -> {dataset_info(d)}")

    print("=" * 80)

def main():
    with h5py.File(FILE_PATH, "r") as f:
        walk_and_print(f)

if __name__ == "__main__":
    main()
