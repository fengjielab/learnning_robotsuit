import h5py
import datetime
import numpy as np
import imageio
from tqdm import tqdm

def copy_group(group, new_group):
    """
    Recursively copy all groups and datasets from a group to a new group,
    including all attributes
    """
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            # Create a new subgroup in the output file
            subgroup = new_group.create_group(key)
            # Copy all attributes from the input subgroup to the output subgroup
            for attr_name, attr_value in group[key].attrs.items():
                subgroup.attrs[attr_name] = attr_value
            # Recursively copy all groups and datasets from the input subgroup
            copy_group(group[key], subgroup)
        elif isinstance(group[key], h5py.Dataset):
            # Copy the dataset from the input file to the output file
            group.copy(key, new_group)
            # Copy all attributes from the input dataset to the output dataset
            for attr_name, attr_value in group[key].attrs.items():
                new_group[key].attrs[attr_name] = attr_value


def copy_group_with_selected_traj(group, new_group, selected_traj):
    """
    Recursively copy all groups and datasets from a group to a new group,
    including all attributes
    """
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            if key == 'data':
                if "data" not in new_group.keys():
                    subgroup = new_group.create_group("data")
                else:
                    subgroup = new_group["data"]
                # Copy all attributes from the input subgroup to the output subgroup
                for attr_name, attr_value in group[key].attrs.items():
                    subgroup.attrs[attr_name] = attr_value

                # write dataset attributes (metadata)
                now = datetime.datetime.now()
                subgroup.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
                subgroup.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
                subgroup.attrs["env"] = "PickPlaceCansMilk"

                copy_group_with_selected_traj(group[key], new_group["data"], selected_traj)
            elif key in selected_traj or key == 'stats':
                # Create a new subgroup in the output file
                subgroup = new_group.create_group(key)
                # Copy all attributes from the input subgroup to the output subgroup
                for attr_name, attr_value in group[key].attrs.items():
                    subgroup.attrs[attr_name] = attr_value
                # Recursively copy all groups and datasets from the input subgroup
                copy_group(group[key], subgroup)
        elif isinstance(group[key], h5py.Dataset):
            # Copy the dataset from the input file to the output file
            group.copy(key, new_group)
            # Copy all attributes from the input dataset to the output dataset
            for attr_name, attr_value in group[key].attrs.items():
                new_group[key].attrs[attr_name] = attr_value


def count_collision(collision_frames, gap=5):
    if len(collision_frames) == 0:
        return 0, []
    left_of_collision = collision_frames[0]
    last_frame = collision_frames[0]
    collision_blocks = []
    for current_frame in collision_frames:
        if current_frame - last_frame > gap:
            collision_blocks.append((left_of_collision, last_frame))
            left_of_collision = current_frame
        last_frame = current_frame
    
    # if left_of_collision != collision_frames[-1]:
    collision_blocks.append((left_of_collision, current_frame))

    number_of_collision_blocks = len(collision_blocks)
    return number_of_collision_blocks, collision_blocks

def update_xml(dst, src):
    if src is None:
        return
    
    for key in dst.attrib.keys():
        if key in src.attrib and dst.attrib[key] != src.attrib[key]:
            dst.attrib[key] = src.attrib[key]

    for dst_child in dst:
        name = dst_child.get('name')
        update_xml(dst_child, src.find(f".//{dst_child.tag}[@name='{name}']"))

def update_state(dst, src):
    dst[1:8] = src[1:8]

    obj_states_dst = dst[10:52].reshape(6, 7)
    obj_states_src = src[31:38].reshape(1, 7)
    
    obj_id = np.argmin(np.sum(obj_states_dst - obj_states_src, axis=1))

    obj_states_dst[obj_id] = obj_states_src

    dst[10:52] = obj_states_dst.reshape(42)
    
    return obj_id, dst

def get_can_pos_from_old_state(state):
    can_pos = state[31:34]
    return can_pos

def save_demo_video(f, ep, video_path):
    writer = imageio.get_writer(video_path, fps=20)

    frontview_image = f["data/{}".format(ep)]["frontview_image"][()]

    num_actions = frontview_image.shape[0]

    pbar = tqdm(range(num_actions), total=num_actions)
    for j in pbar:
        frame = frontview_image[j]
        writer.append_data(frame)
        
    writer.close()
