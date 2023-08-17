import numpy as np
import imageio
from tqdm import tqdm

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
