import os
import numpy as np
from lxml import etree
from pprint import pprint

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

def playback_demo(env, f, ep, video_path, save_video=True):
    writer = imageio.get_writer(video_path, fps=20)
    env.reset()
    
    # read the model xml, using the metadata stored in the attribute for this episode
    model_xml = f["data/{}".format(ep)].attrs["model_file"]

    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()

    states = f["data/{}/states".format(ep)][()]
    actions = f["data/{}/actions".format(ep)][()]

    env.sim.set_state_from_flattened(states[0])
    num_actions = actions.shape[0]

    pbar = tqdm(enumerate(actions), total=num_actions)
    for j, action in pbar:
        obs, _, _, _ = env.step(action)
        if save_video:
            frame = obs['frontview_image'][::-1, :, :].astype(np.uint8)
            writer.append_data(frame)

        if j < num_actions - 1:
            # ensure that the actions deterministically lead to the same recorded states
            state_playback = env.sim.get_state().flatten()
            if not np.all(np.equal(states[j + 1], state_playback)):
                err = np.linalg.norm(states[j + 1] - state_playback)
                print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
        
        env.sim.set_state_from_flattened(states[j])
