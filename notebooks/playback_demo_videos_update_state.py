import json
import os
import random

import h5py
import numpy as np
from tqdm import tqdm

import time

import robosuite
import imageio
from pprint import pprint   
from lxml import etree
from utils import update_xml, update_state

demo_path = '../robosuite/models/assets/demonstrations/can/mh/'
video_path = 'data/demos'
hdf5_path = os.path.join(demo_path, "demo_v141.hdf5")
f = h5py.File(hdf5_path, "r")
env_args = json.loads(f["data"].attrs["env_args"])
env_name = env_args['env_name']
env_info = env_args['env_kwargs']
env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
env_info['has_renderer'] = False
env_info['use_camera_obs'] = True
env_info['has_offscreen_renderer'] = True
env_info['camera_heights'] = 1024
env_info['camera_widths'] = 1024

env = robosuite.make(
    'PickPlaceCansSingle', 
    **env_info
)

# list of all demonstrations episodes
demos = list(f["data"].keys())

print("Playing back random episode... (press ESC to quit)")

i = 20
while i < len(demos):
    ep = demos[i]
    env.reset()
    successful = False
    observations = []
    
    # dst_xml = etree.fromstring(env.sim.model.get_xml())
    # model_xml = f["data/{}".format(ep)].attrs["model_file"]
    # src_xml = etree.fromstring(model_xml)
    # update_xml(dst_xml, src_xml)
    # xml = env.edit_model_xml(etree.tostring(dst_xml).decode())

    state = env.sim.get_state().flatten()
    states = f["data/{}/states".format(ep)][()]
    obj_id, state = update_state(state, states[0])

    # env.reset_from_xml_string(xml)
    env.sim.reset()
    env._reset_internal(obj_id)
    env.sim.set_state_from_flattened(state)
    # env.sim.forward()

    # check the states again: env.sim.get_state().flatten()
    if not np.all(np.equal(state, env.sim.get_state().flatten())):
        print("States are not equal in demo {}!".format(ep))
        break
    if env.object_id != obj_id:
        print("Object id not match in demo {}!".format(ep))
        break


    # load the actions and play them back open-loop
    actions = np.array(f["data/{}/actions".format(ep)][()])
    num_actions = actions.shape[0]

    pbar = tqdm(enumerate(actions), total=num_actions)
    for j, action in pbar:
        pbar.set_description(f"")
        obs, reward, done, info = env.step(action)
        observations.append({camera + "_image": obs[camera + "_image"][::-1, :, :] for camera in env_info['camera_names']})
        # env.render()

        # if env._check_success():
            # if not successful:
                # print(f"Episode {ep} finished successful after {j + 1}/{len(actions)} steps")
                # successful = True
        # elif j == num_actions - 1:
            # print("Episode {} finished unsuccessful after {} steps".format(ep, j + 1))

        successful = env._check_success()

        if j < num_actions - 1:
            # ensure that the actions deterministically lead to the same recorded states
            state_playback = env.sim.get_state().flatten()
            obj_id, state_data = update_state(state_playback.copy(), states[j + 1])
            err = np.linalg.norm(state_data - state_playback)
            # pbar.set_description(f"Episode {ep} - {successful} - playback diverged by {err:.2f}")
            # if not np.all(np.equal(state_data, state_playback)):
            #     err = np.linalg.norm(state_data - state_playback)
                # print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
        pbar.set_description(f"Episode {ep} - {'not ' if not successful else ''}successful{f' - playback diverged by {err:.2f}' if j < num_actions - 1 else ''}")

    if successful:
        i += 1
        for camera in env_info['camera_names']:
            os.path.exists(os.path.join(video_path, ep)) or os.makedirs(os.path.join(video_path, ep))
            writer = imageio.get_writer(os.path.join(video_path, ep, f"{camera}_video.mp4"), fps=20)
            for obs in observations:
                frame = obs[camera + "_image"].astype(np.uint8)
                writer.append_data(frame)
            writer.close()

f.close()