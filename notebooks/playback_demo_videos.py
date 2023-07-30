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

# demo_path = '/Users/pearl/Desktop/robosuite/robosuite/models/assets/demonstrations/1688900181_543158'
demo_path = '../robosuite/models/assets/demonstrations/1688900181_543158'
# video_path = '/Users/pearl/Library/CloudStorage/OneDrive-HKUSTConnect/Research/RLfD/CHI2024/formative-study/dataset/selected/'
video_path = 'data/videos/selected/'
hdf5_path = os.path.join(demo_path, "demo.hdf5")
f = h5py.File(hdf5_path, "r")
env_name = f["data"].attrs["env"]
env_info = json.loads(f["data"].attrs["env_info"])
env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
env_info['has_renderer'] = False
env_info['use_camera_obs'] = True
env_info['ignore_done'] = True
env_info['has_offscreen_renderer'] = True
env_info['camera_heights'] = 1024
env_info['camera_widths'] = 1024

env = robosuite.make(
    **env_info
)

# list of all demonstrations episodes
demos = list(f["data"].keys())

print("Playing back episode... (press ESC to quit)")

i = 0
while i < len(demos):
    ep = demos[i]
    # if ep not in ["demo_2", "demo_5", "demo_10"]:
    #     i += 1
    #     continue
    env.reset()
    successful = False
    observations = []
    
    # read the model xml, using the metadata stored in the attribute for this episode
    model_xml = f["data/{}".format(ep)].attrs["model_file"]

    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()

    states = f["data/{}/states".format(ep)][()]

    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()


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
            state_data = states[j + 1]
            err = np.linalg.norm(state_data - state_playback)
            # pbar.set_description(f"Episode {ep} - {successful} - playback diverged by {err:.2f}")
            # if not np.all(np.equal(state_data, state_playback)):
            #     err = np.linalg.norm(state_data - state_playback)
                # print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
        pbar.set_description(f"Episode {ep} - {'not ' if not successful else ''}successful{f' - playback diverged by {err:.2f}' if j < num_actions - 1 else ''}")

    if successful:
        i += 1
        for camera in env_info['camera_names']:
            os.path.exists(os.path.join(video_path, ep+"_self_collected")) or os.makedirs(os.path.join(video_path, ep+"_self_collected"))
            writer = imageio.get_writer(os.path.join(video_path, ep+"_self_collected", f"{camera}_video.mp4"), fps=100)
            for obs in observations:
                frame = obs[camera + "_image"].astype(np.uint8)
                writer.append_data(frame)
            writer.close()

f.close()