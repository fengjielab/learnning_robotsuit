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

import datetime
import robosuite as suite

# add arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='mh')
parser.add_argument('--demo_path', type=str, default='../robosuite/models/assets/demonstrations/can/')
parser.add_argument('--new_demo_path', type=str, default='../notebooks/data/demonstrations/can/')
args = parser.parse_args()

dataset_type = args.dataset_type
demo_path = os.path.join(args.demo_path, dataset_type)
new_demo_path = os.path.join(args.new_demo_path, dataset_type)
# video_path = f'data/videos/demonstrations/{dataset_type}'
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


''' REVISION '''
# hdf5_new_path = os.path.join(demo_path, "new_dataset1.hdf5")
hdf5_new_path = os.path.join(new_demo_path, f"new_env_demo_{dataset_type}.hdf5")
f_write = h5py.File(hdf5_new_path, "w")
# store some metadata in the attributes of one group
grp = f_write.create_group("data")
''' REVISION '''

# env = robosuite.make(
#     'PickPlaceCansSingle', 
#     **env_info
# )

env = suite.make(
    'PickPlaceCansSingle', 
    **env_info
)

# pre-process
# hdf5_pre = os.path.join(demo_path, "new_dataset_paired.hdf5")
# f_1 = h5py.File(hdf5_pre, "r")
# demos_already = list(f_1["data"].keys())
# print(len(demos_already))

# list of all demonstrations episodes
demos = list(f["data"].keys())
demos = sorted(demos)
print(len(demos))

# demos = [x for x in demos_all if x not in demos_already]
# print(len(demos))

print("Playing back random episode... (press ESC to quit)")

# pre-stop
count_dict = {}

i = 0
while i < len(demos):
# while i < 21:
# while i < 2000:
    ep = demos[i]

    env.reset()
    successful = False
    observations = []
    
    # dst_xml = etree.fromstring(env.sim.model.get_xml())
    # model_xml = f["data/{}".format(ep)].attrs["model_file"]
    # src_xml = etree.fromstring(model_xml)
    # update_xml(dst_xml, src_xml)
    # xml = env.edit_model_xml(etree.tostring(dst_xml).decode())

    # state = env.sim.get_state().flatten()
    # print("original state:")
    # print(state.shape)
    states = f["data/{}/states".format(ep)][()]
    # print("original states:")
    # print(states.shape)
    # obj_id, state = update_state(state, states[0])
    can_pos = states[0][31:34]
    # print("updated state:")
    # print(state.shape)

    # env.reset_from_xml_string(xml)
    env.sim.reset()
    env._reset_internal(can_pos)

    # REVISION: initial state - state
    state_array = env.sim.get_state().flatten()

    # check the states again: env.sim.get_state().flatten()
    # if not np.all(np.equal(state, env.sim.get_state().flatten())):
    #     print("States are not equal in demo {}!".format(ep))
    #     break
    # if env.object_id != obj_id:
    #     print("Object id not match in demo {}!".format(ep))
    #     break


    # load the actions and play them back open-loop
    actions = np.array(f["data/{}/actions".format(ep)][()])
    # print(actions.shape)
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
        state_array = np.vstack((state_array, env.sim.get_state().flatten()))  #加state play back 
        # if j < num_actions - 1:
        #     # ensure that the actions deterministically lead to the same recorded states
        #     state_playback = env.sim.get_state().flatten()
        #     obj_id, state_data = update_state(state_playback.copy(), states[j + 1])
            
        #     # print((np.vstack(state_list)).shape)
        
        #     err = np.linalg.norm(state_data - state_playback)
        #     # pbar.set_description(f"Episode {ep} - {successful} - playback diverged by {err:.2f}")
        #     if not np.all(np.equal(state_data, state_playback)):
        #         err = np.linalg.norm(state_data - state_playback)
        #         # print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
        pbar.set_description(f"{i} - Episode {ep} - {'not ' if not successful else ''}successful")

    if successful:
        i += 1
        os.path.exists(os.path.join(video_path, ep)) or os.makedirs(os.path.join(video_path, ep))
        ep_data_grp = grp.create_group(f"{dataset_type}_{ep}")
        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=state_array)
        ep_data_grp.create_dataset("actions", data=np.array(actions))

        # for camera in env_info['camera_names']:
        #     os.path.exists(os.path.join(video_path, ep)) or os.makedirs(os.path.join(video_path, ep))
        #     writer = imageio.get_writer(os.path.join(video_path, ep, f"{env_name}_{camera}_video.mp4"), fps=20)
        #     for obs in observations:
        #         frame = obs[camera + "_image"].astype(np.uint8)
        #         writer.append_data(frame)
        #     writer.close()

# write dataset attributes (metadata)
now = datetime.datetime.now()
grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
grp.attrs["repository_version"] = suite.__version__

env_name = "PickPlaceCansSingle"
grp.attrs["env"] = env_name

env_info_dump = json.dumps(env_info)
grp.attrs["env_info"] = env_info_dump

f_write.close()
        
f.close()