# %%
'''
imports
''' 
import os
import h5py
import json
import imageio
import argparse

import robosuite
import numpy as np

# %%
'''
add arguments
'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='ph', help='data type to generate')
# hdf5_dir
parser.add_argument('--hdf5_dir', type=str, default='../../trajectory-preference-collection-tool/server/database/raw', help='hdf5 directory')
parser.add_argument('--video_database', type=str, default='../../trajectory-preference-collection-tool/server/database/videos', help='video database')
args = parser.parse_args()

data_type = args.data_type
video_database = args.video_database

# %%
'''
load data and initialize variables
'''
print(f'generating {data_type} videos...')

hdf5_path = os.path.join(args.hdf5_dir, data_type, f'new_env_demo_{data_type}.hdf5')
f = h5py.File(hdf5_path, 'r')
print(f'hdf5 file: {hdf5_path}')


# %%
'''
initialize environment
'''
env_name = 'PickPlaceCansMilk'
env_info = json.loads(f['data'].attrs['env_info'])
env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
env = robosuite.make(
    env_name,
    **env_info
)

# %%
'''
get demos and generate videos
'''
demo_names = list(f['data'].keys())

for ep in demo_names:
    print(f'generating video for {ep}...')
    # initialize video writers
    video_dir = os.path.join(video_database, ep)
    os.path.exists(video_dir) or os.makedirs(video_dir)
    writers = {cam: imageio.get_writer(os.path.join(video_dir, f'{ep}_{cam}.mp4'), fps=20) for cam in env_info['camera_names']}

    states = f[f"data/{ep}/states"][()]
    env.reset()
    # env.sim.set_state_from_flattened(states[0])

    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        [writers[cam].append_data(env.sim.render(camera_name=cam, height=env_info["camera_heights"], width=env_info["camera_widths"])[::-1, :, :].astype(np.uint8))for cam in env_info['camera_names']]

    [writers[cam].close() for cam in env_info['camera_names']]
        

# %%
f.close()
env.close()

# %%
