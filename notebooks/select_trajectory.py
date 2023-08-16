# %%
import h5py
import json
import random

import robosuite as suite

from tqdm import tqdm

# %%
data_types = ['ph', 'mh', 'paired', 'mg']
files = {}
for data_type in data_types:
    files[data_type] = h5py.File(f"/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/{data_type}/new_env_demo_{data_type}.hdf5", "r")

# %%
def check_length(num_frames, fps=20):
    # check if the length of the video is within 10s
    return num_frames / fps <= 8


env_name = "PickPlaceCansMilk"
env_info = json.loads(files['ph']["data"].attrs["env_info"])
# env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
env = suite.make(env_name, **env_info)

# %%
gripped_traj = []
num_total_traj = 0
for data_type in data_types:
    f = files[data_type]

    short_traj = []  # Move short_traj inside the loop

    if data_type == 'mg':
        ep_list = random.sample(f["data"].keys(), 300)
    else:
        ep_list = f["data"].keys()
    for ep in tqdm(ep_list, desc=data_type + ': select < 8s demos'):
        num_total_traj += 1
        states = f[f"data/{ep}"]["states"][()]
        if check_length(states.shape[0]):
            short_traj.append(ep)

    for ep in tqdm(short_traj, desc=data_type + ': select gripped demos'):
        states = f[f"data/{ep}"]["states"][()]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obj = env.objects[0]
            if 'gripper0_finger1_pad_collision' in env.get_contacts(obj) and 'gripper0_finger2_pad_collision' in env.get_contacts(obj):
                gripped_traj.append(ep)
                break

    print(f"Total number of trajectories: {len(gripped_traj)}")

# %%

with open("../../trajectory-preference-collection-tool/server/database/raw/selected_traj.json", 'w') as f:
    # indent=2 is not needed but makes the file human-readable 
    # if the data is nested
    json.dump(gripped_traj, f, indent=2) 


# %%
print(f"Total number of selected trajectories: {len(gripped_traj)}")

# %%
for data_type in data_types:
    f = files[data_type]
    f.close()

# %%



