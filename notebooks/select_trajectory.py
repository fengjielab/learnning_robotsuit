# %%
import h5py
import json
import random
import datetime

import robosuite as suite

from tqdm import tqdm

# %%
data_types = ["ph", "mh", "paired", "mg"]
output_hdf5 = "../../trajectory-preference-collection-tool/server/database/raw/selected_can_demo.hdf5"
data_types = ["ph", "mh", "paired", "mg"]
input_hdf5s = {}
for data_type in data_types:
    input_hdf5s[
        data_type
    ] = f"/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/{data_type}/new_env_demo_{data_type}.hdf5"


# %%
def check_length(num_frames, fps=20):
    # check if the length of the video is within 10s
    return num_frames / fps <= 8


def copy_group(group, new_group):
    """
    Recursively copy all groups and datasets from a group to a new group,
    including all attributes
    """
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            if key == "data":
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
                subgroup.attrs["time"] = "{}:{}:{}".format(
                    now.hour, now.minute, now.second
                )
                subgroup.attrs["env"] = "PickPlaceCansMilk"

                copy_group(group[key], new_group["data"])
            elif key in selected_traj:
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


env_name = "PickPlaceCansMilk"
with h5py.File(input_hdf5s["mg"], "r") as f:
    env_info = json.loads(f["data"].attrs["env_info"])
# env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
env = suite.make(env_name, **env_info)

# %%
# Open the output file in write mode
with h5py.File(output_hdf5, "w") as outfile:
    selected_traj = set()
    num_total_traj = 0

    for data_type in data_types:
        ep_list = []
        with h5py.File(input_hdf5s[data_type], "r") as f:
            if data_type == "mg":
                for ep in tqdm(
                    f["data"].keys(), desc=data_type + ": select suitable mg demos"
                ):
                    # check if eef path and object path are similar
                    states = f[f"data/{ep}"]["states"][()]

                    for i, state in enumerate(states):
                        env.sim.set_state_from_flattened(state)
                        env.sim.forward()
                        obj = env.objects[0]
                        contacts = env.get_contacts(obj)

                        eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
                        if (
                            eef_pos[0] < -0.1
                            or eef_pos[0] > 0.3
                            or eef_pos[1] < -0.25
                            or eef_pos[1] > 0.53
                            or eef_pos[2] < 0.8
                        ):
                            if ep in selected_traj:
                                selected_traj.remove(ep)
                            break

                        if (
                            "gripper0_finger1_pad_collision" in contacts
                            and "gripper0_finger2_pad_collision" in contacts
                        ):
                            selected_traj.add(ep)
            else:
                for ep in tqdm(
                    f["data"].keys(), desc=data_type + ": select < 8s demos"
                ):
                    num_total_traj += 1
                    states = f[f"data/{ep}"]["states"][()]
                    if check_length(states.shape[0]):
                        ep_list.append(ep)

                for ep in tqdm(ep_list, desc=data_type + ": select gripped demos"):
                    states = f[f"data/{ep}"]["states"][()]
                    for state in states:
                        env.sim.set_state_from_flattened(state)
                        env.sim.forward()
                        obj = env.objects[0]
                        if "gripper0_finger1_pad_collision" in env.get_contacts(
                            obj
                        ) and "gripper0_finger2_pad_collision" in env.get_contacts(obj):
                            selected_traj.add(ep)
                            break

            print(f"Total number of trajectories: {len(selected_traj)}")

            # Copy all attributes from the input file to the output file
            for attr_name, attr_value in f.attrs.items():
                outfile.attrs[attr_name] = attr_value

            # Recursively copy all groups and datasets from the input file to the output file
            copy_group(f, outfile)

# %%

with open(
    "../../trajectory-preference-collection-tool/server/database/raw/selected_traj.json",
    "w",
) as f:
    # indent=2 is not needed but makes the file human-readable
    # if the data is nested
    json.dump(list(selected_traj), f, indent=2)


# %%
print(f"Total number of selected trajectories: {len(selected_traj)}")
