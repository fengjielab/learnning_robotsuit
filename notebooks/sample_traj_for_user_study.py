"""
sample_traj_for_user_study.py
    - This script is used to generate the sample trajectories for the user study.
    - The trajectories are saved in the "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study" folder.
    - We sample trajectories proportional to the number of demonstrations in each data type (e.g. ph, mh, paired, mg)
"""
import os
import h5py
import json
import random
import argparse
import datetime

# %%
# args to force sampling new trajectories
parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", default=False)
args = parser.parse_args()

# %%
demo_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/selected_can_demo.hdf5"
feature_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/features.hdf5"

new_demo_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study/sampled_can_demo.hdf5"
new_feature_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study/sampled_features.hdf5"
sampled_demos_id_json_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study/sampled_demos_id.json"

# %%
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

# %%
with h5py.File(demo_path, "r") as demo_file:
    if os.path.exists(sampled_demos_id_json_path) and not args.force:
        print("Loading sampled demos from {}".format(sampled_demos_id_json_path))
        with open(sampled_demos_id_json_path, "r") as f:
            sampled_demos = json.load(f)
        num_sampled_demos = len(sampled_demos)
    else:
        # Get the number of demonstrations in each data type
        demo_types = ["ph", "mh", "paired", "mg"]
        num_demos = {}
        demos = {"ph": [], "mh": [], "paired": [], "mg": []}
        all_demos = demo_file["data"].keys()
        for demo in all_demos:
            demo_type = demo.split("_")[0]
            if demo_type not in num_demos:
                num_demos[demo_type] = 0
            num_demos[demo_type] += 1
            demos[demo_type].append(demo)

        # Random sample trajectories proportional to the number of demonstrations in each data type
        num_sampled_demos = 30
        sampled_demos = [
            random.sample(
                demos[demo_type],
                round(num_sampled_demos * num_demos[demo_type] / len(all_demos)),
            )
            for demo_type in demo_types
        ]
        sampled_demos = [item for sublist in sampled_demos for item in sublist]
        assert len(sampled_demos) == num_sampled_demos
        with open(sampled_demos_id_json_path, "w") as f:
            json.dump(sampled_demos, f)

    # Save the sampled demonstrations
    with h5py.File(new_demo_path, "w") as new_demo_file:
        for attr_name, attr_value in demo_file.attrs.items():
            new_demo_file.attrs[attr_name] = attr_value

        copy_group_with_selected_traj(demo_file, new_demo_file, sampled_demos)

# %%
with h5py.File(feature_path, "r") as feature_file:
    # write sampled demos features to a new hdf5 file
    with h5py.File(new_feature_path, "w") as new_feature_file:
        for attr_name, attr_value in feature_file.attrs.items():
            new_feature_file.attrs[attr_name] = attr_value

        copy_group_with_selected_traj(feature_file, new_feature_file, sampled_demos)

# %%
# check the length of the new hdf5 files
with h5py.File(new_demo_path, "r") as demo_file:
    assert len(demo_file["data"].keys()) == num_sampled_demos
        
with h5py.File(new_feature_path, "r") as feature_file:
    assert len(feature_file["data"].keys()) == num_sampled_demos
