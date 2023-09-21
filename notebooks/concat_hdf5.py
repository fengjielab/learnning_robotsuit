# %%
import h5py
import json
import datetime

# %%
output_hdf5 = "../../trajectory-preference-collection-tool/server/database/raw/selected_can_demo.hdf5"
data_types = ['ph', 'mh', 'paired', 'mg']
input_hdf5s = {}
for data_type in data_types:
    input_hdf5s[data_type] = f"/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/{data_type}/new_env_demo_{data_type}.hdf5"

with open("../../trajectory-preference-collection-tool/server/database/raw/selected_traj.json", 'r') as f:
    selected_traj = json.load(f)

# %%
def copy_group(group, new_group):
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
                


# Open the output file in write mode
with h5py.File(output_hdf5, 'w') as outfile:

    # Loop through the input files
    for data_type in data_types:
        filename = input_hdf5s[data_type]

        # Open the input file in read mode
        with h5py.File(filename, 'r') as infile:

            # Copy all attributes from the input file to the output file
            for attr_name, attr_value in infile.attrs.items():
                outfile.attrs[attr_name] = attr_value

            # Recursively copy all groups and datasets from the input file to the output file
            copy_group(infile, outfile)

# %%
with h5py.File(output_hdf5, 'r') as concat_file:

    print(len(list(concat_file["data"].keys())))
    print(len(selected_traj))
    assert(len(list(concat_file["data"].keys())) == len(selected_traj))
