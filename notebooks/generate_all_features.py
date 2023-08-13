# %%
"""
imports
"""
import os
import h5py
import json
import argparse
import datetime

import numpy as np
from tqdm import tqdm

import robosuite as suite
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# %%
"""
add arguments
"""
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_type", type=str, default="ph", help="data type to generate")
# parser.add_argument(
#     "--hdf5_dir",
#     type=str,
#     default="../../trajectory-preference-collection-tool/server/database/raw",
#     help="hdf5 directory",
# )
# parser.add_argument(
#     "--feature_dir",
#     type=str,
#     default="../../trajectory-preference-collection-tool/server/database/videos",
#     help="feature directory",
# )
# args = parser.parse_args()

# data_type = args.data_type
# hdf5_path = os.path.join(args.hdf5_dir, data_type, f"new_env_demo_{data_type}.hdf5")
# feature_path = os.path.join(args.feature_dir, "features.hdf5")

# %%
"""
load data and initialize variables
"""
data_type = "ph"
hdf5_path = os.path.join("data", "demonstrations", "can", data_type, f"new_env_demo_{data_type}.hdf5")
feature_path = os.path.join("database", "features.hdf5")

print(f"generating safety features in {data_type} data...")

f = h5py.File(hdf5_path, "r")
print(f"hdf5 file: {hdf5_path}")

f_writer = h5py.File(feature_path, "w")
grp = f_writer.create_group("data")
print(f"feature file: {feature_path}")

# %%
"""
initialize environment
"""
env_name = "PickPlaceCansMilk"
env_info = json.loads(f["data"].attrs["env_info"])
# env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
env = suite.make(env_name, **env_info)

# %% [markdown]
# P1. Safety:
# - P1.1. Collision-free;
# - P1.2. Safe-distance;
# - P1.3. No deformation / suitable contact force;

# %% [markdown]
# P2. Efficiency:
# - P2.1. Speed;
# - P2.2. Path Length;
# - P2.3. Success Rate;
# - P2.4. Time;
# - P2.5. Energy;

# %% [markdown]
# P3. Quality
# - P3.1. Smoothness;
# - P3.2. Stableness;
# - P3.3. Trajectory Shape;
# - P3.4. Orientation;

# %%
"""
generate features
"""
demo_names = list(f["data"].keys())

for ep in demo_names[:1]:
    states = f[f"data/{ep}/states"][()]
    env.reset()
    env.sim.set_state_from_flattened(states[-1])
    env.sim.forward()
    env._check_success()
    assert env.objects_in_bins[env.object_id] == 1

    geom_id2name = env.sim.model._geom_id2name

    num_collisions = 0
    prev_contacts = set()
    contact_arrays = []

    distances = []

    speeds = []

    path_lengths = [0, 0, 0]
    path23_start_id = 0
    path2_end_id = len(states) - 1
    path3_end_id = len(states) - 1
    eef_poses = []
    obj_poses = []

    eef_rot_eulers = []
    max_eef_rot_deg = 0
    obj_rot_eulers = []
    max_obj_rot_deg = 0
    rel_rot_eulers = []
    obj_to_eef_angles = []

    with tqdm(enumerate(states), total=len(states)) as pbar:
        for i, state in pbar:
            pbar.set_description(ep)

            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obj_to_use = env.objects[env.object_id]
            obj_geom_id = list(env.obj_geom_id.values())[env.object_id][0]

            ### Safety ###
            ## collision detection
            contacts = env.get_contacts(obj_to_use)
            contact_array = [item[0] in contacts or item[1] in contacts for item in geom_id2name.items()]
            contact_arrays.append(contact_array)
            if (
                np.all([("robot" not in str(c) and "gripper" not in str(c) and c != 7 and c != 21) for c in contacts])
                and len(contacts) != 0
            ):
                print(contacts)
                num_collisions += 1
            # pbar.set_postfix({"#collisions": num_collisions})

            ## distance to table and edge
            obj_pos = env.sim.data.geom_xpos[obj_geom_id].copy()
            table_front_edge_x = env.sim.data.geom_xpos[13][0]
            table_back_edge_x = env.sim.data.geom_xpos[15][0]
            table_left_edge_y = env.sim.data.geom_xpos[11][1]
            table_right_edge_y = env.sim.data.geom_xpos[23][1]
            table_z = env.bin1_pos[2]
            dis_to_table = obj_pos[2] - table_z
            dis_to_left_edge = obj_pos[1] - table_left_edge_y
            dis_to_right_edge = table_right_edge_y - obj_pos[1]
            dis_to_front_edge = table_front_edge_x - obj_pos[0]
            dis_to_back_edge = obj_pos[0] - table_back_edge_x
            distances.append([dis_to_table, dis_to_left_edge, dis_to_right_edge, dis_to_front_edge, dis_to_back_edge])
            
            ## TODO: get contact force


            ### Efficiency ###
            ## speed of eef
            eef_xvelp = env.sim.data.get_body_xvelp("gripper0_eef").copy()
            eef_xvelr = env.sim.data.get_body_xvelr("gripper0_eef").copy()
            speeds.append(np.concatenate([eef_xvelp, eef_xvelr]))

            ## path length: eef to object, eef to bin, object to bin
            ## TODO: accuracy
            eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
            if path23_start_id == 0:
                path_lengths[0] += np.linalg.norm(eef_pos - eef_poses[-1]) if i != 0 else 0
                if 7 in prev_contacts - contacts:
                    path23_start_id = i
            else:
                path_lengths[2] += np.linalg.norm(obj_pos - obj_poses[-1])
                path_lengths[1] += np.linalg.norm(eef_pos - eef_poses[-1])
                if 'gripper0_finger1_pad_collision' in prev_contacts - contacts or 21 in contacts - prev_contacts:
                    path2_end_id = i
                if 21 in prev_contacts - contacts:
                    path3_end_id = i
            eef_poses.append(eef_pos)
            obj_poses.append(obj_pos)
            # pbar.set_postfix({"path_lengths": path_lengths})

            ## TODO: pseudo energy

            ### Quality ###
            ## speed smoothness: "The smoothness value is a cumulative function of the end effector's linear and angular accelerations."
            # eef_accp = env.sim.data.get_body_xaccp("gripper0_eef") 
            # eef_accr = env.sim.data.get_body_xaccr("gripper0_eef")
            ## TODO: speed smoothness

            ## orientation
            # Get rotation matrices for end-effector and can
            ee_xmat = env.sim.data.get_body_xmat("gripper0_eef")
            obj_xmat = env.sim.data.get_body_xmat("Milk_main")

            # Create rotation objects from quaternions
            ee_rot = Rotation.from_matrix(ee_xmat)
            obj_rot = Rotation.from_matrix(obj_xmat)
            ee_to_obj_rot = ee_rot.inv() * obj_rot
            eef_rot_eulers.append(ee_rot.as_euler("xyz", degrees=True))
            obj_rot_eulers.append(obj_rot.as_euler("xyz", degrees=True))
            rel_rot_eulers.append(ee_to_obj_rot.as_euler("xyz", degrees=True))
            rel_rot_deg = np.rad2deg(ee_to_obj_rot.magnitude())
            obj_to_eef_angles.append(rel_rot_deg)

            # end of rollout of this episode
            prev_contacts = contacts

    ## time: eef to object time, eef to bin time, object to bin time, total time
    times = list(np.array([path23_start_id - 1, path2_end_id - path23_start_id, min(len(states), path3_end_id) - path23_start_id, len(states)]) / 20)

    # Fit a curve to the trajectory using cubic spline interpolation
    trajectory = np.array(eef_poses[path23_start_id:path2_end_id + 1])
    t = np.linspace(0, 1, len(trajectory))
    cs = CubicSpline(t, trajectory)

    # TODO: Generate a new trajectory by sampling points along the curve
    '''
    new_trajectory = []
    num_points = 50
    for i in range(num_points):
        t = i / (num_points - 1)
        ee_pos = cs(t)
        new_trajectory.append(ee_pos)
    '''
    ## trajectory stableness
    ## trajectory shape
            

    # write datasets
    ep_data_grp = grp.create_group(ep)
    ## Safety
    # collisions
    ep_data_grp.attrs["num_collisions"] = num_collisions
    ep_data_grp.attrs["geom_id2name"] = json.dumps(geom_id2name)
    ep_data_grp.create_dataset("contacts", data=np.array(contact_arrays))
    # distances
    ep_data_grp.attrs["distance_columns"] = ["distance_to_table", "distance_to_left_edge", "distance_to_right_edge", "distance_to_front_edge", "distance_to_back_edge"]
    ep_data_grp.create_dataset("distances", data=np.array(distances))
    # TODO: contact force

    ## Efficiency
    # speeds
    ep_data_grp.attrs["speed_columns"] = ["xvelp", "yvelp", "zvelp", "xvelr", "yvelr", "zvelr"]
    ep_data_grp.create_dataset("speeds", data=np.array(speeds))    
    # path lengths
    ep_data_grp.attrs["path_length_columns"] = ["eef_to_object", "eef_to_bin", "object_to_bin"]
    ep_data_grp.attrs["path_lengths"] = path_lengths
    ep_data_grp.attrs["path23_start_id"] = path23_start_id
    ep_data_grp.attrs["path2_end_id"] = path2_end_id
    ep_data_grp.attrs["path3_end_id"] = path3_end_id
    ep_data_grp.create_dataset("eef_poses", data=np.array(eef_poses))
    ep_data_grp.create_dataset("obj_poses", data=np.array(obj_poses))
    # times
    ep_data_grp.attrs["time_columns"] = ["eef_to_object", "eef_to_bin", "object_to_bin", "total"]
    ep_data_grp.attrs["times"] = times

    ## Quality
    # TODO: speed smoothness
    # TODO: trajectory stableness
    # TODO: trajectory shape
    # orientation
    ep_data_grp.create_dataset("eef_rot_eulers", data=np.array(eef_rot_eulers))
    ep_data_grp.create_dataset("obj_rot_eulers", data=np.array(obj_rot_eulers))
    ep_data_grp.create_dataset("rel_rot_eulers", data=np.array(rel_rot_eulers))
    ep_data_grp.create_dataset("obj_to_eef_angles", data=np.array(obj_to_eef_angles))

    # ep_data_grp.attrs["model_file"] = f["data"].attrs["model_file"]
 # %%
"""
generate keyframe informations
"""


# %%
# write dataset attributes (metadata)
now = datetime.datetime.now()
grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
grp.attrs["repository_version"] = suite.__version__

env_name = "PickPlaceCansMilk"
grp.attrs["env"] = env_name

env_info_dump = json.dumps(env_info)
grp.attrs["env_info"] = env_info_dump

f_writer.close()

f.close()
env.close()

# %%
