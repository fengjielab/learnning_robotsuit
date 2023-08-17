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

from utils import count_collision

# %%
"""
add arguments
"""
parser = argparse.ArgumentParser()
# parser.add_argument("--data_type", type=str, default="ph", help="data type to generate")
parser.add_argument(
    "--hdf5_dir",
    type=str,
    default="../../trajectory-preference-collection-tool/server/database/raw",
    help="hdf5 directory",
)
parser.add_argument(
    "--feature_dir",
    type=str,
    default="../../trajectory-preference-collection-tool/server/database/raw",
    help="feature directory",
)
args = parser.parse_args()

# data_type = args.data_type
hdf5_path = os.path.join(args.hdf5_dir, "selected_can_demo.hdf5")
feature_path = os.path.join(args.feature_dir, "features_test.hdf5")

# %%
"""
load data and initialize variables
"""
# data_type = "ph"
# hdf5_path = os.path.join("data", "demonstrations", "can", data_type, f"new_env_demo_{data_type}.hdf5")
# feature_path = os.path.join("database", "features.hdf5")

print(f"generating features ...")

f = h5py.File(hdf5_path, "r")
print(f"hdf5 file: {hdf5_path}")

f_writer = h5py.File(feature_path, "w")
grp = f_writer.create_group("data")
stat_grp = f_writer.create_group("stats")
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
# - P2.3. Time;
# - P2.4. Energy;

# %% [markdown]
# P3. Quality
# - P3.1. Smoothness;
# - P3.2. Stableness;
# - P3.3. Orientation;
# - P3.4. Grasp Pose;

# %%
"""
generate features
"""
demo_names = list(f["data"].keys())

with tqdm(demo_names) as pbar:
    for ep in pbar:
        pbar.set_postfix({"episode": ep})
        states = f[f"data/{ep}/states"][()]
        actions = f[f"data/{ep}/actions"][()]
        env.reset()
        env.sim.set_state_from_flattened(states[-1])
        env.sim.forward()
        env.step(actions[-1])
        env._check_success()
        assert env.objects_in_bins[env.object_id] == 1

        geom_id2name = env.sim.model._geom_id2name

        gripped = actions[:, -1] > 0

        num_collisions = 0
        obj_contacts_list = []
        prev_object_contacts = set()
        gripper_contacts_list = []
        collisions_list = []
        collision_frames = []
        collision_blocks = []

        distances = []

        speeds = []

        path_lengths = [0, 0, 0]
        path23_start_id = 0
        path2_end_id = len(states) - 1
        path3_end_id = len(states) - 1
        eef_poses = []
        obj_poses = []

        ee_force = []

        pre_dis_vec = []
        trajectory_smoothness = 0

        joint_pos = []
        pre_joint_pos = []
        pseudo_cost = 0

        eef_rot_eulers = []
        max_eef_rot_deg = 0
        obj_rot_eulers = []
        max_obj_rot_deg = 0
        rel_rot_eulers = []
        obj_to_eef_angles = []

        grasp_pos = np.array([0, 0, 0])

        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obj_geom_id = list(env.obj_geom_id.values())[env.object_id][0]
            gripper_geom_ids = [
                env.sim.model.geom_name2id(name)
                for name in env.robots[0].gripper.contact_geoms
            ]

            ### Safety ###
            ## collision detection
            obj_contacts = set()
            gripper_contacts = set()
            for contact in env.sim.data.contact:
                if contact.geom1 == obj_geom_id:
                    obj_contacts.add(contact.geom2)
                elif contact.geom2 == obj_geom_id:
                    obj_contacts.add(contact.geom1)

                if (
                    contact.geom1 in gripper_geom_ids
                    or contact.geom2 in gripper_geom_ids
                ):
                    gripper_contacts.add(tuple(sorted([contact.geom1, contact.geom2])))

            # check gripped
            gripped[i] = gripped[i] and 101 in obj_contacts and 104 in obj_contacts

            # find collisions in the contacts: collision is defined as contacts that are not 1. between gripper and this can, or 2. between table and this can
            collisions = set()
            if gripped[i]:
                # add obj_contacts to collisions
                collisions |= set(
                    tuple(sorted([contact, obj_geom_id]))
                    for contact in obj_contacts
                    if (contact not in gripper_geom_ids and contact not in [7, 21])
                )
                # add gripper_contacts to collisions
                collisions |= set(
                    tuple(sorted(contact))
                    for contact in gripper_contacts
                    if (
                        not (
                            contact[0] == obj_geom_id and contact[1] in gripper_geom_ids
                        )
                        and not (
                            contact[1] == obj_geom_id and contact[0] in gripper_geom_ids
                        )
                    )
                )
            else:
                collisions |= set(
                    tuple(sorted([contact, obj_geom_id]))
                    for contact in obj_contacts
                    if (contact not in gripper_geom_ids and contact not in [7, 21])
                )
                collisions |= gripper_contacts

            if len(collisions) > 0:
                collision_frames.append(i)
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
            distances.append(
                [
                    dis_to_table,
                    dis_to_left_edge,
                    dis_to_right_edge,
                    dis_to_front_edge,
                    dis_to_back_edge,
                ]
            )

            ## get contact force
            ee_force.append(env.robots[0].ee_force)

            ### Efficiency ###
            ## speed of eef
            eef_xvelp = env.sim.data.get_body_xvelp("gripper0_eef").copy()
            eef_xvelr = env.sim.data.get_body_xvelr("gripper0_eef").copy()
            speeds.append(np.concatenate([eef_xvelp, eef_xvelr]))

            ## path length: eef to object, eef to bin, object to bin
            eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
            if path23_start_id == 0:
                path_lengths[0] += (
                    np.linalg.norm(eef_pos - eef_poses[-1]) if i != 0 else 0
                )
                if 7 in prev_object_contacts - obj_contacts and gripped[i]:
                    path23_start_id = i
            else:
                path_lengths[2] += np.linalg.norm(obj_pos - obj_poses[-1])
                path_lengths[1] += np.linalg.norm(eef_pos - eef_poses[-1])
                if (
                    not gripped[i]
                    and len(set(gripper_geom_ids).intersection(prev_object_contacts))
                    > 0
                    and len(set(gripper_geom_ids).intersection(obj_contacts)) == 0
                ):
                    path2_end_id = i
                if 21 in prev_object_contacts - obj_contacts:
                    path3_end_id = i
            # pbar.set_postfix({"path_lengths": path_lengths})

            ## pseudo energy
            joint_pos = env.robots[0]._joint_positions
            if i > 0:
                joint_move = np.array(joint_pos) - np.array(pre_joint_pos)
                distance = sum(abs(value) for value in joint_move)
                pseudo_cost += distance
            pre_joint_pos = joint_pos

            ### Quality ###

            ## trajectory smoothness
            dis_vec = None
            if i > 0:
                dis_vec = eef_pos - eef_poses[-1]
            if i > 1:  # 确保有两个向量来计算夹角
                dot_product = np.dot(dis_vec, pre_dis_vec)
                norms_product = np.linalg.norm(dis_vec) * np.linalg.norm(pre_dis_vec)
                cos_theta = np.clip(dot_product / norms_product, -1.0, 1.0)
                angle = np.arccos(cos_theta)  # 这是以弧度为单位的角度
                trajectory_smoothness += angle**2

            pre_dis_vec = dis_vec

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

            ## grasp position
            if (
                gripped[i]
                and path23_start_id != 0
                and i > path23_start_id
                and i < path2_end_id
            ):
                grasp_pos = (
                    obj_pos - eef_pos
                    if np.linalg.norm(obj_pos - eef_pos) > np.linalg.norm(grasp_pos)
                    else grasp_pos
                )

            # end of rollout of this episode
            eef_poses.append(eef_pos)
            obj_poses.append(obj_pos)
            prev_object_contacts = obj_contacts
            obj_contacts_list.append(list(obj_contacts))
            gripper_contacts_list.append(list(gripper_contacts))
            collisions_list.append(list(collisions))

        ## count collisions
        num_collisions, collision_blocks = count_collision(collision_frames, gap=3)

        ## time: eef to object time, eef to bin time, object to bin time, total time
        times = list(
            np.array(
                [
                    path23_start_id - 1,
                    path2_end_id - path23_start_id,
                    min(len(states), path3_end_id) - path23_start_id,
                    len(states),
                ]
            )
            / 20
        )

        ## speed smoothness: "The smoothness value is a cumulative function of the end effector's linear and angular accelerations."
        speed_smoothness = 0
        for i, state in enumerate(states):
            actions = np.array(f["data/{}/actions".format(ep)][()])
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.step(actions[i])
            cur_smoothness = np.sqrt(
                np.linalg.norm(env.robots[0].recent_ee_acc.current)
            )
            speed_smoothness += cur_smoothness

        speed_smoothness /= len(states)

        ## trajectory smoothness
        trajectory_smoothness /= len(states)

        # write datasets
        ep_data_grp = grp.create_group(ep)
        ## Safety
        # collisions
        ep_data_grp.attrs["num_collisions"] = num_collisions
        ep_data_grp.create_dataset("num_collisions", data=np.array(num_collisions))
        ep_data_grp.attrs["geom_id2name"] = json.dumps(geom_id2name)
        # ep_data_grp.create_dataset("obj_contacts", data=obj_contacts_list)
        # ep_data_grp.create_dataset("gripper_contacts", data=gripper_contacts_list)
        # ep_data_grp.create_dataset("collisions", data=collisions_list)
        ep_data_grp.create_dataset("collision_frames", data=np.array(collision_frames))
        ep_data_grp.create_dataset("collision_blocks", data=np.array(collision_blocks))
        # distances
        ep_data_grp.attrs["distance_columns"] = [
            "distance_to_table",
            "distance_to_left_edge",
            "distance_to_right_edge",
            "distance_to_front_edge",
            "distance_to_back_edge",
        ]
        ep_data_grp.create_dataset("distances", data=np.array(distances))
        # contact force
        ep_data_grp.create_dataset("contact_force", data=np.array(ee_force))
        ## Efficiency
        # speeds
        ep_data_grp.attrs["speed_columns"] = [
            "xvelp",
            "yvelp",
            "zvelp",
            "xvelr",
            "yvelr",
            "zvelr",
        ]
        ep_data_grp.create_dataset("speeds", data=np.array(speeds))
        # path lengths
        ep_data_grp.attrs["path_length_columns"] = [
            "eef_to_object",
            "eef_to_bin",
            "object_to_bin",
        ]
        ep_data_grp.create_dataset("path_lengths", data=np.array(path_lengths))
        ep_data_grp.attrs["path23_start_id"] = path23_start_id
        ep_data_grp.create_dataset("path23_start_id", data=path23_start_id)
        ep_data_grp.attrs["path2_end_id"] = path2_end_id
        ep_data_grp.create_dataset("path2_end_id", data=path2_end_id)
        ep_data_grp.attrs["path3_end_id"] = path3_end_id
        ep_data_grp.create_dataset("path3_end_id", data=path3_end_id)
        ep_data_grp.create_dataset("eef_poses", data=np.array(eef_poses))
        ep_data_grp.create_dataset("obj_poses", data=np.array(obj_poses))
        # times
        ep_data_grp.attrs["time_columns"] = [
            "eef_to_object",
            "eef_to_bin",
            "object_to_bin",
            "total",
        ]
        ep_data_grp.create_dataset("times", data=np.array(times))
        # pseudo cost
        ep_data_grp.create_dataset("pseudo_cost", data=np.array(pseudo_cost))

        ## Quality
        # speed smoothness
        ep_data_grp.create_dataset("speed_smoothness", data=np.array(speed_smoothness))
        # trajectory smoothness
        ep_data_grp.create_dataset(
            "trajectory_smoothness", data=np.array(trajectory_smoothness)
        )
        # orientation
        ep_data_grp.create_dataset("eef_rot_eulers", data=np.array(eef_rot_eulers))
        ep_data_grp.create_dataset("obj_rot_eulers", data=np.array(obj_rot_eulers))
        ep_data_grp.create_dataset("rel_rot_eulers", data=np.array(rel_rot_eulers))
        ep_data_grp.create_dataset(
            "obj_to_eef_angles", data=np.array(obj_to_eef_angles)
        )
        # grasp position
        ep_data_grp.create_dataset("grasp_pos", data=np.array(grasp_pos))


# %%
# calculate stats and write to 'stats' group
def create_stat_subgroup(group, subgroup_name, subgroup_data, description=None):
    subgroup = group.create_group(subgroup_name)
    if description is not None:
        subgroup.attrs["description"] = description
    subgroup.create_dataset("mean", data=np.mean(subgroup_data))
    subgroup.create_dataset("std", data=np.std(subgroup_data))
    subgroup.create_dataset("min", data=np.min(subgroup_data))
    subgroup.create_dataset("max", data=np.max(subgroup_data))
    subgroup.create_dataset("median", data=np.median(subgroup_data))
    subgroup.create_dataset("percentile_25", data=np.percentile(subgroup_data, 25))
    subgroup.create_dataset("percentile_75", data=np.percentile(subgroup_data, 75))


# Safety
# P1.1. num of collisions
num_collisions = np.array(
    [f_writer["data/{}/num_collisions".format(ep)][()] for ep in demo_names]
)
create_stat_subgroup(stat_grp, "num_collisions", num_collisions, "number of collisions")

# P1.2. highest point + nearest point to table edge
max_height_to_table = np.array(
    [np.max(f_writer["data/{}/distances".format(ep)][()][:, 0]) for ep in demo_names]
)
min_distance_to_edge = np.array(
    [np.min(f_writer["data/{}/distances".format(ep)][()][:, 1:]) for ep in demo_names]
)
dist_grp = stat_grp.create_group("distances")
create_stat_subgroup(
    dist_grp, "max_height_to_table", max_height_to_table, "max height to table"
)
create_stat_subgroup(
    dist_grp, "min_distance_to_edge", min_distance_to_edge, "min distance to edge"
)

# P1.3. max eef force
max_force_magnitude = np.array(
    [
        np.max(
            np.linalg.norm(f_writer["data/{}/contact_force".format(ep)][()], axis=-1)
        )
        for ep in demo_names
    ]
)
create_stat_subgroup(
    stat_grp, "max_ee_force", max_force_magnitude, "max end-effector force"
)

# Efficiency
# P2.1. average speed magnitude
avg_speed_magnitude = np.array(
    [
        np.mean(np.linalg.norm(f_writer["data/{}/speeds".format(ep)][()], axis=-1))
        for ep in demo_names
    ]
)
create_stat_subgroup(stat_grp, "avg_speed", avg_speed_magnitude, "average speed")

# P2.2. path lengths
reach_lengths = np.array(
    [f_writer["data/{}/path_lengths".format(ep)][()][0] for ep in demo_names]
)
grasp_lengths = np.array(
    [f_writer["data/{}/path_lengths".format(ep)][()][1] for ep in demo_names]
)
obj_path_lengths = np.array(
    [f_writer["data/{}/path_lengths".format(ep)][()][2] for ep in demo_names]
)
path_grp = stat_grp.create_group("path_length")
create_stat_subgroup(path_grp, "reach_length", reach_lengths, "reaching path length")
create_stat_subgroup(path_grp, "grasp_length", grasp_lengths, "grasping path length")
create_stat_subgroup(
    path_grp, "obj_path_length", obj_path_lengths, "object path length"
)

# P2.3. total times
times = np.array([f_writer["data/{}/times".format(ep)][()][-1] for ep in demo_names])
create_stat_subgroup(stat_grp, "total_time", times, "total time")

# P2.4. pseudo cost
pseudo_costs = np.array(
    [f_writer["data/{}/pseudo_cost".format(ep)][()] for ep in demo_names]
)
create_stat_subgroup(
    stat_grp, "pseudo_cost", pseudo_costs, "pseudo cost to approximate energy"
)

# Quality
# P3.1. speed smoothness
speed_smoothness = np.array(
    [f_writer["data/{}/speed_smoothness".format(ep)][()] for ep in demo_names]
)
create_stat_subgroup(stat_grp, "speed_smoothness", speed_smoothness, "smoothness")

# P3.2. trajectory smoothness
trajectory_smoothness = np.array(
    [f_writer["data/{}/trajectory_smoothness".format(ep)][()] for ep in demo_names]
)
create_stat_subgroup(
    stat_grp, "trajectory_smoothness", trajectory_smoothness, "trajectory smoothness"
)

# P3.3. orientation: max angle in degree between eef and object
max_obj_to_eef_angles = np.array(
    [np.max(f_writer["data/{}/obj_to_eef_angles".format(ep)][()]) for ep in demo_names]
)
create_stat_subgroup(
    stat_grp,
    "max_obj_to_eef_angle",
    max_obj_to_eef_angles,
    "max angle in degree between eef and object",
)

# P3.4. grasp position: norm of object to eef vector
grasp_pos = np.array(
    [np.linalg.norm(f_writer["data/{}/grasp_pos".format(ep)][()]) for ep in demo_names]
)
create_stat_subgroup(
    stat_grp, "grasp_pos", grasp_pos, "grasp position (length to the center of object)"
)

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
