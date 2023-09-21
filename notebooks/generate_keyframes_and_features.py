"""
This script is used to generate keyframes and features for the demo videos.
P1. Safety
    P1.1. Collision-free; (keyframe & feature: collision keyframes + #collisions)
    P1.2. Distance; (keyframe & feature: highest point, nearest point to edge keyframes + distances)
    P1.3. Deformation / contact force; (feature: max eef force)
P2. Efficiency
    P2.1. Speed; (feature: average speed)
    P2.2. Path Length; (feature: length of paths)
    P2.3. Time; (keyframe: pick up point, release point & feature: total time)
    P2.4. Energy; (feature: pseudo-cost)
P3. Task Quality
    P3.1. Speed Smoothness; (feature)
    P3.2. Trajectory Smoothness; (feature)
    P3.3. Orientation; (feature: max angles)
    P3.4. Grasp Position; (feature: norm obj_to_eef)
"""
# %%
"""
imports
"""
import os
import h5py
import json
import imageio
import argparse
import datetime

import numpy as np
from tqdm import tqdm

import robosuite as suite

# %%
"""
add arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--hdf5_path",
    type=str,
    default="../../trajectory-preference-collection-tool/server/database/raw/user_study/sampled_can_demo.hdf5",
    help="hdf5 directory",
)
parser.add_argument(
    "--feature_path",
    type=str,
    default="../../trajectory-preference-collection-tool/server/database/raw/user_study/sampled_features.hdf5",
    help="feature directory",
)
parser.add_argument(
    "--keyframe_dir",
    type=str,
    default="../../trajectory-preference-collection-tool/server/database/keyframes",
    help="keyframe directory",
)
args = parser.parse_args()

hdf5_path = args.hdf5_path
feature_path = args.feature_path
keyframe_dir = args.keyframe_dir

# constants
TIME_PER_FRAME = 0.05
BUFFER_TIME = 0.25
IMAGE_SIZE = 1024
FEATURE_NAMES = [
    "num_collisions",
    "avg_speed",
    "max_height_to_table",
    "min_distance_to_edge",
    "max_ee_force",
    "reach_length",
    "grasp_length",
    "obj_path_length",
    "total_time",
    "pseudo_cost",
    "speed_smoothness",
    "trajectory_smoothness",
]

# %%
"""
load data and initialize variables
"""
print(f"generating keyframes...")

print(f"hdf5 file: {hdf5_path}")

print(f"feature file: {feature_path}")


# %%
def h5py_to_flattened_dict(h5py_file):
    d = {}
    for k, v in h5py_file.items():
        if isinstance(v, h5py.Dataset):
            value = v[()]
            d[k] = value.tolist() if isinstance(value, (np.ndarray, np.number)) else value
        elif isinstance(v, h5py.Group):
            if k in ["distances", "path_length"]:
                d.update(h5py_to_flattened_dict(v))
            else:
                d[k] = h5py_to_flattened_dict(v)
    return d


def write_keyframe_image(keyframe_path, state, env, camera_name="frontview"):
    with imageio.get_writer(
        keyframe_path,
        duration=1000 * TIME_PER_FRAME,
    ) as writer:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        writer.append_data(
            env.sim.render(
                camera_name=camera_name,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            )[
                ::-1, :, :
            ].astype(np.uint8)
        )


def most_anomalous_feature_id(feautres, feature_means, feature_stds):
    return most_anomalous_feature_id


# %%
with h5py.File(hdf5_path, "r") as demo_f:
    with h5py.File(feature_path, "r") as feature_f:
        # write stats group in feature_f to json file
        stats = h5py_to_flattened_dict(feature_f["stats"])
        stats_path = os.path.join(keyframe_dir, "feature_stats.json")
        with open(stats_path, "w") as stats_f:
            json.dump(stats, stats_f, indent=4)

        env_name = "PickPlaceCansMilk"
        env_info = json.loads(demo_f["data"].attrs["env_info"])
        env_info["camera_names"] = [
            "frontview",
            "birdview",
            "agentview",
            "robot0_eye_in_hand",
        ]
        env = suite.make(env_name, **env_info)

        demos = list(demo_f["data"].keys())
        feature_data_grp = feature_f["data"]
        with tqdm(demos) as pbar:
            for ep in pbar:
                states = demo_f[f"data/{ep}/states"][()]
                keyframe_path = os.path.join(keyframe_dir, ep)
                keyframe_dict = {
                    "collisions": [],
                    "highest_point": [],
                    "nearest_point_to_edge": [],
                    "pick_up_point": [],
                    "release_point": [],
                }
                feature_dict = {}
                if not os.path.exists(keyframe_path):
                    os.makedirs(keyframe_path)
                pbar.set_postfix_str(f"episode: {ep}")

                ### Safety ###

                ## collision ##
                # number of collisions
                num_collisions = int(feature_data_grp[ep]["num_collisions"][()])
                feature_dict["num_collisions"] = num_collisions
                # collision keyframes
                collision_blocks = feature_data_grp[ep]["collision_blocks"]
                for block in collision_blocks:
                    collision_keyframe_name = f"{ep}_collision_{block[0]}_{block[1]}.png"
                    write_keyframe_image(
                        os.path.join(keyframe_path, collision_keyframe_name),
                        states[block[0]],
                        env,
                        "agentview",
                    )
                    keyframe_dict["collisions"].append(
                        [
                            [
                                TIME_PER_FRAME * block[0] - BUFFER_TIME,
                                TIME_PER_FRAME * block[1] + BUFFER_TIME,
                            ],
                            os.path.join(
                                "/",
                                *keyframe_path.split("/")[-2:],
                                collision_keyframe_name,
                            ),
                        ]
                    )

                ## distance ##
                distances = feature_data_grp[ep]["distances"][()]

                # highest point
                highest_point_frame = np.argmax(distances[:, 0])
                highest_point = distances[highest_point_frame, 0]
                max_height_to_table = float(highest_point)
                feature_dict["max_height_to_table"] = max_height_to_table
                # write highest point keyframe
                highest_point_keyframe_name = f"{ep}_highest_point.png"
                write_keyframe_image(
                    os.path.join(keyframe_path, highest_point_keyframe_name),
                    states[highest_point_frame],
                    env,
                    "frontview",
                )
                keyframe_dict["highest_point"] = [
                    [
                        TIME_PER_FRAME * highest_point_frame - BUFFER_TIME,
                        TIME_PER_FRAME * highest_point_frame + BUFFER_TIME,
                    ],
                    os.path.join(
                        "/",
                        *keyframe_path.split("/")[-2:],
                        highest_point_keyframe_name,
                    ),
                ]

                # nearest point to edge
                [nearest_point_to_edge_frame, edge_id] = np.unravel_index(
                    np.argmin(distances[:, 1:]), distances[:, 1:].shape
                )
                edge = feature_f[f"data/{ep}"].attrs["distance_columns"][edge_id + 1].split("_")[-2]
                nearest_point_to_edge = distances[nearest_point_to_edge_frame, edge_id + 1]
                min_distance_to_edge = float(nearest_point_to_edge)
                feature_dict["min_distance_to_edge"] = min_distance_to_edge
                # write nearest point to edge keyframe
                nearest_point_to_edge_keyframe_name = f"{ep}_nearest_point_to_{edge}_edge.png"
                write_keyframe_image(
                    os.path.join(keyframe_path, nearest_point_to_edge_keyframe_name),
                    states[nearest_point_to_edge_frame],
                    env,
                    "birdview",
                )
                keyframe_dict["nearest_point_to_edge"] = [
                    [
                        TIME_PER_FRAME * nearest_point_to_edge_frame - BUFFER_TIME,
                        TIME_PER_FRAME * nearest_point_to_edge_frame + BUFFER_TIME,
                    ],
                    os.path.join(
                        "/",
                        *keyframe_path.split("/")[-2:],
                        nearest_point_to_edge_keyframe_name,
                    ),
                ]

                ## deformation / contact force ##
                # max eef force
                max_ee_force = float(np.max(np.linalg.norm(feature_data_grp[ep]["contact_force"][()], axis=-1)))
                feature_dict["max_ee_force"] = max_ee_force

                ### Efficiency ###
                ## speed ##
                # average speed
                avg_speed = float(np.mean(np.linalg.norm(feature_data_grp[ep]["speeds"][()], axis=-1)))
                feature_dict["avg_speed"] = avg_speed

                ## path length ##
                # length of paths
                # reach path
                reach_length = float(feature_data_grp[ep]["path_lengths"][()][0])
                feature_dict["reach_length"] = reach_length
                # grasped path
                grasp_length = float(feature_data_grp[ep]["path_lengths"][()][1])
                feature_dict["grasp_length"] = grasp_length
                # obj path
                obj_path_length = float(feature_data_grp[ep]["path_lengths"][()][2])
                feature_dict["obj_path_length"] = obj_path_length

                ## time ##
                # total time
                total_time = float(feature_data_grp[ep]["times"][()][-1])
                feature_dict["total_time"] = total_time
                # pick up point keyframe
                pick_up_point_frame = int(feature_data_grp[ep]["path23_start_id"][()])
                pick_up_point_keyframe_name = f"{ep}_pick_up_point.png"
                write_keyframe_image(
                    os.path.join(keyframe_path, pick_up_point_keyframe_name),
                    states[pick_up_point_frame],
                    env,
                    "agentview",
                )
                keyframe_dict["pick_up_point"] = [
                    [
                        TIME_PER_FRAME * pick_up_point_frame - BUFFER_TIME,
                        TIME_PER_FRAME * pick_up_point_frame + BUFFER_TIME,
                    ],
                    os.path.join(
                        "/",
                        *keyframe_path.split("/")[-2:],
                        pick_up_point_keyframe_name,
                    ),
                ]
                # release point keyframe
                release_point_frame = int(feature_data_grp[ep]["path2_end_id"][()])
                release_point_keyframe_name = f"{ep}_release_point.png"
                write_keyframe_image(
                    os.path.join(keyframe_path, release_point_keyframe_name),
                    states[release_point_frame],
                    env,
                    "agentview",
                )
                keyframe_dict["release_point"] = [
                    [
                        TIME_PER_FRAME * release_point_frame - BUFFER_TIME,
                        TIME_PER_FRAME * release_point_frame + BUFFER_TIME,
                    ],
                    os.path.join(
                        "/",
                        *keyframe_path.split("/")[-2:],
                        release_point_keyframe_name,
                    ),
                ]

                ## energy ##
                # pseudo-cost
                pseudo_cost = float(feature_data_grp[ep]["pseudo_cost"][()])
                feature_dict["pseudo_cost"] = pseudo_cost

                ### Task Quality ###
                ## speed smoothness ##
                # speed smoothness
                speed_smoothness = float(feature_data_grp[ep]["speed_smoothness"][()])
                feature_dict["speed_smoothness"] = speed_smoothness

                ## trajectory smoothness ##
                # trajectory smoothness
                trajectory_smoothness = float(feature_data_grp[ep]["trajectory_smoothness"][()])
                feature_dict["trajectory_smoothness"] = trajectory_smoothness

                local_vars = locals()
                features = [local_vars[feature_name] for feature_name in FEATURE_NAMES]
                feature_means = np.array([stats[feature_name]["mean"] for feature_name in FEATURE_NAMES])
                feature_stds = np.array([stats[feature_name]["std"] for feature_name in FEATURE_NAMES])
                
                max_z_score = -float("inf")
                most_anomalous_feature_id = None
                for feature, mean, std in zip(features, feature_means, feature_stds):
                    z_score = abs((feature - mean) / std)
                    if z_score > max_z_score:
                        max_z_score = z_score
                        most_anomalous_feature_id = features.index(feature)

                most_anomalous_feature_name = FEATURE_NAMES[most_anomalous_feature_id]

                # save features to feature dict displaying the most anomalous feature
                feature_dict["anomaly"] = most_anomalous_feature_name

                ### end of episode ###
                keyframe_json_path = os.path.join(keyframe_path, f"{ep}_keyframes.json")
                feature_json_path = os.path.join(keyframe_path, f"{ep}_features.json")
                # Convert int64 values to int values
                with open(keyframe_json_path, "w") as json_f:
                    json.dump(
                        keyframe_dict,
                        json_f,
                        indent=4,
                    )
                with open(feature_json_path, "w") as json_f:
                    json.dump(
                        feature_dict,
                        json_f,
                        indent=4,
                    )
