import h5py
import os
import json
import h5py
import argparse
import numpy as np

from utils import copy_group_with_selected_traj

# args to force sampling new trajectories
parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", default=False)
args = parser.parse_args()

demo_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/selected_can_demo.hdf5"
feature_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/features.hdf5"

new_demo_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study/sampled_can_demo.hdf5"
new_feature_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study/sampled_features.hdf5"
sampled_demos_id_json_path = "/home/hanfang/repos/trajectory-preference-collection-tool/server/database/raw/user_study/sampled_demos_id.json"


# Load the features from the h5py file
with h5py.File(feature_path, "r") as feature_file:
    if os.path.exists(sampled_demos_id_json_path) and not args.force:
        print("Loading sampled demos from {}".format(sampled_demos_id_json_path))
        with open(sampled_demos_id_json_path, "r") as f:
            sampled_demos = json.load(f)
        num_sampled_demos = len(sampled_demos)
    else:
        demo_names = list(feature_file["data"].keys())
        # Safety
        num_collisions = np.array(
            [feature_file["data/{}/num_collisions".format(ep)][()] for ep in demo_names]
        )
        max_height_to_table = np.array(
            [
                np.max(feature_file["data/{}/distances".format(ep)][()][:, 0])
                for ep in demo_names
            ]
        )
        min_distance_to_edge = np.array(
            [
                np.min(feature_file["data/{}/distances".format(ep)][()][:, 1:])
                for ep in demo_names
            ]
        )
        max_force_magnitude = np.array(
            [
                np.max(
                    np.linalg.norm(
                        feature_file["data/{}/contact_force".format(ep)][()], axis=-1
                    )
                )
                for ep in demo_names
            ]
        )

        # Efficiency
        avg_speed_magnitude = np.array(
            [
                np.mean(
                    np.linalg.norm(
                        feature_file["data/{}/speeds".format(ep)][()], axis=-1
                    )
                )
                for ep in demo_names
            ]
        )
        reach_lengths = np.array(
            [
                feature_file["data/{}/path_lengths".format(ep)][()][0]
                for ep in demo_names
            ]
        )
        grasp_lengths = np.array(
            [
                feature_file["data/{}/path_lengths".format(ep)][()][1]
                for ep in demo_names
            ]
        )
        obj_path_lengths = np.array(
            [
                feature_file["data/{}/path_lengths".format(ep)][()][2]
                for ep in demo_names
            ]
        )
        times = np.array(
            [feature_file["data/{}/times".format(ep)][()][-1] for ep in demo_names]
        )
        pseudo_costs = np.array(
            [feature_file["data/{}/pseudo_cost".format(ep)][()] for ep in demo_names]
        )

        # Quality
        speed_smoothness = np.array(
            [
                feature_file["data/{}/speed_smoothness".format(ep)][()]
                for ep in demo_names
            ]
        )
        trajectory_smoothness = np.array(
            [
                feature_file["data/{}/trajectory_smoothness".format(ep)][()]
                for ep in demo_names
            ]
        )
        max_obj_to_eef_angles = np.array(
            [
                np.max(feature_file["data/{}/obj_to_eef_angles".format(ep)][()])
                for ep in demo_names
            ]
        )
        grasp_pos = np.array(
            [
                np.linalg.norm(feature_file["data/{}/grasp_pos".format(ep)][()])
                for ep in demo_names
            ]
        )

        # Combine the features into a single array
        features = np.column_stack(
            [
                num_collisions,
                max_height_to_table,
                min_distance_to_edge,
                max_force_magnitude,
                avg_speed_magnitude,
                reach_lengths,
                grasp_lengths,
                obj_path_lengths,
                times,
                pseudo_costs,
                speed_smoothness,
                trajectory_smoothness,
                max_obj_to_eef_angles,
                grasp_pos,
            ]
        )

        # Calculate the class distribution of the original dataset for each feature
        class_distributions = []
        discrete_features = np.zeros_like(features).astype(int)
        for i in range(features.shape[1]):
            # Discretize the feature into 10 bins
            bins = np.linspace(np.min(features[:, i]), np.max(features[:, i]), num=10)
            discrete_features[:, i] = np.digitize(features[:, i], bins).astype(int)

            # Calculate the class distribution of the original dataset for the discrete feature
            class_distribution = np.bincount(discrete_features[:, i])
            class_distributions.append(class_distribution)

        # Calculate the weights for each episode based on their class distribution for each feature
        weights = np.zeros(len(discrete_features))
        for i in range(features.shape[1]):
            weights += class_distributions[i][discrete_features[:, i]] / np.sum(
                class_distributions[i]
            )
        # means = np.var(features, axis=0)
        # weights = np.zeros(len(features))
        # for i in range(features.shape[1]):
        #     weights += np.linalg.norm(features[:, i] - means[i])

        # Normalize the weights
        weights /= np.sum(weights)

        # Sample 30 episodes with replacement using the calculated weights
        sampled_indices = np.random.choice(
            len(features), size=30, replace=False, p=weights
        )
        sampled_demos = np.array(demo_names)[sampled_indices]
        assert len(sampled_demos) == 30
        with open(sampled_demos_id_json_path, "w") as f:
            json.dump(sampled_demos.tolist(), f, indent=4)

    # write sampled demos features to a new hdf5 file
    with h5py.File(new_feature_path, "w") as new_feature_file:
        for attr_name, attr_value in feature_file.attrs.items():
            new_feature_file.attrs[attr_name] = attr_value

        copy_group_with_selected_traj(feature_file, new_feature_file, sampled_demos)


with h5py.File(demo_path, "r") as demo_file:
    # Save the sampled demonstrations
    with h5py.File(new_demo_path, "w") as new_demo_file:
        for attr_name, attr_value in demo_file.attrs.items():
            new_demo_file.attrs[attr_name] = attr_value

        copy_group_with_selected_traj(demo_file, new_demo_file, sampled_demos)

# check the length of the new hdf5 files
with h5py.File(new_demo_path, "r") as demo_file:
    assert len(demo_file["data"].keys()) == 30

with h5py.File(new_feature_path, "r") as feature_file:
    assert len(feature_file["data"].keys()) == 30
