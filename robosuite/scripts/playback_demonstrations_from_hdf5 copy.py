"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np
from lxml import etree  
from notebooks.utils import update_xml, update_state

import robosuite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo_v141.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    # env_name = env_args['env_name']
    env_name = "PickPlaceCansSingle"
    env_info = env_args['env_kwargs']
    env_info['camera_names'] = ['frontview', 'birdview', 'agentview', 'robot0_eye_in_hand']
    env_info['has_renderer'] = True
    env_info['use_camera_obs'] = False
    env_info['has_offscreen_renderer'] = False

    env = robosuite.make(
        env_name,
        **env_info
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        ep = random.choice(demos)

        env.reset()
        
        # dst_xml = etree.fromstring(env.sim.model.get_xml())
        # model_xml = f["data/{}".format(ep)].attrs["model_file"]
        # src_xml = etree.fromstring(model_xml)
        # update_xml(dst_xml, src_xml)
        # xml = env.edit_model_xml(etree.tostring(dst_xml).decode())

        state = env.sim.get_state().flatten()
        states = f["data/{}/states".format(ep)][()]
        obj_id, state = update_state(state, states[0])

        # env.reset_from_xml_string(xml)
        env.sim.reset()
        env._reset_internal(obj_id)
        env.sim.set_state_from_flattened(state)
        
        # TODO: check the states again: env.sim.get_state().flatten()
        if not np.all(np.equal(state, env.sim.get_state().flatten())):
            print("States are not equal in demo {}!".format(ep))
            break
        if env.object_id != obj_id:
            print("Object id not match in demo {}!".format(ep))
            break

        env.sim.forward()

        # load the actions and play them back open-loop
        actions = np.array(f["data/{}/actions".format(ep)][()])
        num_actions = actions.shape[0]

        for j, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                print("Episode finished after {} timesteps".format(j + 1))


            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                obj_id, state_data = update_state(state_playback.copy(), states[j + 1])
                if not np.all(np.equal(state_data, state_playback)):
                    err = np.linalg.norm(state_data - state_playback)
                    print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

    f.close()
