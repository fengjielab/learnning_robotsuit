"""Camera profile registry for stage-2 local grasp experiments."""

from copy import deepcopy


CAMERA_PROFILES = {
    "realsense_d435i": {
        "vendor": "Intel RealSense",
        "model": "D435i",
        "family": "RealSense D400",
        "serial_number": None,
        "mount_hint": "eye_to_hand_or_wrist_mount",
        "streams": {
            "color": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "format": "bgr8",
            },
            "depth": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "format": "z16",
                "depth_scale_m_per_unit": 0.001,
                "min_distance_m": 0.105,
                "max_distance_m": 10.0,
            },
            "imu": {
                "gyro_hz": 200,
                "accel_hz": 250,
            },
        },
        "intrinsics_placeholder": {
            "fx": None,
            "fy": None,
            "cx": None,
            "cy": None,
        },
        "official_mount_reference": {
            "name": "visp_ros_franka_d435_reference",
            "description": (
                "Official ViSP Franka + RealSense D435 tutorial default eMc values "
                "for the franka-rs-D435-camera-holder.stl mount. Treat as a reference "
                "mount only; replace with your own hand-eye calibration for deployment."
            ),
            "source": {
                "tutorial": "https://docs.ros.org/en/melodic/api/visp_ros/html/tutorial-franka-coppeliasim.html",
                "example_code": "https://docs.ros.org/en/noetic/api/visp_ros/html/tutorial-franka-real-pbvs-apriltag_8cpp-example.html",
            },
            "ee_to_camera": {
                "translation_m": [0.0564668, -0.0375079, -0.150416],
                "theta_u_xyz": [0.0102548, -0.0012236, 1.5412],
                "quaternion_xyzw": [
                    0.004634771931637911,
                    -0.0005530197503171342,
                    0.6965626341850011,
                    0.7174808079074659,
                ],
            },
        },
        "extrinsics_placeholder": {
            "camera_to_robot_translation_m": None,
            "camera_to_robot_quat_xyzw": None,
        },
        "extrinsics_reference_for_sim_alignment": {
            "ee_to_camera_translation_m": [0.0564668, -0.0375079, -0.150416],
            "ee_to_camera_quat_xyzw": [
                0.004634771931637911,
                -0.0005530197503171342,
                0.6965626341850011,
                0.7174808079074659,
            ],
            "validated_mujoco_local_mount": {
                "description": (
                    "Validated robosuite-local eye_in_hand pose obtained by matching the "
                    "official ViSP Franka+D435 reference mount to Panda's right_hand "
                    "camera body in MuJoCo. A small training compensation is applied "
                    "to shift the object slightly right / lower in image and drop the "
                    "gripper a bit lower in frame while staying close to the official "
                    "side-mounted wrist view."
                ),
                "camera_parent_body": "robot0_right_hand",
                "cam_pos_m": [0.0564668, -0.073416, 0.0525079],
                "cam_quat_wxyz": [0.00223829, -0.99876797, 0.0149032, -0.04727896],
            },
            "reference_site_key": "grip_site",
            "reference_orientation_body": "gripper_root_body",
            "camera_frame_convention": "optical",
            "mujoco_camera_frame_conversion": "optical_to_mujoco_rx_pi",
            "robosuite_mount_frame_override": {
                "description": (
                    "Use the raw Panda gripper mount frame that exists before robosuite "
                    "applies gripper merge rotation offsets. This is the frame that best "
                    "matches the official ViSP Franka+D435 reference mount for sim alignment."
                ),
                "body_to_reference_translation_m": [0.0, 0.097, 0.0],
                "body_to_reference_quat_wxyz": [0.707107, 0.0, 0.0, -0.707107],
            },
        },
    }
}


def get_camera_profile(name: str = "realsense_d435i"):
    """Returns a deep copy of the named camera profile."""
    if name not in CAMERA_PROFILES:
        raise KeyError(f"Unknown camera profile: {name}")
    return deepcopy(CAMERA_PROFILES[name])
