MARK_LIST = ['1']

MARK_PARAMS = {
    '1': {
        'num_motors': 2,
        'num_wheels': 2,
        'urdf_name': "robots/turtlebot3_description/urdf/turtlebot3_burger.urdf",
        'motor_names': [
            "wheel_left_joint",
            "wheel_right_joint",
        ],
        'hardware': {
            'camera': {
                'default': 0,
                'cams': [
                    {
                        "name": "front",
                        "position": (0., 0., 0.25),
                        "target": (0.5, 0., 0.),
                    }
                ]
            },
            'lidar': {
                'default': 0,
                "lds": [
                    {
                        "name": "lidar",
                        "link": 5,
                        "angle_resolution_deg": 1,
                        "ray_direction_range_deg": (-180, 180),
                        "ray_length": 3.5,
                        "offset": 0.15,
                    }
                ]
            },
        }
    }
}
