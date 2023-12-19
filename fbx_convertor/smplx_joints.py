class SMPLX_JOINTS:
    orig_joints_name = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",  # body joints
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",  # left hand joints
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",  # right hand joints
        "jaw",  # face jaw joint
    ]

    orig_joint_num = len(
        orig_joints_name
    )  # 53, 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)

    orig_body_flip_pairs = [
        (1, 2),
        (4, 5),
        (7, 8),
        (10, 11),
        (13, 14),
        (16, 17),
        (18, 19),
        (20, 21),  # body joints
    ]
    orig_hand_flip_pairs = [
        (22, 37),
        (23, 38),
        (24, 39),
        (25, 40),
        (26, 41),
        (27, 42),
        (28, 43),
        (29, 44),
        (30, 45),
        (31, 46),
        (32, 47),
        (33, 48),
        (34, 49),
        (35, 50),
        (36, 51),  # hand joints
    ]
    left_body_idx = [left_idx for left_idx, right_idx in orig_body_flip_pairs]
    right_body_idx = [right_idx for left_idx, right_idx in orig_body_flip_pairs]
    left_hand_idx = [left_idx for left_idx, right_idx in orig_hand_flip_pairs]
    right_hand_idx = [right_idx for left_idx, right_idx in orig_hand_flip_pairs]
    hand_idx = left_hand_idx + right_hand_idx

    orig_flip_pairs = orig_body_flip_pairs + orig_hand_flip_pairs
    left_joints_idx = [left_idx for left_idx, right_idx in orig_flip_pairs]
    right_joints_idx = [right_idx for left_idx, right_idx in orig_flip_pairs]

    center_joint_idx = [
        idx
        for idx, joint_name in enumerate(orig_joints_name)
        if "left_" not in joint_name and "right_" not in joint_name
    ]

    orig_root_joint_idx = [orig_joints_name.index("pelvis")]
    orig_body_joint_part = list(
        range(
            orig_joints_name.index("pelvis") + 1,
            orig_joints_name.index("right_wrist") + 1,
        )
    )
    orig_lhand_joint_part = list(
        range(
            orig_joints_name.index("left_index1"),
            orig_joints_name.index("left_thumb3") + 1,
        )
    )
    orig_rhand_joint_part = list(
        range(
            orig_joints_name.index("right_index1"),
            orig_joints_name.index("right_thumb3") + 1,
        )
    )
    orig_jaw_joint_part = list(
        range(orig_joints_name.index("jaw"), orig_joints_name.index("jaw") + 1)
    )

    num_betas = 10
    betas_names = [f"Shape{i:03}" for i in range(num_betas)]

    num_expressions = 10
    expressions_names = [f"Exp{i:03}" for i in range(num_expressions)]
