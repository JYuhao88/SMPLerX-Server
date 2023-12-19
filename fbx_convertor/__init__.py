import pickle

SMPLX_SAVE_ITEMS = [
    "cam_trans",
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]

def save_phase_results(smplerx_results, filepath):
    with open(filepath, "wb") as f:
        results = {}
        results = {item: smplerx_results[item] for item in SMPLX_SAVE_ITEMS}
        pickle.dump(results, f)