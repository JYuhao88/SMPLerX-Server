import os

MMTRACKING_PATH = "pretrained_models/mmtracking/"

video2motion_ckpt = [
    "smpler_x_b32.pth",
    "smpler_x_h32.pth",
    "smpler_x_l32.pth",
    "smpler_x_s32.pth",
]

tracker_ckpt = dict(
    ocsort=dict(
        ckpt=f"{MMTRACKING_PATH}ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth",
        config=f"{MMTRACKING_PATH}configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py",
    ),
    bytetrack_mot17=dict(
        ckpt=f"{MMTRACKING_PATH}bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth",
        config=f"{MMTRACKING_PATH}configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private.py",
    ),
    bytetrack_mot20=dict(
        ckpt=f"{MMTRACKING_PATH}bytetrack/bytetrack_yolox_x_crowdhuman_mot20-private_20220506_101040-9ce38a60.pth",
        config=f"{MMTRACKING_PATH}configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot20-private.py",
    ),
    deepsort=dict(
        ckpt=f"{MMTRACKING_PATH}deepsort/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth",
        reid_ckpt=f"{MMTRACKING_PATH}deepsort/tracktor_reid_r50_iter25245-a452f51f.pth",
        config=f"{MMTRACKING_PATH}configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py",
    ),
    tracktor_mot15=dict(
        ckpt = f"{MMTRACKING_PATH}tracktor/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040-ae733d0c.pth",
        reid_ckpt = f"{MMTRACKING_PATH}tracktor/reid_r50_6e_mot15_20210803_192157-65b5e2d7.pth",
        config = f"{MMTRACKING_PATH}configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot15-private-half.py",
    ),
    tracktor_mot16=dict(
        ckpt=f"{MMTRACKING_PATH}tracktor/faster-rcnn_r50_fpn_4e_mot16-half_20210804_001054-73477869.pth",
        reid_ckpt=f"{MMTRACKING_PATH}tracktor/reid_r50_6e_mot16_20210803_204826-1b3e3cfd.pth",
        config=f"{MMTRACKING_PATH}configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot16-private-half.py",
    ),
    tracktor_mot17=dict(
        ckpt=f"{MMTRACKING_PATH}tracktor/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth",
        reid_ckpt=f"{MMTRACKING_PATH}tracktor/reid_r50_6e_mot17-4bf6b63d.pth",
        config=f"{MMTRACKING_PATH}configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py",
    ),
    tracktor_mot20=dict(
        ckpt=f"{MMTRACKING_PATH}tracktor/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244-2c323fd1.pth",
        reid_ckpt=f"{MMTRACKING_PATH}tracktor/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth",
        config=f"{MMTRACKING_PATH}configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8e_mot20-public.py",
    ),
    qdtrack_mot17=dict(
        ckpt=f"{MMTRACKING_PATH}qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth",
        config=f"{MMTRACKING_PATH}configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py",
    ),
    qdtrack_mot17_crowdhumam=dict(
        ckpt=f"{MMTRACKING_PATH}qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453-68899b0a.pth",
        config=f"{MMTRACKING_PATH}configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17-private-half.py",
    ),
    qdtrack_lvis=dict(
        ckpt=f"{MMTRACKING_PATH}qdtrack/qdtrack_faster-rcnn_r101_fpn_24e_lvis_20220430_024513-88911daf.pth",
        config=f"{MMTRACKING_PATH}configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_24e_lvis.py",
    ),
    qdtrack_tao=dict(
        ckpt=f"{MMTRACKING_PATH}qdtrack/qdtrack_faster-rcnn_r101_fpn_12e_tao_20220613_211934-7cbf4062.pth",
        config=f"{MMTRACKING_PATH}configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_12e_tao.py",
    ),
)
