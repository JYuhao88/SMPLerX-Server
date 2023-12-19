import datetime
from multiprocessing import Pool
import os
import os.path as osp
import pickle
import traceback
from typing import Optional, Union
import cv2
import torch.backends.cudnn as cudnn
import torch
import shortuuid
import jax
import zipfile
from fastapi import FastAPI, Body, File, UploadFile
from fastapi.responses import FileResponse, ORJSONResponse
import threading

from copy import deepcopy
import json
from main.config import cfg as model_cfg
from tqdm import tqdm
from pydantic import BaseModel
from mmtrack.apis import inference_mot, init_model

from main.server.model_info import video2motion_ckpt, tracker_ckpt
from main.server.utils import (
    construct_human_boxes,
    filter_human,
    pack_inputs,
    render_meshs,
    recover_strucutre,
    read_images,
)

from fbx_convertor.smplx_motion_orig import HumanMotion
from fbx_convertor.smplx2fbx import FbxConvertor, ROTATION_ORDER
from fbx_convertor import SMPLX_SAVE_ITEMS, save_phase_results

CACHE_FOLDER = "./cache"
EXP_NAME = "./record/"


class DefaultConfig(BaseModel):
    video2motion_model: str = "smpler_x_h32"
    tracker_model: str = "ocsort"
    fps_speed: Union[int, str] = "default"  # default (Get fps from video)
    fbx_rotation_order: str = "unity3d" # "blender" or "unity3d" 
    show_vertice: bool = True
    show_bbox: bool = True
    human_appear_ratio_llimit: float = 0.2
    human_confidence_llimit: float = 0.5
    max_num_human: int = 5
    bbox_thr: int = -1 # -1 (No filter), normal: 0 <= bbox_thr <= 1
    render_video: bool = False
    save_mmtrack_boxes: bool = False
    save_blendshape: list  = [] # "betas", "expressions" or []
    save_model_results: bool = False

    fixed_blendshape_method: Optional[str] = "confidence_max" # confidence_max, confidence_mean
    frame_index_init_zero: bool = True
    translation_init_origin: bool = True
    smooth_coeff: float = 0.9


default_cfg = DefaultConfig().model_dump()


class Video2MotionServer:
    def __init__(self) -> None:
        self.__dict__.update(default_cfg)
        self.num_gpus = 1
        self.gpu_idx = [0]
        self.num_procs = 1
        self.cache_folder = None
        self.video_name = None
        self.batch_size = 256
        cudnn.benchmark = True

        self.image_info = dict(
            path=None,
            orig_imgs=None,
            vis_imgs=None,
            img_scale=None,
        )
        self.set_config(default_cfg, first_setting=True)

    def set_config(self, config: dict, first_setting=False):
        cur_video2motion_model = self.video2motion_model
        cur_tracker_model = self.tracker_model

        for cfg in default_cfg.keys():
            if cfg not in config.keys():
                config[cfg] = default_cfg[cfg]
        self.__dict__.update(config)
        if self.fbx_rotation_order == "unity3d":
            self.fbx_rotation_order = "zxy"
        elif self.fbx_rotation_order == "blender":
            self.fbx_rotation_order = "xyz"
        elif self.fbx_rotation_order == "default":
            self.fbx_rotation_order = "xyz"
        else:
            assert self.fbx_rotation_order in list(ROTATION_ORDER.keys()), f"{self.fbx_rotation_order} is not a valid rotation order"

        if cur_video2motion_model != config["video2motion_model"] or first_setting:
            self._load_model(config["video2motion_model"], first_setting=first_setting)

        if cur_tracker_model != config["tracker_model"] or first_setting:
            self._load_tracker(config["tracker_model"])

    def _load_model(self, video2motion_model, first_setting=False):
        model_cfg_path = osp.join("main/config", f"config_{video2motion_model}.py")
        ckpt_path = osp.join("pretrained_models", f"{video2motion_model}.pth.tar")

        model_cfg.get_config_fromfile(model_cfg_path)
        model_cfg.update_test_config(
            testset="EHF",
            agora_benchmark="na",
            shapy_eval_split=None,
            pretrained_model_path=ckpt_path,
            use_cache=False,
        )
        model_cfg.update_config(self.num_gpus, exp_name=EXP_NAME)

        if first_setting:
            from common.base import Demoer

            self.pretrained_model = Demoer()
        self.pretrained_model._make_model()
        self.pretrained_model.model.eval()

    def _load_tracker(self, tracker_model):
        ckpt = tracker_ckpt[tracker_model]["ckpt"]
        cfg = tracker_ckpt[tracker_model]["config"]
        self.tracker = init_model(cfg, ckpt, device=f"cuda:{self.gpu_idx[0]}")

        if tracker_model.split("_")[0] in ["deepsort", "tracktor"]:
            self.tracker.cfg.model.reid.init_cfg.checkpoint = tracker_ckpt[
                tracker_model
            ]["reid_ckpt"]

    def split_video_to_frames(self):
        os.makedirs(f"{self.cache_folder}/images", exist_ok=True)
        os.makedirs(f"{self.cache_folder}/results", exist_ok=True)
        os.system(
            f"ffmpeg -i {CACHE_FOLDER}/{self.video_name} -f image2 -vf fps={self.fps}/1 -qscale 0 {self.cache_folder}/images/%06d.jpg "
        )

        print("Complete splitting video to frames")

        (
            self.image_info["path"],
            self.image_info["orig_imgs"],
            self.image_info["vis_imgs"],
            self.image_info["img_scale"],
        ) = read_images(f"{self.cache_folder}/images", 0, -1, self.num_procs)
        print("Complete reading images to memory")

    # frames to video
    def frames_to_video(self):
        os.system(
            f"ffmpeg -y -f image2 -r {self.fps} -i {self.cache_folder}/results/img_mesh/%06d.jpg -vcodec mjpeg -qscale 0 -pix_fmt yuv420p {self.cache_folder}/results/render_{self.video_name}"
        )

    def mmtrack_inference(self):
        ## mmtrack inference
        print("mmtrack inference")
        mmtrack_results = []
        for frame_id, img_path in enumerate(self.image_info["path"]):
            mmtrack_result = inference_mot(self.tracker, img_path, frame_id=frame_id)
            mmtrack_results.append(mmtrack_result)
            if self.save_mmtrack_boxes:
                self.tracker.show_result(
                    img_path,
                    mmtrack_result,
                    out_file=os.path.join(
                        f"{self.cache_folder}/results", "img_track", f"{frame_id:05}.jpg"
                    ),
                    score_thr=self.human_confidence_llimit,
                )
        return mmtrack_results

    def construct_human_boxes(self, mmtrack_results):
        print("Construct human boxes")
        mmtrack_boxes, bbox_info = construct_human_boxes(
            mmtrack_results,
            self.bbox_thr,
            (
                self.image_info["img_scale"]["width"],
                self.image_info["img_scale"]["height"],
            ),
            model_cfg,
        )
        return mmtrack_boxes, bbox_info

    def filter_human_boxes(self, mmtrack_boxes, bbox_info):
        print("Filter human boxes")
        mmtrack_boxes, bbox_info = filter_human(
            mmtrack_boxes,
            bbox_info,
            self.human_appear_ratio_llimit,
            self.max_num_human,
        )
        return mmtrack_boxes, bbox_info

    def inference(self, inputs):
        print("Model inference")
        with torch.no_grad():
            outs = []
            for input_img in inputs:
                out = self.pretrained_model.model(input_img, {}, {}, "test")
                out = jax.tree_util.tree_map(lambda x: x.cpu(), out)
                outs.append(out)
            del inputs
            outs = jax.tree_util.tree_transpose(
                outer_treedef=jax.tree_util.tree_structure([0] * len(outs)),
                inner_treedef=jax.tree_util.tree_structure(outs[0]),
                pytree_to_transpose=outs,
            )
            outs = {k: torch.cat(v, dim=0) for k, v in outs.items()}
        return outs

    def render(self, outs, vis_imgs, mmtrack_boxes, smplx_face):
        print("Render mesh")
        meshs = outs["smplx_mesh_cam"]
        vis_imgs = render_meshs(
            vis_imgs,
            meshs,
            mmtrack_boxes,
            smplx_face,
            self.show_vertice,
            self.show_bbox,
        )

        print("Rendered image")
        save_path_img = os.path.join(self.cache_folder, "results/img_mesh")
        os.makedirs(save_path_img, exist_ok=True)
        for frame_id, vis_img in enumerate(vis_imgs):
            frame_name = self.image_info["path"][frame_id].split("/")[-1]
            cv2.imwrite(
                os.path.join(save_path_img, f"{frame_name}"), vis_img[:, :, ::-1]
            )

    def save_fbx(self, outs, mmtrack_boxes):
        smplx_poses = {}
        smplx_poses = {k: outs[k] for k in SMPLX_SAVE_ITEMS}

        fbx_convertor = FbxConvertor(
            smplx_model_fbx_path="fbx_convertor/smplx-neutral.fbx", blend_type=self.save_blendshape,
            fbx_rotation_order= self.fbx_rotation_order
        )

        smplx_motions = {}
        for human_id in mmtrack_boxes.keys():
            human_smplx_poses = {
                name: smplx_poses[name][human_id] for name in SMPLX_SAVE_ITEMS
            }
            smplx_motions[human_id] = HumanMotion(human_smplx_poses)
            smplx_motions[human_id].rotation_coordinate_transformation(self.fbx_rotation_order)

        # hand_rot_method = ["x/y/z", "nx/y/z", "x/ny/z", "x/y/nz", "nx/ny/z", "nx/y/nz", "x/ny/nz", "nx/ny/nz"]
        # for human_id in mmtrack_boxes.keys():
        #     human_smplx_poses = {
        #         name: smplx_poses[name][human_id] for name in SMPLX_SAVE_ITEMS
        #     }
        #     for rot_id, rot_method in enumerate(hand_rot_method):
        #         smplx_motions[human_id * len(hand_rot_method) + rot_id] = HumanMotion(human_smplx_poses)
        #         smplx_motions[human_id * len(hand_rot_method) + rot_id].rotation_coordinate_transformation(rot_method)

        fbx_convertor.smplxmotion2fbx(
            smplx_motions,
            save_path=os.path.join(
                CACHE_FOLDER, self.video_name.split(".")[0] + ".fbx"
            ),
        )


    @staticmethod
    def get_video_fps(video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        return fps

    def zip_files(self):
        with zipfile.ZipFile(f"{self.cache_folder}/tmp.zip", "w") as zipf:
            fbx_filename = f"{self.video_name.split('.')[0]}.fbx"
            render_videoname = f"render_{self.video_name}"

            fbx_path = os.path.join(CACHE_FOLDER, fbx_filename)
            render_videopath = os.path.join(self.cache_folder, "results", render_videoname)
            if os.path.exists(fbx_path):
                zipf.write(fbx_path, arcname=fbx_filename)
            if os.path.exists(render_videopath):
                zipf.write(render_videopath, arcname=render_videoname)
        os.rename(
            f"{self.cache_folder}/tmp.zip",
            f"{CACHE_FOLDER}/{self.video_name.split('.')[0]}.zip",
        )

    def log_error(self, message):
        json.dump({"status": message}, open(f"{self.cache_folder}/status.json", "w"))

    def unpack_outputs(self, ouputs, input_structure):
        for k, v in ouputs.items():
            ouputs[k] = jax.tree_util.tree_unflatten(
                input_structure, torch.split(v, 1, dim=0)
            )
        return ouputs

    def video2motion(self, video_name, cache_folder):
        self.video_name = video_name
        self.cache_folder = cache_folder

        if self.fps_speed == "default":
            self.fps = self.get_video_fps(f"{CACHE_FOLDER}/{self.video_name}")
        else:
            message = "Not support fps speed"
            self.log_error(message)
            raise NotImplementedError(message)

        self.split_video_to_frames()

        mmtrack_results = self.mmtrack_inference()
        mmtrack_boxes, bbox_info = self.construct_human_boxes(mmtrack_results)
        mmtrack_boxes, bbox_info = self.filter_human_boxes(mmtrack_boxes, bbox_info)

        print("Pack inputs")
        if len(mmtrack_boxes) == 0:
            message = "No human detected, please change the other trakcer."
            self.log_error(message)
            raise Exception(message)

        inputs, input_structure = pack_inputs(
            self.image_info["orig_imgs"],
            mmtrack_boxes,
            self.batch_size,
            model_cfg,
        )
        self.image_info["orig_imgs"] = None

        outs = self.inference(inputs)
        outs = self.unpack_outputs(outs, input_structure)

        if self.render_video:
            from common.utils.human_models import smpl_x
            self.render(
                outs,
                self.image_info["vis_imgs"],
                mmtrack_boxes,
                smpl_x.face,
            )
            self.frames_to_video()

        if self.save_model_results:
            results_path = f"{self.cache_folder}/results.pkl"
            save_phase_results(outs, results_path)
                        
        # post processs
        outs = HumanMotion.curve_continuty(outs)

        save_phase_results(outs, f"{CACHE_FOLDER}/curve_continuity.pkl")
        if self.fixed_blendshape_method is not None:
            outs = HumanMotion.fixed_blendshape_method(outs, mmtrack_boxes, self.fixed_blendshape_method)
        if self.frame_index_init_zero:
            outs = HumanMotion.frame_index_init_zero(outs)
        if self.translation_init_origin:
            outs = HumanMotion.translation_init_origin(outs)
        # save_phase_results(outs, f"{CACHE_FOLDER}/translation_init_origin.pkl")

        outs = HumanMotion.frame_insertion(outs)
        # save_phase_results(outs, f"{CACHE_FOLDER}/frame_insertion.pkl")
        # if self.smooth_coeff is not None:
        #     outs = HumanMotion.smooth_motion(outs, self.smooth_coeff)
        # save_phase_results(outs, f"{CACHE_FOLDER}/smooth_motion.pkl")

        self.save_fbx(outs, mmtrack_boxes)
        print("Save Fbx Done")

        self.zip_files()
        print("Zip Files Done")

# server app
app = FastAPI(title="Convert video to motion Fbx File Server")

server = Video2MotionServer()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/v2m_config")
async def set_model_config(config: dict = default_cfg):
    server.set_config(config)
    return {"status": "Set model config"}


@app.post("/complete")
async def video2motion(video: UploadFile):
    os.makedirs(f"{CACHE_FOLDER}", exist_ok=True)
    os.system(f"rm -rf {CACHE_FOLDER}/*")

    # Get video
    video_content = await video.read()
    with open(f"{CACHE_FOLDER}/{video.filename}", "wb") as f:
        f.write(video_content)

    # Rename video
    video_format = video.filename.split(".")[-1]
    run_id = shortuuid.ShortUUID(
        alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz")
    ).random(8)
    os.system(
        f"mv {CACHE_FOLDER}/{video.filename} {CACHE_FOLDER}/{run_id}.{video_format}"
    )

    # Set cache folder
    cache_folder = (
        f"{CACHE_FOLDER}/"
        + run_id
        + "_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(cache_folder, exist_ok=True)

    threading.Thread(
        target=server.video2motion, args=(f"{run_id}.{video_format}", cache_folder)
    ).start()
    return run_id + ".zip", cache_folder


@app.get("/query")
async def query(filename: str, cache_folder: str):
    is_complete = os.path.exists(os.path.join(CACHE_FOLDER, filename))
    if is_complete:
        return {"status": "Complete"}
    else:
        status_file = os.path.join(cache_folder, "status.json")
        if os.path.exists(status_file):
            return json.load(open(status_file, "r"))
        else:
            return {"status": "Processing"}


@app.get("/download", response_class=FileResponse)
async def download(filename: str):
    return FileResponse(
        os.path.join(CACHE_FOLDER, filename),
        media_type="application/octet-stream",
        filename=filename,
    )
