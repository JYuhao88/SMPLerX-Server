from multiprocessing import Pool
import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import jax
from copy import deepcopy

from main.config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector
from mmtrack.apis import inference_mot, init_model
from mmtrack.models.mot import BaseMultiObjectTracker
from common.utils.inference_utils import process_mmdet_results, non_max_suppression
from fbx_convertor.smplx_motion_orig import HumanMotion
from fbx_convertor.smplx2fbx import FbxConvertor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, dest="num_gpus")
    parser.add_argument(
        "--num_procs", type=int, default=len(os.sched_getaffinity(0)) // 2
    )
    parser.add_argument("--exp_name", type=str, default="output/test")
    parser.add_argument("--pretrained_model", type=str, default=0)
    parser.add_argument("--testset", type=str, default="EHF")
    parser.add_argument("--agora_benchmark", type=str, default="na")
    parser.add_argument("--img_path", type=str, default="input.png")
    parser.add_argument("--start", type=str, default=1)
    parser.add_argument("--end", type=str, default=1)
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--demo_dataset", type=str, default="na")
    parser.add_argument("--demo_scene", type=str, default="all")
    parser.add_argument("--show_verts", action="store_true")
    parser.add_argument("--show_bbox", action="store_true")
    parser.add_argument("--save_mesh", action="store_true")
    # parser.add_argument("--multi_person", action="store_true")
    # parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--bbox_thr", type=int, default=50)
    parser.add_argument("--object_show_score_llimit", type=float, default=0.5)
    parser.add_argument("--human_appear_ratio_llimit", type=float, default=0.2)
    parser.add_argument("--human_confidence_llimit", type=float, default=0.5)
    parser.add_argument("--max_num_human", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()
    return args



transform = transforms.ToTensor()

def patch_image(original_img, bbox):
    from common.utils.preprocessing import generate_patch_image

    img, img2bb_trans, bb2img_trans = generate_patch_image(
        original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
    )
    img = transform(img.astype(np.float32)) / 255
    img = img.cuda()[None, :, :, :]
    return img


def get_bbox_from_detector(mmdet_box, img_width, img_height):
    from common.utils.preprocessing import process_bbox

    mmdet_box_xywh = np.zeros((4))
    mmdet_box_xywh[0] = mmdet_box[0]
    mmdet_box_xywh[1] = mmdet_box[1]
    mmdet_box_xywh[2] = abs(mmdet_box[2] - mmdet_box[0])
    mmdet_box_xywh[3] = abs(mmdet_box[3] - mmdet_box[1])

    # for bbox visualization
    start_point = (int(mmdet_box[0]), int(mmdet_box[1]))
    end_point = (int(mmdet_box[2]), int(mmdet_box[3]))

    bbox = process_bbox(mmdet_box_xywh, img_width, img_height)
    bbox_info = dict(
        mmdet_box_xywh=mmdet_box_xywh, start_point=start_point, end_point=end_point
    )
    return bbox, bbox_info


def construct_human_boxes(mmtrack_results, bbox_thr, image_scale, exp_cfg):
    # bboxes: n_frame, n_human, 6 (id, box, score)
    # return new_bboxes: Dict[human_id, Dict[frame_id, Dict[bbox, score]]]
    from common.utils.preprocessing import process_bbox

    new_bboxes = {}
    n_frame = len(mmtrack_results)
    for frame_id, mmtrack_result in enumerate(mmtrack_results):
        for human_id, mmtrack_bbox, score in zip(
            mmtrack_result["track_bboxes"][0][:, 0].tolist(),
            mmtrack_result["track_bboxes"][0][:, 1:5].tolist(),
            mmtrack_result["track_bboxes"][0][:, 5].tolist(),
        ):
            if mmtrack_bbox[2] < bbox_thr or mmtrack_bbox[3] < bbox_thr * 3:
                continue
            human_id = int(human_id)
            if human_id not in new_bboxes:
                new_bboxes[human_id] = {}
            box_xywh = mmtrack_bbox.copy()
            box_xywh[2] = box_xywh[2] - box_xywh[0]
            box_xywh[3] = box_xywh[3] - box_xywh[1]
            bbox = process_bbox(box_xywh, image_scale[0], image_scale[1])
            focal = [
                exp_cfg.focal[0] / exp_cfg.input_body_shape[1] * bbox[2],
                exp_cfg.focal[1] / exp_cfg.input_body_shape[0] * bbox[3],
            ]
            princpt = [
                exp_cfg.princpt[0] / exp_cfg.input_body_shape[1] * bbox[2] + bbox[0],
                exp_cfg.princpt[1] / exp_cfg.input_body_shape[0] * bbox[3] + bbox[1],
            ]
            new_bboxes[human_id][frame_id] = dict(
                mmtrack_bbox=mmtrack_bbox,
                bbox=bbox,
                score=score,
                box_xywh=box_xywh,
                focal=focal,
                princpt=princpt,
                start_point=(int(mmtrack_bbox[0]), int(mmtrack_bbox[1])),
                end_point=(int(mmtrack_bbox[2]), int(mmtrack_bbox[3])),
            )
    bbox_info = {}
    for human_id, bbox_dict in new_bboxes.items():
        bbox_info[human_id] = {}
        bbox_info[human_id]["human_appear_ratio"] = len(bbox_dict) / n_frame
        bbox_info[human_id]["human_confidence"] = np.mean(
            [bbox["score"] for bbox in bbox_dict.values()]
        ).item()
    return new_bboxes, bbox_info


def filter_human(mmtrack_boxes, bbox_info, human_appear_ratio_llimit, max_num_human):
    # mmtrack_boxes: Dict[human_id, Dict[frame_id, Dict[bbox, score]]]
    # return new_bboxes: Dict[human_id, Dict[frame_id, Dict[bbox, score]]]
    new_bboxes = {}
    new_bbox_info = {}
    for human_id, info in bbox_info.items():
        bbox = mmtrack_boxes[human_id]
        if info["human_appear_ratio"] < human_appear_ratio_llimit:
            continue
        new_bboxes[human_id] = mmtrack_boxes[human_id]
        new_bbox_info[human_id] = bbox_info[human_id]
    if len(new_bboxes) > max_num_human:
        final_objects = dict(
            sorted(
                new_bbox_info.items(),
                key=lambda item: item[1]["human_confidence"],
                reverse=True,
            )[:max_num_human]
        ).keys()
        new_bboxes = {human_id: new_bboxes[human_id] for human_id in final_objects}
        new_bbox_info = {
            human_id: new_bbox_info[human_id] for human_id in final_objects
        }
    return new_bboxes, new_bbox_info


def generate_inputs(original_imgs, mmtrack_boxes, batch_size):
    from common.utils.preprocessing import generate_patch_image

    imgs = {}
    for human_id, mmtrack_box in mmtrack_boxes.items():
        imgs[human_id] = {}
        for frame_id, bbox_info in mmtrack_box.items():
            img, _, _ = generate_patch_image(
                original_imgs[frame_id],
                bbox_info["bbox"],
                1.0,
                0.0,
                False,
                cfg.input_img_shape,
            )
            img = transform(img.astype(np.float32)) / 255
            imgs[human_id][frame_id] = img.cuda()[None, :, :, :]
    inputs_img, input_structure = jax.tree_util.tree_flatten(imgs)
    inputs_img = torch.cat(inputs_img, dim=0)
    inputs = [
        {"img": input_img} for input_img in torch.split(inputs_img, batch_size, dim=0)
    ]
    return inputs, input_structure


def render_meshs(
    vis_imgs, meshs, mmtrack_boxes, smplx_face, mesh_as_vertices, show_bbox
):
    from common.utils.vis import render_mesh

    for human_id, mmtrack_box in mmtrack_boxes.items():
        for frame_id, bbox_info in mmtrack_box.items():
            mesh = meshs[human_id][frame_id]
            vis_img = vis_imgs[frame_id]
            vis_img = render_mesh(
                vis_img,
                mesh[0].cpu().numpy(),
                smplx_face,
                {"focal": bbox_info["focal"], "princpt": bbox_info["princpt"]},
                mesh_as_vertices,
            )
            if show_bbox:
                start_point = bbox_info["start_point"]
                end_point = bbox_info["end_point"]
                vis_img = cv2.rectangle(
                    vis_img,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    thickness=2,
                )
            vis_imgs[frame_id] = vis_img
    return vis_imgs


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def recover_strucutre(inputs, input_structure):
    inputs = np.split(inputs, inputs.shape[0], axis=0)
    return jax.tree_util.tree_unflatten(input_structure, inputs)


def read_images(img_path, start, end, num_procs):
    from common.utils.preprocessing import load_img

    imgs_path = []
    original_imgs = []
    imgs_path = findAllFile(img_path)
    imgs_path = sorted(imgs_path, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    imgs_path = imgs_path[start:end]
    print(f"Total {len(imgs_path)} images")
    with Pool(num_procs) as pool:
        with tqdm(total=len(imgs_path)) as pbar:
            for img in pool.imap(load_img, imgs_path):
                original_imgs.append(img)
                pbar.update(1)
    vis_imgs = deepcopy(original_imgs)
    img_scale = {
        "width": original_imgs[0].shape[1],
        "height": original_imgs[0].shape[0],
    }
    print("End read images")
    return imgs_path, original_imgs, vis_imgs, img_scale


def main():
    args = parse_args()
    config_path = osp.join("main/config", f"config_{args.pretrained_model}.py")
    ckpt_path = osp.join("pretrained_models", f"{args.pretrained_model}.pth.tar")

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(
        args.testset,
        args.agora_benchmark,
        shapy_eval_split=None,
        pretrained_model_path=ckpt_path,
        use_cache=False,
    )
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from common.base import Demoer
    from common.utils.preprocessing import load_img, generate_patch_image
    from common.utils.vis import render_mesh, save_obj
    from common.utils.human_models import smpl_x

    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    start = int(args.start)
    end = start + int(args.end)

    ### mmdet init
    checkpoint_file = "pretrained_models/mmtracking/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth"
    config_file = "pretrained_models/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py"
    model: BaseMultiObjectTracker = init_model(
        config_file, checkpoint_file, device="cuda:0"
    )  # or device='cuda:0'

    os.makedirs(args.output_folder, exist_ok=True)

    imgs_path, original_imgs, vis_imgs, img_scale = read_images(
        args.img_path, start, end, args.num_procs
    )

    ## mmtrack inference
    print("mmtrack inference")
    mmtrack_results = []
    for frame_id, img_path in tqdm(enumerate(imgs_path)):
        mmtrack_result = inference_mot(model, img_path, frame_id=frame_id)
        mmtrack_results.append(mmtrack_result)
        model.show_result(
            img_path,
            mmtrack_result,
            out_file=os.path.join(
                args.output_folder, "img_track", f"{frame_id:05}.jpg"
            ),
            score_thr=args.object_show_score_llimit,
        )

    print("Construct human boxes")
    mmtrack_boxes, bbox_info = construct_human_boxes(
        mmtrack_results,
        args.bbox_thr,
        (img_scale["width"], img_scale["height"]),
        cfg,
    )
    print("Filter human boxes")
    mmtrack_boxes, bbox_info = filter_human(
        mmtrack_boxes,
        bbox_info,
        args.human_appear_ratio_llimit,
        args.max_num_human,
    )
    print("Generate inputs")
    inputs, input_structure = generate_inputs(
        original_imgs,
        mmtrack_boxes,
        args.batch_size,
    )
    del original_imgs

    print("Model inference")
    with torch.no_grad():
        outs = []
        for input_img in tqdm(inputs):
            out = demoer.model(input_img, {}, {}, "test")
            out = jax.tree_util.tree_map(lambda x: x.cpu(), out)
            outs.append(out)
        del inputs
        outs = jax.tree_util.tree_transpose(
            outer_treedef=jax.tree_util.tree_structure([0] * len(outs)),
            inner_treedef=jax.tree_util.tree_structure(outs[0]),
            pytree_to_transpose=outs,
        )
        outs = {k: torch.cat(v, dim=0) for k, v in outs.items()}

    meshs = jax.tree_util.tree_unflatten(
        input_structure, torch.split(outs["smplx_mesh_cam"], 1, dim=0)
    )
    print("Render mesh")
    vis_imgs = render_meshs(
        vis_imgs, meshs, mmtrack_boxes, smpl_x.face, args.show_verts, args.show_bbox
    )

    # save rendered image with all person
    print("Rendered image")
    for frame_id, vis_img in tqdm(enumerate(vis_imgs)):
        frame_name = imgs_path[frame_id].split("/")[-1]
        save_path_img = os.path.join(args.output_folder, "img_mesh")
        os.makedirs(save_path_img, exist_ok=True)
        cv2.imwrite(os.path.join(save_path_img, f"{frame_name}"), vis_img[:, :, ::-1])

    # save fbx
    smplx_poses = {}
    smplx_pose_names = [
        "cam_trans",
        "smplx_root_pose",
        "smplx_body_pose",
        "smplx_lhand_pose",
        "smplx_rhand_pose",
        "smplx_jaw_pose",
        "smplx_shape",
        "smplx_expr",
    ]
    for name in smplx_pose_names:
        smplx_poses[name] = recover_strucutre(outs[name].cpu().numpy(), input_structure)

    fbx_convertor = FbxConvertor(smplx_model_fbx_path="fbx_convertor/smplx-neutral.fbx", blend_type=[])
    smplx_motions = {}

    for human_id in mmtrack_boxes.keys():
        human_smplx_poses = {
            name: smplx_poses[name][human_id] for name in smplx_pose_names
        }
        smplx_motions[human_id] = HumanMotion(human_smplx_poses)
        smplx_motions[human_id].rotation_coordinate_transformation("unity3d")

    # for find right rotation order
    # transform_list = ["x/y/z", "nx/ny/nz", "nx/y/z", "x/ny/z", "x/y/nz", "nx/ny/z", "nx/y/nz", "x/ny/nz"]
    # for human_id in mmtrack_boxes.keys():
    #     human_smplx_poses = {
    #         name: smplx_poses[name][human_id] for name in smplx_pose_names
    #     }
    #     for trans_idx, transform in enumerate(transform_list):
    #         smplx_motions[human_id*len(transform_list) + trans_idx] = HumanMotion(human_smplx_poses)
    #         smplx_motions[human_id*len(transform_list) + trans_idx].rotation_coordinate_transformation(transform)

    fbx_convertor.smplxmotion2fbx(
        smplx_motions, save_path=os.path.join(args.output_folder, "motion.fbx")
    )

    

if __name__ == "__main__":
    main()
