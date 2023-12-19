from copy import deepcopy

# from multiprocessing import Pool
import os
import torchvision.transforms as transforms
import numpy as np
import torch
import jax
import cv2
import tqdm

import asyncio

# import aiohttp
from concurrent.futures import ThreadPoolExecutor


transform = transforms.ToTensor()


def patch_image(original_img, bbox, model_cfg):
    from common.utils.preprocessing import generate_patch_image

    img, img2bb_trans, bb2img_trans = generate_patch_image(
        original_img, bbox, 1.0, 0.0, False, model_cfg.input_img_shape
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
    bbox_thr_x = bbox_thr * image_scale[0] if bbox_thr != -1 else 0
    bbox_thr_y = bbox_thr * image_scale[1] if bbox_thr != -1 else 0 
    for frame_id, mmtrack_result in enumerate(mmtrack_results):
        for human_id, mmtrack_bbox, score in zip(
            mmtrack_result["track_bboxes"][0][:, 0].tolist(),
            mmtrack_result["track_bboxes"][0][:, 1:5].tolist(),
            mmtrack_result["track_bboxes"][0][:, 5].tolist(),
        ):
            cur_bbox_x = mmtrack_bbox[2] - mmtrack_bbox[0]
            cur_bbox_y = mmtrack_bbox[3] - mmtrack_bbox[1]
            if ((cur_bbox_x < bbox_thr_x) or (cur_bbox_y <bbox_thr_y)):
                continue
            human_id = int(human_id)
            if human_id not in new_bboxes:
                new_bboxes[human_id] = {}
            box_xywh = mmtrack_bbox.copy()
            box_xywh[2] = cur_bbox_x
            box_xywh[3] = cur_bbox_y
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


def pack_inputs(original_imgs, mmtrack_boxes, batch_size, model_cfg):
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
                model_cfg.input_img_shape,
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


def load_img(path, order="RGB"):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == "RGB":
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def read_images(img_path, start, end, num_procs):
    # from common.utils.preprocessing import load_img

    imgs_path = []
    original_imgs = []
    imgs_path = findAllFile(img_path)
    imgs_path = sorted(imgs_path, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    imgs_path = imgs_path[start:] if end == -1 else imgs_path[start:end]
    print(f"Total {len(imgs_path)} images")
    # with Pool(num_procs) as pool:
    #     with tqdm(total=len(imgs_path)) as pbar:
    #         for img in pool.imap_unordered(load_img, imgs_path):
    #             original_imgs.append(img)
    #             pbar.update(1)
    for img_path in imgs_path:
        img = load_img(img_path)
        original_imgs.append(img)

    vis_imgs = deepcopy(original_imgs)
    img_scale = {
        "width": original_imgs[0].shape[1],
        "height": original_imgs[0].shape[0],
    }
    print("End read images")
    return imgs_path, original_imgs, vis_imgs, img_scale
