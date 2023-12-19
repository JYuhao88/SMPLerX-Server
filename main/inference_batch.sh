#!/usr/bin/env bash
set -x

PARTITION=Zoetrope

INPUT_VIDEO=$1
FORMAT=$2
FPS=$3
CKPT=$4

GPUS=4
JOB_NAME=inference_${INPUT_VIDEO}

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4 # ${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

IMG_PATH=demo/images/${INPUT_VIDEO}
SAVE_DIR=demo/results/${INPUT_VIDEO}

# video to images
mkdir $IMG_PATH
mkdir $SAVE_DIR
ffmpeg -i demo/videos/${INPUT_VIDEO}.${FORMAT} -f image2 -vf fps=${FPS}/1 -qscale 0 demo/images/${INPUT_VIDEO}/%06d.jpg 

end_count=$(find "$IMG_PATH" -type f | wc -l)
echo $end_count

# inference
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m main.inference_batch \
--num_gpus ${GPUS_PER_NODE} \
--num_procs 80 \
--exp_name output/demo_${JOB_NAME} \
--pretrained_model ${CKPT} \
--agora_benchmark agora_model \
--img_path ${IMG_PATH} \
--start 1 \
--end  $end_count \
--output_folder ${SAVE_DIR} \
--show_verts \
--show_bbox \
--save_mesh \
--max_num_human 10 \
--object_show_score_llimit 0 \
--batch_size 1024 \


# images to video
ffmpeg -y -f image2 -r ${FPS} -i ${SAVE_DIR}/img_mesh/%06d.jpg -vcodec mjpeg -qscale 0 -pix_fmt yuv420p demo/results/${INPUT_VIDEO}.mp4
