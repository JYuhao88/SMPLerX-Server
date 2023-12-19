# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:22.12-py3 

WORKDIR /smplerx_inference

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
RUN sh Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -b -u
ENV PATH /root/miniconda3/bin:$PATH

RUN pip install chardet
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get install -y python-opengl libosmesa6
RUN apt-get install -y zip unzip


# download SMPLer-X
COPY . .

ENV https_proxy="http://172.19.3.16:8888"
ENV http_proxy=http_proxy="http://172.19.3.16:8888"
RUN conda env create -f ./main/server/environment.yml

## MMCV
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.12.0/index.html
RUN git clone https://github.com/open-mmlab/mmtracking.git && pip install -r mmtracking/requirements/build.txt && conda run -n smplerx pip install -v -e ./mmtracking

RUN cd ./main/transformer_utils/ &&  conda run -n smplerx pip install -v -e . &&  conda run -n smplerx pip install torchgeometry --find-links .

RUN cd ./fbx_sdk && chmod ugo+x fbx202034_fbxsdk_linux && mkdir fbx202034_fbxsdk && echo -e "y\nyes\nn" | ./fbx202034_fbxsdk_linux ./fbx202034_fbxsdk
ENV FBXSDK_ROOT=/smplerx_inference/fbx_sdk/fbx202034_fbxsdk/
RUN  conda run -n smplerx python -m pip install --force-reinstall -v "sip==6.6.2" && cd ./fbx_sdk/fbx_python_sdk_202034 &&  conda run -n smplerx python -m pip install .

CMD ["conda", "run", "-n", "smplerx", "--no-capture-output", "uvicorn", "main.server:app", "--host=0.0.0.0", "--port=80"]


