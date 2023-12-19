# Build Smpler-X Server

## Download code
~~~bash
git clone https://github.com/JYuhao88/SMPLer-X.git
git checkout -b yuhao_loacal
~~~

## Download Smpler-X checkpoint
create pretrained_models folder under this project, and [download](https://pjlab-my.sharepoint.cn/personal/openmmlab_pjlab_org_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fopenmmlab%5Fpjlab%5Forg%5Fcn%2FDocuments%2Fopenmmlab%2Dshare%2Fdatasets%2Fhuman3d%2F%5Fmodels%2Fsmpler%5Fx%2Fsmpler%5Fx%5Fh32%2Epth%2Etar&parent=%2Fpersonal%2Fopenmmlab%5Fpjlab%5Forg%5Fcn%2FDocuments%2Fopenmmlab%2Dshare%2Fdatasets%2Fhuman3d%2F%5Fmodels%2Fsmpler%5Fx&ga=1)  Smpler-X checkpoint to pretrained_models folder

## Download Smplx Fbx
[Download](https://smpl-x.is.tue.mpg.de/download.php) smplx unity fbx file to fbx_convertor folder

## Download Smplx parameters
create human_model_files folder under "common/utils"

## Download 

The file structure should be like:
SMPLer-X/
├── common/
│   └── utils/
│       └── human_model_files/  # body model
│           ├── smpl/
│           │   ├──SMPL_NEUTRAL.pkl
│           │   ├──SMPL_MALE.pkl
│           │   └──SMPL_FEMALE.pkl
│           └── smplx/
│               ├──MANO_SMPLX_vertex_ids.pkl
│               ├──SMPL-X__FLAME_vertex_ids.npy
│               ├──SMPLX_NEUTRAL.pkl
│               ├──SMPLX_to_J14.pkl
│               ├──SMPLX_NEUTRAL.npz
│               ├──SMPLX_MALE.npz
│               └──SMPLX_FEMALE.npz
├── data/
├── main/
├── demo/  
│   ├── videos/       
│   ├── images/      
│   └── results/ 
├── pretrained_models/  # pretrained ViT-Pose, SMPLer_X and mmdet models
│   ├── mmtracking/
|   |   └──ocsort
|   │      └──/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth
│   ├── smpler_x_s32.pth.tar
│   ├── smpler_x_b32.pth.tar
│   ├── smpler_x_l32.pth.tar
│   ├── smpler_x_h32.pth.tar
│   ├── vitpose_small.pth
│   ├── vitpose_base.pth
│   ├── vitpose_large.pth
│   └── vitpose_huge.pth
└── dataset/  
    ├── AGORA/       
    ├── ARCTIC/      
    ├── BEDLAM/      
    ├── Behave/      
    ├── CHI3D/       
    ├── CrowdPose/   
    ├── EgoBody/     
    ├── EHF/         
    ├── FIT3D/                
    ├── GTA_Human2/           
    ├── Human36M/             
    ├── HumanSC3D/            
    ├── InstaVariety/         
    ├── LSPET/                
    ├── MPII/                 
    ├── MPI_INF_3DHP/         
    ├── MSCOCO/               
    ├── MTP/                    
    ├── MuCo/                   
    ├── OCHuman/                
    ├── PoseTrack/                
    ├── PROX/                   
    ├── PW3D/                   
    ├── RenBody/
    ├── RICH/
    ├── SPEC/
    ├── SSP3D/
    ├── SynBody/
    ├── Talkshow/
    ├── UBody/
    ├── UP3D/
    └── preprocessed_datasets/  # HumanData files

## Build docker image
~~~bash
docker build -t smplerx_server ../..
~~~