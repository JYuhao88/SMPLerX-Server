from typing import Dict, List
import fbx
from fbx_convertor import FbxCommon
from fbx_convertor.smplx_motion_orig import HumanMotion
from fbx_convertor.smplx_joints import SMPLX_JOINTS


def set_animation_curve(pTime, pAnimCurve, pValue, pCurve, actual_time):
    # Create keys on the animation curve.
    pTime.SetSecondDouble(actual_time)
    lKeyIndex = pCurve.KeyAdd(pTime)[0]
    pAnimCurve.KeySetValue(lKeyIndex, pValue)
    pAnimCurve.KeySetInterpolation(lKeyIndex, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
    # fbx.EFbxQuatInterpMode.eQuatInterpSlerp

ROTATION_ORDER = dict(
    xyz = fbx.EFbxRotationOrder.eEulerXYZ,
    xzy = fbx.EFbxRotationOrder.eEulerXZY,
    yzx = fbx.EFbxRotationOrder.eEulerYZX,
    yxz = fbx.EFbxRotationOrder.eEulerYXZ,
    zxy = fbx.EFbxRotationOrder.eEulerZXY,
    zyx = fbx.EFbxRotationOrder.eEulerZYX,
)

class FbxConvertor:
    def __init__(self, smplx_model_fbx_path, blend_type:list=["betas", "expressions"], fbx_rotation_order="zxy", fbx_file_format=0):
        self.smplx_model_fbx_path = smplx_model_fbx_path
        self.init_scene()
        self.blend_shape_idx = 0
        self.blend_type = blend_type
        self.fbx_rotation_order = fbx_rotation_order # 
        self.fbx_file_format = fbx_file_format # 0: binary, 1: ascii 2: encryped

    def smplxmotion2fbx(self, smplx_motions, save_path, fps=20):
        self.set_animation_curve(smplx_motions, fps)
        self.save_scene(save_path)
        self.reset_scene()

    def init_scene(self):
        # Prepare the FBX SDK.
        self.smplxFbxSdkManager, self.smplxFbxScene = FbxCommon.InitializeSdkObjects()
        # Load the scene.
        self.smplxFbxResult = FbxCommon.LoadScene(
            self.smplxFbxSdkManager, self.smplxFbxScene, self.smplx_model_fbx_path
        )

        # self.smplxSkeletonRoot = self.find_skeleton_node(self.smplxFbxScene.GetRootNode(), "root")

        self.smplxSkeletonRoot = (
            self.smplxFbxScene.GetRootNode().GetChild(1).GetChild(0)
        )
        # self.smplxMeshNode = self.find_skeleton_node(self.smplxFbxScene.GetRootNode(), "SMPLX-mesh-neutral")
        self.smplxMeshNode = self.smplxFbxScene.GetRootNode().GetChild(0)

    def set_animation_curve(self, smplx_motions: Dict[int, HumanMotion], fps: int = 24):
        for human_id, smplx_motion in smplx_motions.items():
            lAnimStackName = f"Human_{human_id:03}"
            lAnimStack = fbx.FbxAnimStack.Create(self.smplxFbxScene, lAnimStackName)
            lAnimLayer = fbx.FbxAnimLayer.Create(self.smplxFbxScene, "Base Layer")
            lAnimStack.AddMember(lAnimLayer)

            # Create the AnimCurve on the pelvis(actual root) joint's translation.
            self.set_translation_curve(smplx_motion, lAnimLayer, fps)

            # Create the AnimCurve on joint's rotation.
            self.set_rotation_curve(smplx_motion, lAnimLayer, fps)

            # Create the AnimCurve on betas.
            self.set_blendshape_curve(smplx_motion, lAnimLayer, fps)

    def save_scene(self, save_path):
        # Save the scene.
        self.smplxFbxResult = FbxCommon.SaveScene(
            self.smplxFbxSdkManager, self.smplxFbxScene, save_path, self.fbx_file_format,
        )

    def reset_scene(self):
        # Destroy all objects created by the FBX SDK.
        self.smplxFbxSdkManager.Destroy()

    def find_skeleton_node(self, pNode, pNodeName="root"):
        # Find the joint node of the skeleton root
        pNodeStack = [pNode]
        while len(pNodeStack) > 0:
            pNode = pNodeStack.pop()
            if (
                pNode.GetName() == pNodeName
                and pNode.GetNodeAttribute().GetAttributeType()
                == fbx.FbxNodeAttribute.EType.eSkeleton
            ):
                return pNode
            for i in range(pNode.GetChildCount()):
                pNodeStack.append(pNode.GetChild(i))
        raise Exception(f"Skeleton has no node named {pNodeName}")

    def set_translation_curve(
        self, smplx_motion: HumanMotion, pAnimLayer: fbx.FbxAnimLayer, fps: int
    ):
        pNode = self.find_skeleton_node(self.smplxSkeletonRoot, "pelvis")
        for chn_idx, chn in enumerate("XYZ"):
            lAnimCurve = pNode.LclTranslation.GetCurve(pAnimLayer, chn, True)
            lTime = fbx.FbxTime()
            lAnimCurve.KeyModifyBegin()
            for time_idx, trans_value in smplx_motion.smplx_trans.items():
                value = trans_value[chn_idx]
                actual_time = time_idx / fps
                set_animation_curve(
                    lTime, lAnimCurve, value, lAnimCurve, actual_time
                )
            lAnimCurve.KeyModifyEnd()

    def set_rotation_curve(
        self, smplx_motion: HumanMotion, pAnimLayer: fbx.FbxAnimLayer, fps: int
    ):
        for joint_idx, joint in enumerate(SMPLX_JOINTS.orig_joints_name):
            pNode = self.find_skeleton_node(self.smplxSkeletonRoot, joint)
            # pNode.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, ROTATION_ORDER[self.fbx_rotation_order])
            lAnimCurveNode = pNode.LclRotation.GetCurveNode(pAnimLayer, True)
            for chn_idx, chn in enumerate("XYZ"):
                lAnimCurve = fbx.FbxAnimCurve.Create(self.smplxFbxScene, chn)
                lTime = fbx.FbxTime()
                lAnimCurve.KeyModifyBegin()
                for time_idx, rot_value in smplx_motion.smplx_rotation.items():
                    value = rot_value[joint_idx, chn_idx]
                    actual_time = time_idx / fps
                    set_animation_curve(
                        lTime,
                        lAnimCurve,
                        value,
                        lAnimCurve,
                        actual_time,
                    )
                lAnimCurve.KeyModifyEnd()
                lAnimCurveNode.ConnectToChannel(lAnimCurve, chn)
        rot_filter = fbx.FbxAnimCurveFilterUnroll()
        rot_filter.SetForceAutoTangents(True)
        rot_filter.Apply(lAnimCurveNode)
        print("Complete setting rotation animation curve")

    def set_blendshape_curve(
        self,
        smplx_motion: HumanMotion,
        pAnimLayer: fbx.FbxAnimLayer,
        fps: int,
        
    ):
        # blend_type: betas, expressions, all
        lMeshAttribute = self.smplxMeshNode.GetNodeAttribute()

        if "betas" in self.blend_type:
            for chn_idx, chn in enumerate(SMPLX_JOINTS.betas_names):
                lShapeCurve = lMeshAttribute.GetShapeChannel(self.blend_shape_idx, chn_idx, pAnimLayer, True)
                lTime = fbx.FbxTime()
                lShapeCurve.KeyModifyBegin()
                for time_idx, value in smplx_motion.smplx_betas.items():
                    actual_time = time_idx / fps
                    set_animation_curve(
                        lTime,
                        lShapeCurve,
                        value[chn_idx],
                        lShapeCurve,
                        actual_time,
                    )
                lShapeCurve.KeyModifyEnd()
        
        if "expressions" in self.blend_type:
            start_idx = len(SMPLX_JOINTS.betas_names)
            for chn_idx, chn in enumerate(SMPLX_JOINTS.expressions_names):
                lShapeCurve = lMeshAttribute.GetShapeChannel(self.blend_shape_idx, start_idx + chn_idx, pAnimLayer, True)
                lTime = fbx.FbxTime()
                lShapeCurve.KeyModifyBegin()
                for time_idx, value in smplx_motion.smplx_expr.items():
                    actual_time = time_idx / fps
                    set_animation_curve(
                        lTime,
                        lShapeCurve,
                        value[chn_idx],
                        lShapeCurve,
                        actual_time,
                    )
                lShapeCurve.KeyModifyEnd()

