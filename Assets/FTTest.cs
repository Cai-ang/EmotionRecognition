
using System.Collections.Generic;
using Unity.XR.PXR;
using UnityEngine;
using TMPro;

public class FTTest : MonoBehaviour
{
    [Header("面部渲染器")]
    public SkinnedMeshRenderer skin;
    public SkinnedMeshRenderer tongueBlendShape;
    public SkinnedMeshRenderer leftEyeExample;
    public SkinnedMeshRenderer rightEyeExample;

    [Header("UI元素")]
    public GameObject text;
    public Transform TextParent;
    public Transform EmotionUIParent; // 情绪UI的父对象

    [Header("表情识别")]
    public MicroExpressionRecognizer emotionRecognizer;
    public EmotionFeedbackSystem emotionFeedback;

    private List<TMP_Text> texts = new List<TMP_Text>();

    private float[] blendShapeWeight = new float[72];

    private List<string> blendShapeList = new List<string>
    {
        "eyeLookDownLeft",
        "noseSneerLeft",
        "eyeLookInLeft",
        "browInnerUp",
        "browDownRight",
        "mouthClose",
        "mouthLowerDownRight",
        "jawOpen",
        "mouthUpperUpRight",
        "mouthShrugUpper",
        "mouthFunnel",
        "eyeLookInRight",
        "eyeLookDownRight",
        "noseSneerRight",
        "mouthRollUpper",
        "jawRight",
        "browDownLeft",
        "mouthShrugLower",
        "mouthRollLower",
        "mouthSmileLeft",
        "mouthPressLeft",
        "mouthSmileRight",
        "mouthPressRight",
        "mouthDimpleRight",
        "mouthLeft",
        "jawForward",
        "eyeSquintLeft",
        "mouthFrownLeft",
        "eyeBlinkLeft",
        "cheekSquintLeft",
        "browOuterUpLeft",
        "eyeLookUpLeft",
        "jawLeft",
        "mouthStretchLeft",
        "mouthPucker",
        "eyeLookUpRight",
        "browOuterUpRight",
        "cheekSquintRight",
        "eyeBlinkRight",
        "mouthUpperUpLeft",
        "mouthFrownRight",
        "eyeSquintRight",
        "mouthStretchRight",
        "cheekPuff",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeWideRight",
        "eyeWideLeft",
        "mouthRight",
        "mouthDimpleLeft",
        "mouthLowerDownLeft",
        "tongueOut",
        "viseme_PP",
        "viseme_CH",
        "viseme_o",
        "viseme_O",
        "viseme_i",
        "viseme_I",
        "viseme_RR",
        "viseme_XX",
        "viseme_aa",
        "viseme_FF",
        "viseme_u",
        "viseme_U",
        "viseme_TH",
        "viseme_kk",
        "viseme_SS",
        "viseme_e",
        "viseme_DD",
        "viseme_E",
        "viseme_nn",
        "viseme_sil",
    };

    private int[] indexList = new int[72];
    private int tongueIndex;
    private int leftLookDownIndex;
    private int leftLookUpIndex;
    private int leftLookInIndex;
    private int leftLookOutIndex;

    private int rightLookDownIndex;
    private int rightLookUpIndex;
    private int rightLookInIndex;
    private int rightLookOutIndex;

    private PxrFaceTrackingInfo faceTrackingInfo;
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("FTTest: Initializing face tracking...");
        
        // 检查设备能力
        if (!PXR_Plugin.System.UPxr_QueryDeviceAbilities(PxrDeviceAbilities.PxrTrackingModeFaceBit))
        {
            Debug.LogError("FTTest: Device does not support face tracking!");
            return;
        }
        Debug.Log("FTTest: Device supports face tracking");
        
        // 开始面部追踪
        PXR_MotionTracking.WantFaceTrackingService();
        FaceTrackingStartInfo info = new FaceTrackingStartInfo();
        info.mode = FaceTrackingMode.PXR_FTM_FACE_LIPS_BS;
        PXR_MotionTracking.StartFaceTracking(ref info);
        Debug.Log("FTTest: Face tracking started with mode: " + info.mode);

        // 初始化BlendShape索引
        for (int i = 0; i < indexList.Length; i++)
        {
            indexList[i] = skin.sharedMesh.GetBlendShapeIndex(blendShapeList[i]);
            if (TextParent != null) // 只有在有TextParent的情况下创建调试文本
            {
                GameObject textGO = GameObject.Instantiate(text, TextParent);
                texts.Add(textGO.GetComponent<TMP_Text>());
            }
        }

        
        tongueIndex = tongueBlendShape.sharedMesh.GetBlendShapeIndex("tongueOut");
        leftLookDownIndex = leftEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookDownLeft");
        leftLookUpIndex = leftEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookUpLeft");
        leftLookInIndex = leftEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookInLeft");
        leftLookOutIndex = leftEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookOutLeft");
        rightLookDownIndex = rightEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookDownRight");
        rightLookUpIndex = rightEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookUpRight");
        rightLookInIndex = rightEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookInRight");
        rightLookOutIndex = rightEyeExample.sharedMesh.GetBlendShapeIndex("eyeLookOutRight");
        
        // 初始化表情识别和反馈系统
        InitializeEmotionSystems();

    }
    
    // 初始化表情识别和反馈系统
    private void InitializeEmotionSystems()
    {
        // 如果没有指定表情识别器，尝试获取
        if (emotionRecognizer == null)
        {
            emotionRecognizer = GetComponent<MicroExpressionRecognizer>();
            if (emotionRecognizer == null)
            {
                Debug.LogWarning("FTTest: 未找到MicroExpressionRecognizer组件，已添加");
                emotionRecognizer = gameObject.AddComponent<MicroExpressionRecognizer>();
            }
        }
        
        // 如果没有指定情绪反馈系统，尝试获取
        if (emotionFeedback == null)
        {
            emotionFeedback = GetComponent<EmotionFeedbackSystem>();
            if (emotionFeedback == null)
            {
                Debug.LogWarning("FTTest: 未找到EmotionFeedbackSystem组件，已添加");
                emotionFeedback = gameObject.AddComponent<EmotionFeedbackSystem>();
            }
        }
        
        Debug.Log("FTTest: 表情识别和反馈系统已初始化");
    }

    // Update is called once per frame
    void Update()
    {
        if (PXR_Plugin.System.UPxr_QueryDeviceAbilities(PxrDeviceAbilities.PxrTrackingModeFaceBit))
        {
            switch (PXR_Manager.Instance.trackingMode)
            {
                case FaceTrackingMode.PXR_FTM_FACE_LIPS_BS:
                    PXR_System.GetFaceTrackingData(0, GetDataType.PXR_GET_FACELIP_DATA, ref faceTrackingInfo);

                    break;
                case FaceTrackingMode.PXR_FTM_FACE:
                    PXR_System.GetFaceTrackingData(0, GetDataType.PXR_GET_FACE_DATA, ref faceTrackingInfo);

                    break;
                case FaceTrackingMode.PXR_FTM_LIPS:
                    PXR_System.GetFaceTrackingData(0, GetDataType.PXR_GET_LIP_DATA, ref faceTrackingInfo);

                    break;
            }
            //blendShapeWeight = faceTrackingInfo.blendShapeWeight;
            unsafe
            {
                fixed (float* source = faceTrackingInfo.blendShapeWeight)
                {
                    for (int i = 0; i < 72; i++)
                    {
                        blendShapeWeight[i] = source[i];

                        texts[i].text = $"{blendShapeList[i]}\n{(int)(blendShapeWeight[i] * 120)}";

                        if (indexList[i] >= 0)
                        {
                            skin.SetBlendShapeWeight(indexList[i], 100 * blendShapeWeight[i]);
                        }


                    }
                }
            }
            

            
            tongueBlendShape.SetBlendShapeWeight(tongueIndex, 100 * blendShapeWeight[51]);
            
            leftEyeExample.SetBlendShapeWeight(leftLookUpIndex, 100 * blendShapeWeight[31]);
            leftEyeExample.SetBlendShapeWeight(leftLookDownIndex, 100 * blendShapeWeight[0]);
            leftEyeExample.SetBlendShapeWeight(leftLookInIndex, 100 * blendShapeWeight[2]);
            leftEyeExample.SetBlendShapeWeight(leftLookOutIndex, 100 * blendShapeWeight[44]);
            rightEyeExample.SetBlendShapeWeight(rightLookUpIndex, 100 * blendShapeWeight[35]);
            rightEyeExample.SetBlendShapeWeight(rightLookDownIndex, 100 * blendShapeWeight[12]);
            rightEyeExample.SetBlendShapeWeight(rightLookInIndex, 100 * blendShapeWeight[11]);
            rightEyeExample.SetBlendShapeWeight(rightLookOutIndex, 100 * blendShapeWeight[45]);
            
            // 处理表情识别数据
            if (emotionRecognizer != null)
            {
                emotionRecognizer.ProcessFaceTrackingData(blendShapeWeight, blendShapeList);
            }
        }
        else
        {
            if (Time.frameCount % 60 == 0) // 每60帧打印一次，避免日志过多
            {
                Debug.LogWarning("FTTest: Face tracking not available or not started!");
            }
        }
    }

    public void ToggleDebugUI()
    {
        TextParent.gameObject.SetActive(!TextParent.gameObject.activeSelf);
    }
    
    // 获取当前blendShape权重（用于其他组件访问）
    public float[] GetCurrentBlendShapeWeights()
    {
        return (float[])blendShapeWeight.Clone();
    }
    
    // 获取blendShape名称列表
    public List<string> GetBlendShapeNames()
    {
        return blendShapeList;
    }
}

