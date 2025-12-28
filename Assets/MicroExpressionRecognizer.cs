using System.Collections.Generic;
using UnityEngine;

// 情绪类型枚举
public enum EmotionType
{
    Neutral,    // 中性
    Happy,      // 快乐
    Sad,        // 悲伤
    Angry,      // 愤怒
    Surprised,  // 惊讶
    Fear,       // 恐惧
    Disgusted,  // 厌恶
    Confused    // 困惑
}

// 微表情识别数据
[System.Serializable]
public class MicroExpressionData
{
    public EmotionType emotion;
    public float confidence;
    public Dictionary<string, float> blendShapeValues;
    public float detectionTime;
}

// 微表情识别器
public class MicroExpressionRecognizer : MonoBehaviour
{
    [Header("表情识别参数")]
    [Range(0.1f, 1.0f)]
    public float confidenceThreshold = 0.7f; // 识别置信度阈值
    
    [Range(0.5f, 5.0f)]
    public float expressionHoldTime = 1.5f; // 表情持续时间阈值
    
    [Range(1, 30)]
    public int smoothingFrames = 5; // 平滑处理的帧数
    
    [Header("调试信息")]
    public bool showDebugInfo = true;
    
    // 内部变量
    private Queue<float[]> recentBlendShapes = new Queue<float[]>();
    private EmotionType currentEmotion = EmotionType.Neutral;
    private EmotionType previousEmotion = EmotionType.Neutral;
    private float emotionStartTime = 0f;
    private float currentEmotionConfidence = 0f;
    
    // 表情特征权重 - 根据心理学研究中面部表情特征的重要性设置
    private Dictionary<string, float> emotionWeights = new Dictionary<string, float>
    {
        // 嘴部相关特征
        {"mouthSmileLeft", 0.8f},
        {"mouthSmileRight", 0.8f},
        {"mouthFrownLeft", 0.7f},
        {"mouthFrownRight", 0.7f},
        {"mouthPressLeft", 0.5f},
        {"mouthPressRight", 0.5f},
        {"mouthStretchLeft", 0.4f},
        {"mouthStretchRight", 0.4f},
        {"mouthShrugUpper", 0.5f},
        {"mouthLeft", 0.2f},
        {"mouthRight", 0.2f},
        {"jawOpen", 0.6f},
        
        // 眼部相关特征
        {"eyeSquintLeft", 0.4f}, // 既与快乐相关，也与愤怒相关
        {"eyeSquintRight", 0.4f}, // 既与快乐相关，也与愤怒相关
        {"eyeLookDownLeft", 0.5f},
        {"eyeLookDownRight", 0.5f},
        {"eyeWideLeft", 0.65f}, // 既与惊讶相关，也与恐惧相关
        {"eyeWideRight", 0.65f}, // 既与惊讶相关，也与恐惧相关
        {"eyeLookInLeft", 0.3f},
        {"eyeLookInRight", 0.3f},
        
        // 眉毛相关特征
        {"browInnerUp", 0.4f}, // 既与悲伤相关，也与恐惧相关，还与困惑相关
        {"browOuterUpLeft", 0.4f}, // 既与惊讶相关，也与困惑相关
        {"browOuterUpRight", 0.4f}, // 既与惊讶相关，也与困惑相关
        {"browDownLeft", 0.6f},
        {"browDownRight", 0.6f},
        
        // 鼻部相关特征
        {"noseSneerLeft", 0.7f},
        {"noseSneerRight", 0.7f}
    };
    
    // 事件委托
    public delegate void EmotionChangedHandler(EmotionType newEmotion, float confidence);
    public event EmotionChangedHandler OnEmotionChanged;
    
    // 获取当前情绪
    public EmotionType GetCurrentEmotion()
    {
        return currentEmotion;
    }
    
    // 获取当前情绪置信度
    public float GetCurrentEmotionConfidence()
    {
        return currentEmotionConfidence;
    }
    
    // 处理新的面部追踪数据
    public void ProcessFaceTrackingData(float[] blendShapeWeights, List<string> blendShapeNames)
    {
        // 平滑处理
        if (recentBlendShapes.Count >= smoothingFrames)
        {
            recentBlendShapes.Dequeue();
        }
        recentBlendShapes.Enqueue((float[])blendShapeWeights.Clone());
        
        // 如果数据不足，返回
        if (recentBlendShapes.Count < smoothingFrames)
            return;
            
        // 计算平均值
        float[] averagedWeights = CalculateAveragedBlendShapes();
        
        // 识别表情
        EmotionType detectedEmotion = RecognizeEmotion(averagedWeights, blendShapeNames);
        float confidence = CalculateEmotionConfidence(detectedEmotion, averagedWeights, blendShapeNames);
        
        // 更新当前情绪
        UpdateCurrentEmotion(detectedEmotion, confidence);
    }
    
    // 计算平均blendshape值
    private float[] CalculateAveragedBlendShapes()
    {
        float[] average = new float[72];
        
        foreach (float[] frame in recentBlendShapes)
        {
            for (int i = 0; i < frame.Length; i++)
            {
                average[i] += frame[i];
            }
        }
        
        for (int i = 0; i < average.Length; i++)
        {
            average[i] /= recentBlendShapes.Count;
        }
        
        return average;
    }
    
    // 识别表情
    private EmotionType RecognizeEmotion(float[] blendShapeWeights, List<string> blendShapeNames)
    {
        Dictionary<EmotionType, float> emotionScores = new Dictionary<EmotionType, float>();
        
        // 初始化所有情绪得分为0
        emotionScores[EmotionType.Neutral] = 0f;
        emotionScores[EmotionType.Happy] = 0f;
        emotionScores[EmotionType.Sad] = 0f;
        emotionScores[EmotionType.Angry] = 0f;
        emotionScores[EmotionType.Surprised] = 0f;
        emotionScores[EmotionType.Fear] = 0f;
        emotionScores[EmotionType.Disgusted] = 0f;
        emotionScores[EmotionType.Confused] = 0f;
        
        // 根据blendshape值计算各种情绪的得分
        for (int i = 0; i < blendShapeWeights.Length && i < blendShapeNames.Count; i++)
        {
            string blendShapeName = blendShapeNames[i];
            float weight = blendShapeWeights[i];
            
            // 如果当前blendshape在权重字典中
            if (emotionWeights.ContainsKey(blendShapeName))
            {
                float weightValue = emotionWeights[blendShapeName];
                
                // 根据blendshape名称和值更新对应情绪得分
                UpdateEmotionScores(blendShapeName, weight, weightValue, emotionScores);
            }
        }
        
        // 找出得分最高的情绪
        EmotionType detectedEmotion = EmotionType.Neutral;
        float maxScore = 0f;
        
        foreach (var pair in emotionScores)
        {
            if (pair.Value > maxScore)
            {
                maxScore = pair.Value;
                detectedEmotion = pair.Key;
            }
        }
        
        // 如果所有得分都很低，认为是中性表情
        if (maxScore < 0.3f)
            detectedEmotion = EmotionType.Neutral;
            
        return detectedEmotion;
    }
    
    // 更新情绪得分
    private void UpdateEmotionScores(string blendShapeName, float value, float weight, Dictionary<EmotionType, float> emotionScores)
    {
        // 根据具体的blendshape名称更新对应情绪得分
        switch (blendShapeName)
        {
            // 嘴部相关
            case "mouthSmileLeft":
            case "mouthSmileRight":
                emotionScores[EmotionType.Happy] += value * weight;
                break;
            case "mouthFrownLeft":
            case "mouthFrownRight":
                emotionScores[EmotionType.Sad] += value * weight;
                break;
            case "mouthPressLeft":
            case "mouthPressRight":
                emotionScores[EmotionType.Angry] += value * weight;
                break;
            case "mouthStretchLeft":
            case "mouthStretchRight":
                emotionScores[EmotionType.Surprised] += value * weight * 0.5f;
                emotionScores[EmotionType.Fear] += value * weight * 0.5f;
                break;
            case "mouthShrugUpper":
                emotionScores[EmotionType.Disgusted] += value * weight;
                break;
            case "mouthLeft":
            case "mouthRight":
                emotionScores[EmotionType.Confused] += value * weight;
                break;
            case "jawOpen":
                emotionScores[EmotionType.Surprised] += value * weight * 0.7f;
                emotionScores[EmotionType.Fear] += value * weight * 0.3f;
                break;
                
            // 眼部相关
            case "eyeSquintLeft":
            case "eyeSquintRight":
                emotionScores[EmotionType.Happy] += value * weight * 0.5f;
                emotionScores[EmotionType.Angry] += value * weight * 0.5f;
                break;
            case "eyeLookDownLeft":
            case "eyeLookDownRight":
                emotionScores[EmotionType.Sad] += value * weight;
                break;
            case "eyeWideLeft":
            case "eyeWideRight":
                emotionScores[EmotionType.Surprised] += value * weight * 0.5f;
                emotionScores[EmotionType.Fear] += value * weight * 0.5f;
                break;
            case "eyeLookInLeft":
            case "eyeLookInRight":
                emotionScores[EmotionType.Confused] += value * weight;
                break;
                
            // 眉毛相关
            case "browInnerUp":
                emotionScores[EmotionType.Sad] += value * weight * 0.3f;
                emotionScores[EmotionType.Fear] += value * weight * 0.3f;
                emotionScores[EmotionType.Confused] += value * weight * 0.4f;
                break;
            case "browOuterUpLeft":
            case "browOuterUpRight":
                emotionScores[EmotionType.Surprised] += value * weight * 0.5f;
                emotionScores[EmotionType.Confused] += value * weight * 0.5f;
                break;
            case "browDownLeft":
            case "browDownRight":
                emotionScores[EmotionType.Angry] += value * weight;
                break;
                
            // 鼻部相关
            case "noseSneerLeft":
            case "noseSneerRight":
                emotionScores[EmotionType.Disgusted] += value * weight;
                break;
        }
    }
    
    // 计算情绪置信度
    private float CalculateEmotionConfidence(EmotionType emotion, float[] blendShapeWeights, List<string> blendShapeNames)
    {
        if (emotion == EmotionType.Neutral)
            return 0.5f; // 中性表情的默认置信度
            
        float totalWeight = 0f;
        float relevantWeight = 0f;
        
        // 计算相关特征的总权重和实际权重
        for (int i = 0; i < blendShapeWeights.Length && i < blendShapeNames.Count; i++)
        {
            string blendShapeName = blendShapeNames[i];
            float weight = blendShapeWeights[i];
            
            // 如果是相关特征
            if (IsRelevantFeature(blendShapeName, emotion))
            {
                relevantWeight += weight;
                totalWeight += 1f;
            }
        }
        
        if (totalWeight == 0f)
            return 0f;
            
        return Mathf.Clamp01(relevantWeight / totalWeight);
    }
    
    // 判断是否是相关特征
    private bool IsRelevantFeature(string blendShapeName, EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Happy:
                return blendShapeName.Contains("Smile") || blendShapeName.Contains("eyeSquint");
            case EmotionType.Sad:
                return blendShapeName.Contains("Frown") || blendShapeName.Contains("LookDown");
            case EmotionType.Angry:
                return blendShapeName.Contains("browDown") || blendShapeName.Contains("Press");
            case EmotionType.Surprised:
                return blendShapeName.Contains("jawOpen") || blendShapeName.Contains("eyeWide");
            case EmotionType.Fear:
                return blendShapeName.Contains("eyeWide") || blendShapeName.Contains("browInnerUp");
            case EmotionType.Disgusted:
                return blendShapeName.Contains("noseSneer") || blendShapeName.Contains("mouthShrugUpper");
            case EmotionType.Confused:
                return blendShapeName.Contains("browInnerUp") || blendShapeName.Contains("eyeLookIn");
            default:
                return false;
        }
    }
    
    // 更新当前情绪
    private void UpdateCurrentEmotion(EmotionType detectedEmotion, float confidence)
    {
        // 如果情绪变化了
        if (detectedEmotion != currentEmotion)
        {
            emotionStartTime = Time.time;
            previousEmotion = currentEmotion;
            currentEmotion = detectedEmotion;
            currentEmotionConfidence = confidence;
        }
        // 如果情绪未变化但置信度提高
        else if (confidence > currentEmotionConfidence)
        {
            currentEmotionConfidence = confidence;
        }
        
        // 只有当情绪持续时间足够且置信度足够高时才触发事件
        if (Time.time - emotionStartTime >= expressionHoldTime && currentEmotionConfidence >= confidenceThreshold)
        {
            // 如果情绪确实发生了变化（相对于上一次触发的情绪）
            if (previousEmotion != currentEmotion)
            {
                OnEmotionChanged?.Invoke(currentEmotion, currentEmotionConfidence);
                previousEmotion = currentEmotion; // 更新上一次触发的情绪
                
                if (showDebugInfo)
                {
                    Debug.Log($"表情变化检测: {currentEmotion}, 置信度: {currentEmotionConfidence:F2}");
                }
            }
        }
    }
    
    // 显示调试信息
    private void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label($"当前情绪: {currentEmotion}");
        GUILayout.Label($"置信度: {currentEmotionConfidence:F2}");
        GUILayout.Label($"持续时间: {Time.time - emotionStartTime:F1}秒");
        
        if (GUILayout.Button("重置情绪"))
        {
            currentEmotion = EmotionType.Neutral;
            previousEmotion = EmotionType.Neutral;
            emotionStartTime = Time.time;
            currentEmotionConfidence = 0f;
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}