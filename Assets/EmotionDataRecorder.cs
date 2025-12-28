using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Globalization;
using System.Linq;

// 表情数据记录器
public class EmotionDataRecorder : MonoBehaviour
{
    [Header("记录设置")]
    public bool recordOnStart = false;
    public bool recordEmotions = true;
    public bool recordBlendShapes = true;
    public float recordingInterval = 0.1f; // 记录间隔（秒）
    
    [Header("文件设置")]
    public string fileName = "EmotionData";
    public string fileExtension = ".csv";
    public bool includeTimestamp = true;
    
    [Header("调试")]
    public bool showDebugInfo = true;
    
    // 内部变量
    private bool isRecording = false;
    private float lastRecordingTime = 0f;
    private MicroExpressionRecognizer emotionRecognizer;
    private List<string> blendShapeNames;
    private StreamWriter writer;
    private string filePath;
    
    // 记录的数据
    public struct EmotionRecord
    {
        public float timestamp;
        public EmotionType emotion;
        public float confidence;
        public float[] blendShapeWeights;
    }
    
    private List<EmotionRecord> recordedData = new List<EmotionRecord>();
    
    void Start()
    {
        // 获取表情识别器
        emotionRecognizer = GetComponent<MicroExpressionRecognizer>();
        if (emotionRecognizer == null)
        {
            Debug.LogError("EmotionDataRecorder: 未找到MicroExpressionRecognizer组件!");
            return;
        }
        
        // 获取blendshape名称列表（从FTTest获取）
        FTTest ftTest = GetComponent<FTTest>();
        if (ftTest != null)
        {
            // 这里需要从FTTest获取blendShapeNames
            // 由于blendShapeNames是私有变量，我们需要在FTTest中添加一个公共方法来获取它
            // 或者我们可以在这里定义一个固定的列表
            blendShapeNames = GetStandardBlendShapeNames();
        }
        
        // 如果设置了在开始时记录，则开始记录
        if (recordOnStart)
        {
            StartRecording();
        }
    }
    
    void Update()
    {
        // 如果正在记录
        if (isRecording)
        {
            // 检查是否到了记录时间
            if (Time.time - lastRecordingTime >= recordingInterval)
            {
                RecordCurrentData();
                lastRecordingTime = Time.time;
            }
        }
    }
    
    // 开始记录
    public void StartRecording()
    {
        if (isRecording)
        {
            Debug.LogWarning("EmotionDataRecorder: 已经在记录中!");
            return;
        }
        
        // 创建文件路径
        string timestamp = includeTimestamp ? DateTime.Now.ToString("yyyyMMdd_HHmmss") : "";
        filePath = Path.Combine(Application.persistentDataPath, $"{fileName}{timestamp}{fileExtension}");
        
        try
        {
            // 创建文件和写入器
            writer = new StreamWriter(filePath);
            
            // 写入CSV头部
            WriteCSVHeader();
            
            isRecording = true;
            lastRecordingTime = Time.time;
            recordedData.Clear();
            
            if (showDebugInfo)
                Debug.Log($"EmotionDataRecorder: 开始记录数据到 {filePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"EmotionDataRecorder: 无法创建记录文件: {e.Message}");
        }
    }
    
    // 停止记录
    public void StopRecording()
    {
        if (!isRecording)
        {
            Debug.LogWarning("EmotionDataRecorder: 没有在记录中!");
            return;
        }
        
        isRecording = false;
        
        // 关闭写入器
        if (writer != null)
        {
            writer.Close();
            writer = null;
        }
        
        if (showDebugInfo)
        {
            Debug.Log($"EmotionDataRecorder: 停止记录，共记录 {recordedData.Count} 条数据");
            Debug.Log($"EmotionDataRecorder: 数据已保存到 {filePath}");
        }
    }
    
    // 记录当前数据
    private void RecordCurrentData()
    {
        // 获取当前情绪
        EmotionType currentEmotion = emotionRecognizer.GetCurrentEmotion();
        float confidence = emotionRecognizer.GetCurrentEmotionConfidence();
        
        // 获取当前blendshape权重（需要从FTTest获取）
        float[] currentBlendShapes = GetCurrentBlendShapes();
        
        // 创建记录
        EmotionRecord record = new EmotionRecord
        {
            timestamp = Time.time,
            emotion = currentEmotion,
            confidence = confidence,
            blendShapeWeights = currentBlendShapes
        };
        
        // 添加到记录列表
        recordedData.Add(record);
        
        // 写入CSV文件
        WriteCSVRecord(record);
    }
    
    // 写入CSV头部
    private void WriteCSVHeader()
    {
        List<string> headers = new List<string>();
        
        // 添加时间戳
        headers.Add("Timestamp");
        
        // 如果记录情绪
        if (recordEmotions)
        {
            headers.Add("Emotion");
            headers.Add("Confidence");
        }
        
        // 如果记录blendshape
        if (recordBlendShapes && blendShapeNames != null)
        {
            foreach (string name in blendShapeNames)
            {
                headers.Add($"BS_{name}");
            }
        }
        
        // 写入头部
        writer.WriteLine(string.Join(",", headers));
    }
    
    // 写入CSV记录
    private void WriteCSVRecord(EmotionRecord record)
    {
        List<string> values = new List<string>();
        
        // 添加时间戳
        values.Add(record.timestamp.ToString(CultureInfo.InvariantCulture));
        
        // 如果记录情绪
        if (recordEmotions)
        {
            values.Add(record.emotion.ToString());
            values.Add(record.confidence.ToString(CultureInfo.InvariantCulture));
        }
        
        // 如果记录blendshape
        if (recordBlendShapes && record.blendShapeWeights != null && blendShapeNames != null)
        {
            for (int i = 0; i < record.blendShapeWeights.Length && i < blendShapeNames.Count; i++)
            {
                values.Add(record.blendShapeWeights[i].ToString(CultureInfo.InvariantCulture));
            }
        }
        
        // 写入记录
        writer.WriteLine(string.Join(",", values));
    }
    
    // 获取当前blendshape权重
    private float[] GetCurrentBlendShapes()
    {
        // 从FTTest获取当前blendshape权重
        FTTest ftTest = GetComponent<FTTest>();
        if (ftTest != null)
        {
            // 使用FTTest中新增的公共方法获取blendshape权重
            return ftTest.GetCurrentBlendShapeWeights();
        }
        
        return new float[72]; // 如果无法获取，返回空数组
    }
    
    // 获取标准blendshape名称列表
    private List<string> GetStandardBlendShapeNames()
    {
        return new List<string>
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
    }
    
    // 获取记录数据
    public List<EmotionRecord> GetRecordedData()
    {
        return recordedData;
    }
    
    // 清除记录数据
    public void ClearRecordedData()
    {
        recordedData.Clear();
    }
    
    // 分析情绪数据
    public Dictionary<EmotionType, float> AnalyzeEmotionData()
    {
        Dictionary<EmotionType, float> emotionFrequency = new Dictionary<EmotionType, float>();
        
        // 初始化所有情绪的频率
        emotionFrequency[EmotionType.Neutral] = 0f;
        emotionFrequency[EmotionType.Happy] = 0f;
        emotionFrequency[EmotionType.Sad] = 0f;
        emotionFrequency[EmotionType.Angry] = 0f;
        emotionFrequency[EmotionType.Surprised] = 0f;
        emotionFrequency[EmotionType.Fear] = 0f;
        emotionFrequency[EmotionType.Disgusted] = 0f;
        emotionFrequency[EmotionType.Confused] = 0f;
        
        // 计算每种情绪的频率
        foreach (EmotionRecord record in recordedData)
        {
            emotionFrequency[record.emotion] += 1f;
        }
        
        // 计算百分比
        int totalRecords = recordedData.Count;
        if (totalRecords > 0)
        {
            foreach (EmotionType emotion in emotionFrequency.Keys.ToList())
            {
                emotionFrequency[emotion] = (emotionFrequency[emotion] / totalRecords) * 100f;
            }
        }
        
        return emotionFrequency;
    }
    
    // 显示调试信息
    private void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(320, 10, 300, 200));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label($"记录状态: {(isRecording ? "记录中" : "已停止")}");
        GUILayout.Label($"记录条数: {recordedData.Count}");
        GUILayout.Label($"记录间隔: {recordingInterval}秒");
        
        if (GUILayout.Button(isRecording ? "停止记录" : "开始记录"))
        {
            if (isRecording)
                StopRecording();
            else
                StartRecording();
        }
        
        if (GUILayout.Button("清除数据"))
        {
            ClearRecordedData();
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
    
    // 在应用程序退出时停止记录
    void OnDestroy()
    {
        if (isRecording)
        {
            StopRecording();
        }
    }
}