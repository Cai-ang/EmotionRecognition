using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;
using UnityEngine.UI;
using TMPro;

// 情绪历史分析器
public class EmotionHistoryAnalyzer : MonoBehaviour
{
    [Header("分析设置")]
    public int historySize = 100; // 保存的历史记录数量
    public float analysisInterval = 5f; // 分析间隔（秒）
    
    [Header("可视化")]
    public bool showEmotionGraph = true;
    public bool showEmotionStats = true;
    public RectTransform graphContainer;
    public GameObject emotionBarPrefab;
    
    [Header("调试")]
    public bool showDebugInfo = true;
    
    // 内部变量
    private MicroExpressionRecognizer emotionRecognizer;
    private Queue<EmotionSnapshot> emotionHistory = new Queue<EmotionSnapshot>();
    private float lastAnalysisTime = 0f;
    private Dictionary<EmotionType, float> emotionFrequency = new Dictionary<EmotionType, float>();
    private EmotionType dominantEmotion = EmotionType.Neutral;
    private float emotionalStability = 0f; // 情绪稳定性（0-1）
    private float emotionalVolatility = 0f; // 情绪波动性（0-1）
    
    // 情绪快照结构
    public struct EmotionSnapshot
    {
        public float timestamp;
        public EmotionType emotion;
        public float confidence;
    }
    
    void Start()
    {
        // 获取表情识别器
        emotionRecognizer = GetComponent<MicroExpressionRecognizer>();
        if (emotionRecognizer == null)
        {
            Debug.LogError("EmotionHistoryAnalyzer: 未找到MicroExpressionRecognizer组件!");
            return;
        }
        
        // 订阅情绪变化事件
        emotionRecognizer.OnEmotionChanged += OnEmotionChanged;
        
        // 初始化情绪频率字典
        InitializeEmotionFrequency();
        
        if (showDebugInfo)
            Debug.Log("EmotionHistoryAnalyzer: 已初始化");
    }
    
    void Update()
    {
        // 定期分析情绪历史
        if (Time.time - lastAnalysisTime >= analysisInterval)
        {
            AnalyzeEmotionHistory();
            UpdateVisualization();
            lastAnalysisTime = Time.time;
        }
    }
    
    // 初始化情绪频率字典
    private void InitializeEmotionFrequency()
    {
        emotionFrequency[EmotionType.Neutral] = 0f;
        emotionFrequency[EmotionType.Happy] = 0f;
        emotionFrequency[EmotionType.Sad] = 0f;
        emotionFrequency[EmotionType.Angry] = 0f;
        emotionFrequency[EmotionType.Surprised] = 0f;
        emotionFrequency[EmotionType.Fear] = 0f;
        emotionFrequency[EmotionType.Disgusted] = 0f;
        emotionFrequency[EmotionType.Confused] = 0f;
    }
    
    // 情绪变化事件处理
    private void OnEmotionChanged(EmotionType newEmotion, float confidence)
    {
        // 创建情绪快照
        EmotionSnapshot snapshot = new EmotionSnapshot
        {
            timestamp = Time.time,
            emotion = newEmotion,
            confidence = confidence
        };
        
        // 添加到历史记录
        emotionHistory.Enqueue(snapshot);
        
        // 如果超过历史大小，移除最旧的记录
        if (emotionHistory.Count > historySize)
        {
            emotionHistory.Dequeue();
        }
        
        if (showDebugInfo)
            Debug.Log($"EmotionHistoryAnalyzer: 记录情绪变化: {newEmotion}, 置信度: {confidence:F2}");
    }
    
    // 分析情绪历史
    private void AnalyzeEmotionHistory()
    {
        if (emotionHistory.Count == 0) return;
        
        // 重置情绪频率
        InitializeEmotionFrequency();
        
        // 计算情绪频率
        foreach (EmotionSnapshot snapshot in emotionHistory)
        {
            emotionFrequency[snapshot.emotion]++;
        }
        
        // 转换为百分比
        foreach (EmotionType emotion in emotionFrequency.Keys.ToList())
        {
            emotionFrequency[emotion] = (emotionFrequency[emotion] / emotionHistory.Count) * 100f;
        }
        
        // 确定主导情绪
        dominantEmotion = emotionFrequency.OrderByDescending(kvp => kvp.Value).First().Key;
        
        // 计算情绪稳定性
        CalculateEmotionalStability();
        
        // 计算情绪波动性
        CalculateEmotionalVolatility();
        
        if (showDebugInfo)
        {
            Debug.Log($"EmotionHistoryAnalyzer: 主导情绪: {dominantEmotion}, 稳定性: {emotionalStability:F2}, 波动性: {emotionalVolatility:F2}");
        }
    }
    
    // 计算情绪稳定性
    private void CalculateEmotionalStability()
    {
        if (emotionHistory.Count < 2) return;
        
        // 情绪稳定性定义为相同情绪连续出现的比例
        int stableTransitions = 0;
        int totalTransitions = emotionHistory.Count - 1;
        
        var historyArray = emotionHistory.ToArray();
        for (int i = 1; i < historyArray.Length; i++)
        {
            if (historyArray[i].emotion == historyArray[i-1].emotion)
            {
                stableTransitions++;
            }
        }
        
        emotionalStability = (float)stableTransitions / totalTransitions;
    }
    
    // 计算情绪波动性
    private void CalculateEmotionalVolatility()
    {
        if (emotionHistory.Count < 3) return;
        
        // 情绪波动性定义为情绪变化的频率和幅度
        var historyArray = emotionHistory.ToArray();
        
        // 计算情绪变化次数
        int changeCount = 0;
        for (int i = 1; i < historyArray.Length; i++)
        {
            if (historyArray[i].emotion != historyArray[i-1].emotion)
            {
                changeCount++;
            }
        }
        
        // 波动性 = 情绪变化次数 / (历史记录数 - 1)
        emotionalVolatility = (float)changeCount / (historyArray.Length - 1);
    }
    
    // 更新可视化
    private void UpdateVisualization()
    {
        if (!showEmotionGraph) return;
        
        // 清除现有的可视化元素
        ClearVisualization();
        
        // 创建情绪条形图
        CreateEmotionBarChart();
    }
    
    // 清除可视化元素
    private void ClearVisualization()
    {
        if (graphContainer == null) return;
        
        // 移除所有子对象
        foreach (Transform child in graphContainer)
        {
            Destroy(child.gameObject);
        }
    }
    
    // 创建情绪条形图
    private void CreateEmotionBarChart()
    {
        if (graphContainer == null || emotionBarPrefab == null) return;
        
        float containerWidth = graphContainer.rect.width;
        float barWidth = containerWidth / emotionFrequency.Count;
        float maxHeight = graphContainer.rect.height * 0.8f; // 留出20%的空间
        
        // 找出最大频率值用于缩放
        float maxFrequency = emotionFrequency.Values.Max();
        
        int index = 0;
        foreach (var kvp in emotionFrequency)
        {
            // 创建条形
            GameObject bar = Instantiate(emotionBarPrefab, graphContainer);
            RectTransform barRect = bar.GetComponent<RectTransform>();
            
            // 设置大小和位置
            barRect.sizeDelta = new Vector2(barWidth * 0.8f, 0f); // 初始高度为0，之后会动画增长
            barRect.anchoredPosition = new Vector2(barWidth * index + barWidth / 2, 0f);
            
            // 计算高度
            float height = (kvp.Value / maxFrequency) * maxHeight;
            
            // 设置条形的高度（如果有动画组件，可以使用动画）
            barRect.sizeDelta = new Vector2(barRect.sizeDelta.x, height);
            
            // 设置颜色
            Image barImage = bar.GetComponent<Image>();
            if (barImage != null)
            {
                barImage.color = GetEmotionColor(kvp.Key);
            }
            
            // 添加标签
            TextMeshProUGUI label = bar.GetComponentInChildren<TextMeshProUGUI>();
            if (label != null)
            {
                label.text = $"{GetEmotionShortName(kvp.Key)}\n{kvp.Value:F1}%";
            }
            
            index++;
        }
    }
    
    // 获取情绪颜色
    private Color GetEmotionColor(EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Neutral:
                return Color.gray;
            case EmotionType.Happy:
                return Color.yellow;
            case EmotionType.Sad:
                return Color.blue;
            case EmotionType.Angry:
                return Color.red;
            case EmotionType.Surprised:
                return Color.magenta;
            case EmotionType.Fear:
                return Color.black;
            case EmotionType.Disgusted:
                return Color.green;
            case EmotionType.Confused:
                return new Color(0.7f, 0.5f, 0f);
            default:
                return Color.gray;
        }
    }
    
    // 获取情绪简称
    private string GetEmotionShortName(EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Neutral:
                return "中性";
            case EmotionType.Happy:
                return "快乐";
            case EmotionType.Sad:
                return "悲伤";
            case EmotionType.Angry:
                return "愤怒";
            case EmotionType.Surprised:
                return "惊讶";
            case EmotionType.Fear:
                return "恐惧";
            case EmotionType.Disgusted:
                return "厌恶";
            case EmotionType.Confused:
                return "困惑";
            default:
                return "未知";
        }
    }
    
    // 获取主导情绪
    public EmotionType GetDominantEmotion()
    {
        return dominantEmotion;
    }
    
    // 获取情绪频率
    public Dictionary<EmotionType, float> GetEmotionFrequency()
    {
        return emotionFrequency;
    }
    
    // 获取情绪稳定性
    public float GetEmotionalStability()
    {
        return emotionalStability;
    }
    
    // 获取情绪波动性
    public float GetEmotionalVolatility()
    {
        return emotionalVolatility;
    }
    
    // 获取情绪历史记录
    public Queue<EmotionSnapshot> GetEmotionHistory()
    {
        return emotionHistory;
    }
    
    // 清除历史记录
    public void ClearHistory()
    {
        emotionHistory.Clear();
        InitializeEmotionFrequency();
        dominantEmotion = EmotionType.Neutral;
        emotionalStability = 0f;
        emotionalVolatility = 0f;
        ClearVisualization();
        
        if (showDebugInfo)
            Debug.Log("EmotionHistoryAnalyzer: 已清除历史记录");
    }
    
    // 显示调试信息
    private void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(10, 220, 300, 300));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label($"主导情绪: {GetEmotionShortName(dominantEmotion)}");
        GUILayout.Label($"情绪稳定性: {emotionalStability:P1}");
        GUILayout.Label($"情绪波动性: {emotionalVolatility:P1}");
        
        GUILayout.Space(10);
        GUILayout.Label("情绪频率:");
        
        foreach (var kvp in emotionFrequency.OrderByDescending(kvp => kvp.Value))
        {
            GUILayout.Label($"{GetEmotionShortName(kvp.Key)}: {kvp.Value:F1}%");
        }
        
        GUILayout.Space(10);
        
        if (GUILayout.Button("清除历史记录"))
        {
            ClearHistory();
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}