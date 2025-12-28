using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;

// 情绪反馈系统
public class EmotionFeedbackSystem : MonoBehaviour
{
    [Header("UI元素")]
    public TextMeshProUGUI emotionText;
    public Slider confidenceSlider;
    public Image emotionIcon;
    public Image backgroundPanel;
    
    [Header("情绪图标")]
    public Sprite neutralIcon;
    public Sprite happyIcon;
    public Sprite sadIcon;
    public Sprite angryIcon;
    public Sprite surprisedIcon;
    public Sprite fearIcon;
    public Sprite disgustedIcon;
    public Sprite confusedIcon;
    
    [Header("颜色设置")]
    public Color neutralColor = Color.gray;
    public Color happyColor = Color.yellow;
    public Color sadColor = Color.blue;
    public Color angryColor = Color.red;
    public Color surprisedColor = Color.magenta;
    public Color fearColor = Color.black;
    public Color disgustedColor = Color.green;
    public Color confusedColor = new Color(0.7f, 0.5f, 0f);
    
    [Header("音频反馈")]
    public AudioSource audioSource;
    public AudioClip happySound;
    public AudioClip sadSound;
    public AudioClip angrySound;
    public AudioClip surprisedSound;
    public AudioClip fearSound;
    public AudioClip disgustedSound;
    public AudioClip confusedSound;
    
    [Header("视觉反馈")]
    public ParticleSystem happyParticles;
    public ParticleSystem sadParticles;
    public ParticleSystem angryParticles;
    public ParticleSystem surprisedParticles;
    public ParticleSystem fearParticles;
    public ParticleSystem disgustedParticles;
    public ParticleSystem confusedParticles;
    
    [Header("反馈选项")]
    public bool useAudioFeedback = true;
    public bool useVisualFeedback = true;
    public bool useColorFeedback = true;
    public bool useHapticFeedback = false; // 震动反馈（如果设备支持）
    
    // 内部变量
    private MicroExpressionRecognizer emotionRecognizer;
    private EmotionType lastEmotion = EmotionType.Neutral;
    private float lastFeedbackTime = 0f;
    public float feedbackCooldown = 1f; // 反馈冷却时间（秒）
    
    void Start()
    {
        // 获取情绪识别器
        emotionRecognizer = GetComponent<MicroExpressionRecognizer>();
        if (emotionRecognizer == null)
        {
            Debug.LogError("EmotionFeedbackSystem: 未找到MicroExpressionRecognizer组件!");
            return;
        }
        
        // 订阅情绪变化事件
        emotionRecognizer.OnEmotionChanged += HandleEmotionChanged;
        
        // 初始化UI
        UpdateUI(EmotionType.Neutral, 0f);
    }
    
    // 处理情绪变化
    private void HandleEmotionChanged(EmotionType newEmotion, float confidence)
    {
        // 检查冷却时间
        if (Time.time - lastFeedbackTime < feedbackCooldown)
            return;
            
        lastFeedbackTime = Time.time;
        
        // 更新UI
        UpdateUI(newEmotion, confidence);
        
        // 提供反馈
        if (useAudioFeedback)
            PlayAudioFeedback(newEmotion);
            
        if (useVisualFeedback)
            PlayVisualFeedback(newEmotion);
            
        if (useHapticFeedback)
            PlayHapticFeedback(newEmotion);
            
        lastEmotion = newEmotion;
    }
    
    // 更新UI
    private void UpdateUI(EmotionType emotion, float confidence)
    {
        // 更新文本
        if (emotionText != null)
        {
            emotionText.text = GetEmotionText(emotion);
        }
        
        // 更新滑块
        if (confidenceSlider != null)
        {
            confidenceSlider.value = confidence;
        }
        
        // 更新图标和颜色
        if (emotionIcon != null)
        {
            emotionIcon.sprite = GetEmotionIcon(emotion);
        }
        
        if (useColorFeedback && backgroundPanel != null)
        {
            backgroundPanel.color = GetEmotionColor(emotion);
        }
    }
    
    // 播放音频反馈
    private void PlayAudioFeedback(EmotionType emotion)
    {
        if (audioSource == null) return;
        
        AudioClip clip = GetEmotionAudioClip(emotion);
        if (clip != null)
        {
            audioSource.PlayOneShot(clip);
        }
    }
    
    // 播放视觉反馈
    private void PlayVisualFeedback(EmotionType emotion)
    {
        ParticleSystem particle = GetEmotionParticleSystem(emotion);
        if (particle != null)
        {
            particle.Stop();
            particle.Play();
        }
    }
    
    // 播放震动反馈（如果设备支持）
    private void PlayHapticFeedback(EmotionType emotion)
    {
        // 这里可以添加设备震动反馈的代码
        // PICO设备可能有自己的震动API
        // 示例代码（需要根据实际API调整）：
        // PXR_Input.SetControllerVibration(0, 1, 0.2f, 0.5f);
    }
    
    // 获取情绪文本
    private string GetEmotionText(EmotionType emotion)
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
    
    // 获取情绪图标
    private Sprite GetEmotionIcon(EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Neutral:
                return neutralIcon;
            case EmotionType.Happy:
                return happyIcon;
            case EmotionType.Sad:
                return sadIcon;
            case EmotionType.Angry:
                return angryIcon;
            case EmotionType.Surprised:
                return surprisedIcon;
            case EmotionType.Fear:
                return fearIcon;
            case EmotionType.Disgusted:
                return disgustedIcon;
            case EmotionType.Confused:
                return confusedIcon;
            default:
                return neutralIcon;
        }
    }
    
    // 获取情绪颜色
    private Color GetEmotionColor(EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Neutral:
                return neutralColor;
            case EmotionType.Happy:
                return happyColor;
            case EmotionType.Sad:
                return sadColor;
            case EmotionType.Angry:
                return angryColor;
            case EmotionType.Surprised:
                return surprisedColor;
            case EmotionType.Fear:
                return fearColor;
            case EmotionType.Disgusted:
                return disgustedColor;
            case EmotionType.Confused:
                return confusedColor;
            default:
                return neutralColor;
        }
    }
    
    // 获取情绪音频
    private AudioClip GetEmotionAudioClip(EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Happy:
                return happySound;
            case EmotionType.Sad:
                return sadSound;
            case EmotionType.Angry:
                return angrySound;
            case EmotionType.Surprised:
                return surprisedSound;
            case EmotionType.Fear:
                return fearSound;
            case EmotionType.Disgusted:
                return disgustedSound;
            case EmotionType.Confused:
                return confusedSound;
            default:
                return null;
        }
    }
    
    // 获取情绪粒子系统
    private ParticleSystem GetEmotionParticleSystem(EmotionType emotion)
    {
        switch (emotion)
        {
            case EmotionType.Happy:
                return happyParticles;
            case EmotionType.Sad:
                return sadParticles;
            case EmotionType.Angry:
                return angryParticles;
            case EmotionType.Surprised:
                return surprisedParticles;
            case EmotionType.Fear:
                return fearParticles;
            case EmotionType.Disgusted:
                return disgustedParticles;
            case EmotionType.Confused:
                return confusedParticles;
            default:
                return null;
        }
    }
    
    // 切换反馈类型
    public void ToggleAudioFeedback()
    {
        useAudioFeedback = !useAudioFeedback;
    }
    
    public void ToggleVisualFeedback()
    {
        useVisualFeedback = !useVisualFeedback;
    }
    
    public void ToggleColorFeedback()
    {
        useColorFeedback = !useColorFeedback;
    }
    
    public void ToggleHapticFeedback()
    {
        useHapticFeedback = !useHapticFeedback;
    }
    
    // 测试函数
    [ContextMenu("测试快乐情绪")]
    public void TestHappyEmotion()
    {
        HandleEmotionChanged(EmotionType.Happy, 0.8f);
    }
    
    [ContextMenu("测试悲伤情绪")]
    public void TestSadEmotion()
    {
        HandleEmotionChanged(EmotionType.Sad, 0.8f);
    }
    
    [ContextMenu("测试愤怒情绪")]
    public void TestAngryEmotion()
    {
        HandleEmotionChanged(EmotionType.Angry, 0.8f);
    }
    
    [ContextMenu("测试惊讶情绪")]
    public void TestSurprisedEmotion()
    {
        HandleEmotionChanged(EmotionType.Surprised, 0.8f);
    }
}