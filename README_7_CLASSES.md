# 7类表情识别训练与部署指南

## 概述
支持7种情绪的实时表情识别：
- anger（愤怒）
- disgust（厌恶）
- fear（恐惧）
- happy（快乐）
- neutral（中性）
- sad（悲伤）
- surprise（惊讶）

## 训练步骤

### 1. 准备数据
确保以下CSV文件存在于`EmotionData/`文件夹：
- anger.csv
- disgust.csv
- fear.csv
- happy.csv
- neutral.csv
- sad.csv
- surprise.csv

### 2. 运行训练脚本
```bash
python emotion_classifier.py
```

### 3. 生成的文件
训练成功后会生成以下文件：
- `onnx_models/emotion_classifier.onnx` - ONNX模型（约25KB）
- `onnx_models/model_metadata_fixed.json` - Unity兼容的元数据
- `onnx_models/metadata_manual.txt` - 手动配置参考
- `emotion_classification_results.png` - 结果可视化
- `feature_importance.png` - 特征重要性分析

## Unity部署

### 1. 复制文件到Unity
将以下文件复制到Unity项目：
- `emotion_classifier.onnx` → `Assets/Resources/`
- `model_metadata_fixed.json` → `Assets/Resources/`
- `Assets/Scripts/EmotionRecognizer.cs` → `Assets/Scripts/`（如果已存在则覆盖）

### 2. 配置场景
1. 在场景中创建一个GameObject
2. 添加`EmotionRecognizer`组件
3. 配置组件：
   - **Model Asset**: 拖入`emotion_classifier.onnx`
   - **Metadata Asset**: 拖入`model_metadata_fixed.json`
   - **Result Text** (可选): 拖入TextMeshPro组件用于显示结果
   - **Confidence Text** (可选): 拖入另一个TextMeshPro组件显示置信度

### 3. 使用FTTest.cs
1. 找到场景中的`FTTest`对象
2. 在Inspector中，将`Emotion Recognizer`字段关联到包含`EmotionRecognizer`组件的GameObject
3. 运行场景

### 4. 调用预测
在代码中调用：
```csharp
string emotion = emotionRecognizer.PredictEmotion(blendShapeWeights);
```

或者获取所有概率：
```csharp
var prediction = emotionRecognizer.PredictEmotionWithProbabilities(blendShapeWeights);
Debug.Log($"预测表情: {prediction.predictedEmotion}");
for (int i = 0; i < prediction.classNames.Length; i++)
{
    Debug.Log($"{prediction.classNames[i]}: {prediction.probabilities[i] * 100:F1}%");
}
```

## 预期性能
- **准确率**: 95%+ (取决于数据质量)
- **推理速度**: < 1ms (移动设备)
- **模型大小**: ~25KB

## 常见问题

### 1. 编译错误
确保删除旧的`EmotionRecognizer.cs`文件，只保留`Assets/Scripts/`中的版本。

### 2. 模型未初始化
检查：
- `emotion_classifier.onnx`是否在`Assets/Resources/`文件夹
- `model_metadata_fixed.json`是否在`Assets/Resources/`文件夹
- 两个资源是否正确拖入Inspector

### 3. 预测结果不准确
- 确保使用了PICO的面部追踪模式：`PXR_FTM_FACE_LIPS_BS`
- 检查数据采集时的表情是否与实际使用时一致
- 考虑增加训练数据量

## 数据采集建议

每种情绪建议采集：
- 至少300帧数据（10秒@30Hz）
- 保持表情自然但明显
- 避免头部转动
- 多人采集以提高泛化能力

## 测试
使用adb查看日志：
```bash
adb logcat -s Unity
```

预期看到：
```
EmotionRecognizer: Model initialized successfully!
FTTest: Predicted emotion: happy
FTTest: Predicted emotion: sad
```
