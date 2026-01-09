# 表情识别模型导出ONNX指南

## 功能特性

本脚本现已支持：
- ✅ 传统机器学习模型（SVM, Random Forest）评估
- ✅ 深度神经网络模型训练
- ✅ **ONNX模型导出**（适用于Unity Barracuda）
- ✅ 模型元数据保存（用于预处理）
- ✅ Unity示例脚本自动生成

## 安装依赖

```bash
pip install -r requirements.txt
```

新增加的依赖包：
- `tensorflow` - 深度学习框架
- `tf2onnx` - TensorFlow到ONNX转换工具
- `onnx` - ONNX模型操作库

## 使用方法

### 1. 运行训练和导出

```bash
python emotion_classifier.py
```

脚本会自动完成：
1. 训练传统机器学习模型（对比）
2. 训练神经网络模型
3. 评估模型性能
4. 导出ONNX模型
5. 保存模型元数据
6. 生成Unity集成脚本

### 2. 输出文件

运行后会在 `onnx_models` 文件夹生成：

| 文件 | 说明 |
|------|------|
| `emotion_classifier.onnx` | ONNX格式模型（Barracuda可加载） |
| `model_metadata.json` | 标准化参数和类别信息 |
| `EmotionRecognizer.cs` | Unity集成示例脚本 |

## Unity集成步骤

### 1. 安装Barracuda

1. 打开Unity项目
2. 进入 `Window > Package Manager`
3. 点击 "+" > "Add package from git URL"
4. 输入：`com.unity.barracuda`

### 2. 导入模型文件

将以下文件复制到Unity项目：
```
Assets/
├── Resources/
│   ├── emotion_classifier.onnx
│   └── model_metadata.json
└── Scripts/
    └── EmotionRecognizer.cs
```

### 3. 设置场景

1. 创建空GameObject，命名为 "EmotionRecognizer"
2. 添加 `EmotionRecognizer` 脚本
3. 在Inspector中设置：
   - `Model Asset`: 拖入ONNX模型文件
   - `Result Text`: 拖入显示结果的Text组件

### 4. 在代码中使用

```csharp
// 获取EmotionRecognizer组件
EmotionRecognizer recognizer = GetComponent<EmotionRecognizer>();

// 准备输入数据（72个blendShape权重）
float[] blendShapeWeights = new float[72];
// ... 从PICO面部追踪获取数据

// 预测表情
string predictedEmotion = recognizer.PredictEmotion(blendShapeWeights);
Debug.Log($"预测表情: {predictedEmotion}");
```

## ONNX导出说明

### 关键参数

- **Opset Version**: 12（Barracuda兼容性）
- **输入维度**: [1, 57]（batch_size, features）
- **输出维度**: [1, 2]（batch_size, classes）

### 模型架构

```
Input (57 features)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(16) + ReLU
    ↓
Dense(2) + Softmax
    ↓
Output (2 classes)
```

### 预处理流程

在Unity中，输入数据需要经过标准化：

```csharp
// 标准化公式
normalized_value = (input - mean) / std

// 从metadata.json中读取
scaler_mean = [mean1, mean2, ..., mean57]
scaler_std = [std1, std2, ..., std57]
```

## 性能优化建议

### 1. 模型简化

如果实时性要求高，可以简化模型：

```python
# 在emotion_classifier.py中修改create_neural_network函数
def create_neural_network(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(32, activation='relu'),  # 减少神经元
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 2. 量化导出

```python
# 量化为INT8以减少模型大小
from tf2onnx import optimizer

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    output_path=output_path,
    opset=12
)

# 应用优化
optimized_model = optimizer.optimize(onnx_model)
```

### 3. Barracuda Worker选择

在Unity脚本中：

```csharp
// CPU（兼容性最好）
WorkerFactory.Type.ComputePreferable

// GPU（性能最好，但需要GPU支持）
WorkerFactory.Type.GPUCompute
```

## 故障排查

### 问题1: ONNX导出失败

**症状**: `tf2onnx` 相关错误

**解决方案**:
```bash
pip install --upgrade tensorflow tf2onnx onnx
```

### 问题2: Barracuda加载失败

**症状**: `Barracuda version not supported`

**解决方案**:
- 确保Unity版本 >= 2021.3
- 更新Barracuda包到最新版本
- 检查opset_version是否为12

### 问题3: 预测结果不正确

**症状**: 预测结果总是同一个类别

**可能原因**:
1. 输入数据未标准化
2. 特征维度不匹配
3. 模型元数据未正确加载

**解决方案**:
- 确保在Unity中正确应用标准化
- 检查输入数组长度是否为57
- 验证metadata.json是否正确加载

### 问题4: 性能太慢

**解决方案**:
1. 减少模型层数和神经元数量
2. 使用GPU Compute worker
3. 降低推理频率（例如每5帧预测一次）

## 扩展功能

### 添加更多表情类别

1. 收集更多表情数据（sad, fear, surprise等）
2. 修改`load_data`函数加载更多CSV

```python
sad_df = load_data("EmotionData/sad.csv", 'sad')
fear_df = load_data("EmotionData/fear.csv", 'fear')

# 合并所有数据
all_data = pd.concat([anger_df, happy_df, sad_df, fear_df], axis=0)
```

3. 重新训练和导出模型

### 实时预测优化

在Unity中使用协程降低更新频率：

```csharp
private IEnumerator PredictCoroutine(float[] blendShapeWeights)
{
    while (true)
    {
        string emotion = PredictEmotion(blendShapeWeights);
        yield return new WaitForSeconds(0.1f); // 每100ms预测一次
    }
}
```

## 注意事项

1. **数据质量**: 确保收集的数据质量高，不同表情有明显差异
2. **平衡数据集**: 各类别的样本数量应相近
3. **避免过拟合**: 使用Dropout和交叉验证
4. **模型验证**: 在真实设备上测试模型性能
5. **版本兼容**: 注意TensorFlow和Barracuda的版本兼容性

## 参考资料

- [Unity Barracuda文档](https://docs.unity3d.com/Packages/com.unity.barracuda@latest)
- [ONNX算子支持](https://github.com/Unity-Technologies/barracuda-release)
- [tf2onnx GitHub](https://github.com/onnx/tensorflow-onnx)
