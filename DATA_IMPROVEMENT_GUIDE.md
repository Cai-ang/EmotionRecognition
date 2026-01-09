# 数据采集改进指南

## 当前问题
- Neutral被误判为anger/disgust
- Fear和Surprise混淆

## 改进建议

### 1. Neutral数据（最重要！）

**问题**：中性表情特征不明显，容易被误判

**解决方案**：
- **增加数据量**：至少500-1000帧（当前可能只有300帧）
- **采集技巧**：
  - 保持完全放松的面部
  - 眼睛自然睁开，不要瞪大
  - 嘴巴完全闭合，不要有微笑或撇嘴
  - 不要有皱眉、挑眉等微表情
  - 保持头部直立，不要有任何转向
- **采集环境**：
  - 在不同时间段采集（早上、下午、晚上）
  - 确保采集者状态放松
  - 每次采集后休息一下

### 2. Fear和Surprise数据

**问题**：两个表情相似度太高（都是"惊讶"类）

**解决方案**：
- **Fear特征**：
  - 眉毛上扬但眼皮下压
  - 嘴巴张开但嘴角向下
  - 眼睛睁大但有恐惧感
- **Surprise特征**：
  - 眉毛上扬明显
  - 嘴巴张大成O型
  - 眼睛睁大但没有恐惧感
- **区分点**：
  - Fear强调恐惧感的眉毛和眼型
  - Surprise强调嘴巴的O型和自然的惊讶感

### 3. 数据平衡性检查

**建议比例**：
- Neutral: 20-30% （提高比例）
- Happy: 10-15%
- Anger: 10-15%
- Sad: 10-15%
- Surprise: 10-15%
- Fear: 10-15%
- Disgust: 10-15%

**检查方法**：
```python
import pandas as pd
df = pd.read_csv('EmotionData/neutral.csv')
print(f"Neutral样本数: {len(df)}")  # 建议至少500

# 检查所有类别
for emotion in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
    df = pd.read_csv(f'EmotionData/{emotion}.csv')
    print(f"{emotion}: {len(df)}")
```

### 4. 数据质量提升

**避免采集**：
- 过渡表情（从一种表情变到另一种）
- 混合表情（比如"惊讶中带点微笑"）
- 不自然的表情（夸张或僵硬）
- 眼神变化（眨眼、眼动）
- 头部运动

**数据清洗**：
```python
import pandas as pd
import numpy as np

def clean_data(file_path, threshold=0.5):
    df = pd.read_csv(file_path)
    # 移除时间戳列
    features = df.drop('timestamp', axis=1)

    # 检测异常值（某个特征突然变化）
    diff = features.diff().abs()
    outliers = (diff > threshold).any(axis=1)

    # 移除异常帧
    clean_df = df[~outliers].copy()

    print(f"原始数据: {len(df)} 帧")
    print(f"清洗后数据: {len(clean_df)} 帧")
    print(f"移除异常: {len(df) - len(clean_df)} 帧")

    return clean_df

# 对每个情绪清洗
for emotion in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
    clean_data(f'EmotionData/{emotion}.csv')
```

### 5. 数据增强（可选）

如果数据不足，可以用数据增强：

```python
import pandas as pd
import numpy as np

def augment_data(df, factor=2):
    """
    通过添加小噪声来增强数据
    """
    augmented = df.copy()

    for _ in range(factor - 1):
        # 复制数据
        new_df = df.copy()

        # 添加小噪声（模拟细微差异）
        features = new_df.drop('timestamp', axis=1)
        noise = np.random.normal(0, 0.01, features.shape)
        features_noisy = features + noise

        # 确保值在[0, 1]范围内
        features_noisy = np.clip(features_noisy, 0, 1)

        # 更新数据
        new_df[features.columns] = features_noisy

        # 更新时间戳（避免重复）
        new_df['timestamp'] = new_df['timestamp'] + 1000 * (_ + 1)

        # 合并
        augmented = pd.concat([augmented, new_df], axis=0)

    return augmented.reset_index(drop=True)

# 使用示例
df = pd.read_csv('EmotionData/neutral.csv')
augmented_df = augment_data(df, factor=2)
augmented_df.to_csv('EmotionData/neutral_augmented.csv', index=False)
```

### 6. 多人采集

**重要性**：不同人的面部特征差异很大

**建议**：
- 至少2-3个人采集数据
- 年龄、性别多样化
- 采集习惯不同的人（有人表情夸张，有人内敛）

### 7. 采集时间分布

**避免疲劳影响**：
- 每种情绪分多次采集（每次2-3秒）
- 每次采集之间休息1-2分钟
- 不同时间段采集（上午、下午、晚上）

## 重新训练建议

1. 按上述方法重新采集neutral数据（最重要！）
2. 检查所有类别的数据量，确保平衡
3. 重新运行训练：
   ```bash
   python emotion_classifier.py
   ```
4. 对比准确率，应该有提升

## 评估改进效果

训练后检查：
1. Neutral在测试集上的准确率
2. Confusion Matrix中Neutral与Anger/Disgust的混淆
3. Fear与Surprise的混淆情况
