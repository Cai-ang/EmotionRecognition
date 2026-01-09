# 表情识别分类测试说明

## 文件说明

- `emotion_classifier.py` - 表情分类主程序
- `EmotionData/` - 数据文件夹（包含 anger.csv 和 happy.csv）
- `requirements.txt` - Python依赖包

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 使用方法

1. 确保 `EmotionData` 文件夹中包含至少两个表情的CSV文件
2. 运行脚本：

```bash
python emotion_classifier.py
```

## 功能说明

脚本会自动完成以下任务：

### 1. 数据加载
- 读取CSV格式的表情数据
- 自动移除timestamp列
- 添加表情标签

### 2. 数据预处理
- 合并多个表情数据
- 可选：去除Viseme特征（最后15个，通常为0）
- 特征标准化

### 3. 模型训练与评估

训练并对比三个经典模型：

| 模型 | 说明 |
|------|------|
| SVM (RBF) | 支持向量机，RBF核 |
| SVM (Linear) | 支持向量机，线性核 |
| Random Forest | 随机森林，100棵树 |

### 4. 评估指标

- **准确率** (Accuracy)
- **交叉验证** (5-fold CV)
- **分类报告** (Precision, Recall, F1-score)
- **混淆矩阵** (Confusion Matrix)

### 5. 可视化输出

生成两个图表：

1. **emotion_classification_results.png** - 模型性能对比和混淆矩阵
2. **feature_importance.png** - Random Forest特征重要性分析

## 输出示例

```
============================================================
数据集信息:
  - 总样本数: 600
  - 训练集: 420 样本
  - 测试集: 180 样本
  - anger样本: 300
  - happy样本: 300
============================================================

============================================================
训练模型: SVM (RBF)
============================================================

准确率: 0.9778
5折交叉验证准确率: 0.9833 (+/- 0.0236)

分类报告:
              precision    recall  f1-score   support

       anger       0.98      0.98      0.98        90
       happy       0.98      0.98      0.98        90

    accuracy                           0.98       180
   macro avg       0.98      0.98      0.98       180
weighted avg       0.98      0.98      0.98       180

混淆矩阵:
[[88  2]
 [ 2 88]]

============================================================
Top 20 最重要特征 (Random Forest):
============================================================
          feature  importance
  mouthSmileLeft    0.123456
 mouthSmileRight    0.098765
   browInnerUp    0.076543
         ...
```

## 自定义使用

### 修改数据路径

在 `main()` 函数中修改：

```python
anger_path = "EmotionData/anger.csv"
happy_path = "EmotionData/happy.csv"
```

### 添加更多表情类型

修改 `main()` 函数：

```python
# 加载多个表情
sad_df = load_data("EmotionData/sad.csv", 'sad')
fear_df = load_data("EmotionData/fear.csv", 'fear')

# 合并所有数据
all_data = pd.concat([anger_df, happy_df, sad_df, fear_df], axis=0)
```

### 保留Viseme特征

在 `preprocess_data()` 调用时：

```python
X, y = preprocess_data(anger_df, happy_df, remove_visemes=False)
```

### 调整测试集比例

```python
results = train_and_evaluate(X, y, test_size=0.2)  # 20%测试集
```

### 使用其他模型

在 `train_and_evaluate()` 的 `models` 字典中添加：

```python
models = {
    'SVM': SVC(kernel='rbf'),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
}
```

## 注意事项

1. **数据平衡**：确保不同表情的样本数量相近
2. **Viseme特征**：录制时未说话时Viseme全为0，建议去除
3. **特征选择**：可以通过特征重要性分析，去除不重要的特征
4. **模型选择**：小数据量时SVM表现较好，大数据量时可用神经网络

## 下一步

1. 收集更多表情数据（sad, fear, surprise等）
2. 尝试深度学习模型（CNN, LSTM等）
3. 进行特征工程和优化
4. 实时预测（集成到Unity中）
