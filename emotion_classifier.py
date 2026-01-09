import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle

# 深度学习相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ONNX导出相关
try:
    import tf2onnx
    import onnx
    from onnx import version_converter
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False
    print("警告: tf2onnx未安装，TensorFlow模型ONNX导出功能不可用")

try:
    from sklearn import __version__ as sklearn_version
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKLEARN_ONNX_AVAILABLE = True
except ImportError:
    SKLEARN_ONNX_AVAILABLE = False
    print("警告: skl2onnx未安装，sklearn模型ONNX导出功能不可用")

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path, label):
    """
    加载CSV数据并添加标签
    """
    df = pd.read_csv(file_path)
    # 移除timestamp列（只保留blendShape特征）
    features = df.drop('timestamp', axis=1)
    
    # 添加标签列
    features['label'] = label
    
    return features

def train_and_evaluate(X, y, test_size=0.3, random_state=42):
    """
    训练并评估多个分类模型
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print(f"数据集信息:")
    print(f"  - 总样本数: {len(y)}")
    print(f"  - 训练集: {len(y_train)} 样本")
    print(f"  - 测试集: {len(y_test)} 样本")

    # 显示各类别样本数
    unique_labels = sorted(list(set(y)))
    for label in unique_labels:
        count = sum(y == label)
        print(f"  - {label}: {count} 样本")

    print("="*60 + "\n")
    
    # 定义多个模型进行对比
    models = {
        'SVM (RBF)': SVC(kernel='rbf', random_state=random_state),
        'SVM (Linear)': SVC(kernel='linear', random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {name}")
        print('='*60)
        
        # 训练
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        print(f"\n准确率: {accuracy:.4f}")
        print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("混淆矩阵:")
        print(cm)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'confusion_matrix': cm
        }
    
    return results, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_results(results, y_test):
    """
    可视化结果
    """
    # 1. 模型对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率对比
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracies, width, label='测试集准确率', alpha=0.8)
    axes[0].bar(x + width/2, cv_means, width, yerr=cv_stds, label='交叉验证准确率', alpha=0.8, capsize=5)
    axes[0].set_xlabel('模型', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].set_title('模型性能对比', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # 在柱状图上添加数值
    for i, (acc, cv_mean) in enumerate(zip(accuracies, cv_means)):
        axes[0].text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, cv_mean + 0.02, f'{cv_mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 混淆矩阵（使用最佳模型）
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    cm = results[best_model_name]['confusion_matrix']

    # 获取所有唯一的类别标签
    unique_classes = sorted(list(set(y_test)))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=unique_classes,
                yticklabels=unique_classes)
    axes[1].set_xlabel('预测标签', fontsize=12)
    axes[1].set_ylabel('真实标签', fontsize=12)
    axes[1].set_title(f'混淆矩阵 - {best_model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('emotion_classification_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图表已保存为: emotion_classification_results.png")
    plt.close()  # 关闭图形以释放内存，避免阻塞

def feature_importance(results, X_train):
    """
    分析特征重要性（仅适用于Random Forest）
    """
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        feature_names = X_train.columns
        
        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*60)
        print("Top 20 最重要特征 (Random Forest):")
        print("="*60)
        print(importance_df.head(20).to_string(index=False))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['importance'], alpha=0.8)
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('重要性', fontsize=12)
        plt.title('Top 20 特征重要性', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n特征重要性图已保存为: feature_importance.png")
        plt.close()  # 关闭图形以释放内存，避免阻塞

class EmotionClassifierNN(nn.Module):
    """
    PyTorch神经网络模型
    适用于ONNX导出
    """
    def __init__(self, input_shape, num_classes):
        super(EmotionClassifierNN, self).__init__()

        self.layer1 = nn.Linear(input_shape, 64)
        self.dropout1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)

        self.layer3 = nn.Linear(32, 16)

        self.output = nn.Linear(16, num_classes)

    def forward(self, x):
        # 第一层
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # 第二层
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # 第三层
        x = self.layer3(x)
        x = torch.relu(x)

        # 输出层
        x = self.output(x)
        x = torch.softmax(x, dim=1)

        return x

def create_neural_network(input_shape, num_classes):
    """
    创建PyTorch神经网络模型
    """
    model = EmotionClassifierNN(input_shape, num_classes)
    return model

def train_neural_network(X_train, y_train, X_test, y_test, label_encoder, epochs=100, batch_size=32):
    """
    训练PyTorch神经网络模型
    """
    print("\n" + "="*60)
    print("训练PyTorch神经网络模型")
    print("="*60)

    # 将标签转换为数值
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test_encoded)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    input_shape = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = create_neural_network(input_shape, num_classes)

    print(f"\n模型架构:")
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练循环
    print("\n开始训练...")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 训练
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            val_loss = criterion(outputs, y_test_tensor).item()
            _, predicted = torch.max(outputs.data, 1)
            val_acc = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = label_encoder.inverse_transform(predicted.numpy())

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print("神经网络评估结果:")
    print("="*60)
    print(f"\n准确率: {accuracy:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)

    history = {
        'loss': train_losses,
        'accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }

    return model, history, accuracy, y_pred, cm

def export_pytorch_to_onnx(model, output_path, input_shape, num_classes, opset_version=12):
    """
    将PyTorch模型导出为ONNX格式
    参数:
        model: PyTorch模型
        output_path: 输出ONNX文件路径
        input_shape: 输入形状
        num_classes: 类别数
        opset_version: ONNX opset版本（Barracuda推荐12）
    """
    print("\n" + "="*60)
    print("导出PyTorch模型到ONNX")
    print("="*60)
    print(f"输出路径: {output_path}")
    print(f"Opset版本: {opset_version}")
    print(f"模型类型: {type(model).__name__}")

    try:
        # 创建示例输入
        dummy_input = torch.randn(1, input_shape)

        # 导出为ONNX格式
        print("开始转换...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"✓ ONNX模型已保存到: {output_path}")
        print(f"✓ ONNX opset版本: {opset_version}")

        # 验证文件是否存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ 文件大小: {file_size / 1024:.2f} KB")
        else:
            print(f"✗ 警告: 文件未创建!")

        # 验证ONNX模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过!")

        return onnx_model

    except Exception as e:
        print(f"✗ ONNX导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def export_sklearn_to_onnx(model, output_path, input_shape, opset_version=12):
    """
    将sklearn模型导出为ONNX格式
    参数:
        model: sklearn模型
        output_path: 输出ONNX文件路径
        input_shape: 输入形状 (num_features,)
        opset_version: ONNX opset版本（Barracuda推荐12）
    """
    if not SKLEARN_ONNX_AVAILABLE:
        print("错误: skl2onnx未安装，无法导出ONNX模型")
        print("请运行: pip install skl2onnx")
        return None

    print("\n" + "="*60)
    print("导出sklearn模型到ONNX")
    print("="*60)
    print(f"输出路径: {output_path}")
    print(f"Opset版本: {opset_version}")
    print(f"模型类型: {type(model).__name__}")

    try:
        # 定义输入类型
        initial_type = [('input', FloatTensorType([None, input_shape]))]

        # 转换为ONNX
        print("开始转换...")
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=opset_version
        )

        # 保存ONNX模型
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✓ ONNX模型已保存到: {output_path}")
        print(f"✓ ONNX opset版本: {opset_version}")

        # 验证文件是否存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ 文件大小: {file_size / 1024:.2f} KB")
        else:
            print(f"✗ 警告: 文件未创建!")

        # 验证ONNX模型
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过!")

        return onnx_model

    except Exception as e:
        print(f"✗ ONNX导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_model_metadata(scaler, label_encoder, feature_names, output_dir):
    """
    保存模型元数据（用于Unity预处理）
    """
    metadata = {
        'feature_names': feature_names.tolist(),
        'num_features': len(feature_names),
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n模型元数据已保存到: {metadata_path}")
    
    return metadata_path

def create_unity_example_script(metadata_path, onnx_path, output_dir, unity_metadata=None):
    """
    创建Unity示例脚本
    """
    # 读取Unity兼容的元数据
    if unity_metadata is None:
        with open(metadata_path, 'r') as f:
            unity_metadata = json.load(f)

    num_features = unity_metadata['num_features']
    num_classes = unity_metadata['num_classes']

    script_content = f'''using UnityEngine;
using Unity.Barracuda;
using System.IO;

public class EmotionRecognizer : MonoBehaviour
{{
    [Header("Model Settings")]
    public NNModel modelAsset;
    public Model runtimeModel;

    [Header("UI References")]
    public UnityEngine.UI.Text resultText;

    private IWorker worker;
    private float[] scalerMean;
    private float[] scalerStd;
    private string[] classNames;

    void Start()
    {{
        // 加载模型
        if (modelAsset != null)
        {{
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Compute, runtimeModel);

            // 加载元数据
            LoadMetadata();

            Debug.Log("模型加载成功!");
        }}
        else
        {{
            Debug.LogError("模型资产未设置!");
        }}
    }}

    void LoadMetadata()
    {{
        // 从Resources加载JSON元数据文件
        TextAsset metadataText = Resources.Load<TextAsset>("model_metadata_fixed");
        if (metadataText != null)
        {{
            ModelMetadata metadata = JsonUtility.FromJson<ModelMetadata>(metadataText.text);

            // 解析JSON数组字符串为C#数组
            // scaler_mean_json 格式: "[0.1, 0.2, ...]"
            // scaler_std_json 格式: "[0.3, 0.4, ...]"
            // class_names_json 格式: "[\\"anger\\", \\"happy\\"]"

            // 提取并转换scaler_mean
            string meanStr = metadata.scaler_mean_json.Trim('[', ']');
            string[] meanParts = meanStr.Split(',');
            scalerMean = new float[{num_features}];
            for (int i = 0; i < {num_features}; i++)
            {{
                scalerMean[i] = float.Parse(meanParts[i].Trim());
            }}

            // 提取并转换scaler_std
            string stdStr = metadata.scaler_std_json.Trim('[', ']');
            string[] stdParts = stdStr.Split(',');
            scalerStd = new float[{num_features}];
            for (int i = 0; i < {num_features}; i++)
            {{
                scalerStd[i] = float.Parse(stdParts[i].Trim());
            }}

            // 提取并转换class_names
            string classStr = metadata.class_names_json.Trim('[', ']');
            string[] classParts = classStr.Split(',');
            classNames = new string[{num_classes}];
            for (int i = 0; i < {num_classes}; i++)
            {{
                classNames[i] = classParts[i].Trim('"', ' ');
            }}
        }}
        else
        {{
            Debug.LogError("无法加载元数据文件! 请确保model_metadata_fixed.json在Resources文件夹中");
        }}
    }}

    public string PredictEmotion(float[] blendShapeWeights)
    {{
        if (worker == null || blendShapeWeights.Length != {num_features})
        {{
            Debug.LogError("模型未初始化或特征维度不匹配");
            return "Unknown";
        }}

        // 1. 标准化输入
        float[] normalizedInput = new float[{num_features}];
        for (int i = 0; i < {num_features}; i++)
        {{
            normalizedInput[i] = (blendShapeWeights[i] - scalerMean[i]) / scalerStd[i];
        }}

        // 2. 创建输入张量
        Tensor inputTensor = new Tensor(1, {num_features}, normalizedInput);

        // 3. 执行推理
        worker.Execute(inputTensor);

        // 4. 获取输出
        Tensor outputTensor = worker.PeekOutput();
        float[] output = outputTensor.data.Download(outputTensor.shape);

        // 5. 清理
        inputTensor.Dispose();

        // 6. 找到最大概率的类别
        int predictedClass = 0;
        float maxProb = output[0];
        for (int i = 1; i < output.Length; i++)
        {{
            if (output[i] > maxProb)
            {{
                maxProb = output[i];
                predictedClass = i;
            }}
        }}

        // 7. 返回结果
        string result = classNames[predictedClass];
        if (resultText != null)
        {{
            resultText.text = $"表情: {{result}}\\n概率: {{maxProb:F2}}";
        }}

        return result;
    }}

    void OnDestroy()
    {{
        worker?.Dispose();
    }}

    // 元数据结构
    [System.Serializable]
    private class ModelMetadata
    {{
        public int num_features;
        public int num_classes;
        public string scaler_mean_json;
        public string scaler_std_json;
        public string class_names_json;
    }}
}}
'''

    script_path = os.path.join(output_dir, 'EmotionRecognizer.cs')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"Unity示例脚本已创建: {{script_path}}")
    return script_path

def main():
    # 创建输出目录
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)

    # 数据目录
    emotion_data_dir = "EmotionData"

    print("="*60)
    print("多类别表情识别 & ONNX导出（支持多人数据）")
    print("="*60)

    # 自动检测人物文件夹
    person_dirs = []
    if os.path.isdir(emotion_data_dir):
        # 查找所有子文件夹（ca/, lcs/等）
        for item in os.listdir(emotion_data_dir):
            item_path = os.path.join(emotion_data_dir, item)
            if os.path.isdir(item_path):
                person_dirs.append(item)
                print(f"发现人物文件夹: {item}")

    if len(person_dirs) == 0:
        print("警告: 未发现人物文件夹，尝试直接在EmotionData/中查找CSV文件")
        # 没有子文件夹，直接查找CSV
        emotion_files = {
            'anger': 'EmotionData/anger.csv',
            'disgust': 'EmotionData/disgust.csv',
            'fear': 'EmotionData/fear.csv',
            'happy': 'EmotionData/happy.csv',
            'neutral': 'EmotionData/neutral.csv',
            'sad': 'EmotionData/sad.csv',
            'surprise': 'EmotionData/surprise.csv'
        }

        data_frames = []
        for emotion, file_path in emotion_files.items():
            if os.path.exists(file_path):
                df = load_data(file_path, emotion)
                data_frames.append(df)
                print(f"  {emotion}: {df.shape[0]} 样本")
    else:
        # 有子文件夹，加载所有人物的数据
        print(f"共发现 {len(person_dirs)} 个人物")
        data_frames = []

        emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        for person_id in person_dirs:
            person_path = os.path.join(emotion_data_dir, person_id)
            print(f"\n加载 {person_id} 的数据...")

            person_data_count = 0
            for emotion in emotions:
                # 尝试不同的文件名格式：emotion_personid.csv 或 emotion.csv
                possible_files = [
                    os.path.join(person_path, f"{emotion}_{person_id}.csv"),
                    os.path.join(person_path, f"{emotion}.csv")
                ]

                file_found = False
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        df = load_data(file_path, emotion)
                        data_frames.append(df)
                        person_data_count += df.shape[0]
                        file_found = True
                        break

                if file_found:
                    print(f"  {emotion}: {df.shape[0]} 样本")
                else:
                    print(f"  ⚠️  {emotion}: 文件未找到")

            print(f"  {person_id} 总计: {person_data_count} 样本")

    # 合并所有数据
    if len(data_frames) > 0:
        all_data = pd.concat(data_frames, axis=0).reset_index(drop=True)
        print(f"\n✓ 总数据量: {all_data.shape[0]} 样本")
        print(f"✓ 情绪类别: 7 种")
    else:
        print("\n错误: 未加载到任何数据!")
        return

    # 数据分布分析
    print("\n" + "="*60)
    print("数据分布分析")
    print("="*60)

    # 按情绪统计
    emotion_counts = all_data['label'].value_counts().sort_values(ascending=False)
    print("\n按情绪统计:")
    total_samples = len(all_data)
    for emotion in emotion_counts.index:
        count = emotion_counts[emotion]
        percentage = count / total_samples * 100
        print(f"  {emotion}: {count} 样本 ({percentage:.1f}%)")

    # 如果有多人数据，显示按人物统计
    if len(person_dirs) > 0:
        print("\n按人物统计:")
        for person_id in person_dirs:
            person_mask = all_data.apply(lambda row: person_id in str(row.name), axis=1) if 'person_id' in locals() else False
            # 简单方法：计算每个人物的大约比例
            person_percentage = 1.0 / len(person_dirs) * 100
            print(f"  {person_id}: 约 {total_samples // len(person_dirs)} 样本 ({person_percentage:.1f}%)")

    # 绘制数据分布图
    print("\n生成数据分布图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 按情绪的样本数
    colors = ['#ff6b6b', '#8b4513', '#2c3e50', '#f1c40f', '#27ae60', '#16a085', '#d4ac0d']
    axes[0].bar(emotion_counts.index, emotion_counts.values, color=colors)
    axes[0].set_xlabel('情绪', fontsize=11)
    axes[0].set_ylabel('样本数', fontsize=11)
    axes[0].set_title('按情绪的数据分布', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 在柱状图上添加数值
    for i, (emotion, count) in enumerate(zip(emotion_counts.index, emotion_counts.values)):
        percentage = count / total_samples * 100
        axes[0].text(i, count, f'\n{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. 数据平衡性饼图
    axes[1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%',
                         colors=colors, startangle=90)
    axes[1].set_title('数据平衡性', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 数据分布图已保存: data_distribution.png")
    plt.close()

    # 提取特征和标签
    X = all_data.drop('label', axis=1)
    y = all_data['label']

    # 去除Viseme特征（最后15个）
    viseme_columns = [col for col in X.columns if col.startswith('viseme_')]
    X = X.drop(viseme_columns, axis=1)
    print(f"去除Viseme特征后剩余特征数: {X.shape[1]}")

    # 创建标签编码器
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"类别编码: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "="*60)
    print(f"数据集信息:")
    print(f"  - 总样本数: {len(y)}")
    print(f"  - 训练集: {len(y_train)} 样本")
    print(f"  - 测试集: {len(y_test)} 样本")

    for emotion in label_encoder.classes_:
        count = sum(y == emotion)
        percentage = count / len(y) * 100
        print(f"  - {emotion}: {count} 样本 ({percentage:.1f}%)")

    print("="*60 + "\n")

    # 训练和评估传统机器学习模型
    results, _, _, _, _, _ = train_and_evaluate(X, y)

    # 训练PyTorch神经网络模型（用于ONNX导出）
    print("\n" + "="*60)
    print("训练PyTorch神经网络（用于ONNX导出）")
    print("="*60)
    nn_model, history, nn_accuracy, y_pred_nn, cm_nn = train_neural_network(
        X_train_scaled, y_train, X_test_scaled, y_test, label_encoder,
        epochs=150, batch_size=32  # 增加epoch以适应更多类别
    )

    # 可视化结果
    plot_results(results, y_test)

    # 特征重要性分析
    feature_importance(results, pd.DataFrame(X_train_scaled, columns=X.columns))

    # 选择最佳的传统模型用于对比
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']

    print("\n" + "="*60)
    print(f"最佳传统模型: {best_model_name} (准确率: {best_accuracy:.4f})")
    print(f"PyTorch神经网络模型准确率: {nn_accuracy:.4f}")
    print("="*60)

    # 导出ONNX模型（使用PyTorch模型，稳定且效果好）
    print("\n" + "="*60)
    print("准备导出ONNX模型...")
    print("="*60)
    print(f"输出目录: {output_dir}")

    onnx_path = os.path.join(output_dir, 'emotion_classifier.onnx')
    print(f"ONNX文件路径: {onnx_path}")

    # 使用PyTorch模型导出
    num_features = X_train_scaled.shape[1]
    num_classes = len(label_encoder.classes_)
    onnx_model = export_pytorch_to_onnx(nn_model, onnx_path, num_features, num_classes, opset_version=12)

    if onnx_model is None:
        print("\n警告: ONNX模型导出失败！")
        print("可能的原因:")
        print("1. skl2onnx未正确安装 (运行: pip install skl2onnx onnx)")
        print("2. 导出过程中出现错误（见上方错误信息）")
        print("\n继续生成其他文件...")
    else:
        # 验证ONNX文件是否真的被创建
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path)
            print(f"✓ ONNX文件已成功创建，大小: {file_size / 1024:.2f} KB")
        else:
            print(f"✗ 警告: ONNX模型对象已创建但文件未找到: {onnx_path}")
    
    # 保存模型元数据
    metadata_path = save_model_metadata(scaler, label_encoder, X.columns, output_dir)
    
    # 修复metadata格式以兼容Unity
    fixed_metadata_path = os.path.join(output_dir, 'model_metadata_fixed.json')
    
    # 创建Unity兼容的格式
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    unity_metadata = {
        'num_features': metadata['num_features'],
        'num_classes': metadata['num_classes'],
        'scaler_mean_json': str(metadata['scaler_mean']).replace("'", '"'),
        'scaler_std_json': str(metadata['scaler_std']).replace("'", '"'),
        'class_names_json': str(metadata['class_names']).replace("'", '"')
    }
    
    with open(fixed_metadata_path, 'w') as f:
        json.dump(unity_metadata, f, indent=2)
    
    print(f"Unity兼容的metadata已保存到: {fixed_metadata_path}")
    
    # 创建简化版本用于手动配置
    simple_path = os.path.join(output_dir, 'metadata_manual.txt')
    with open(simple_path, 'w', encoding='utf-8') as f:
        f.write(f"num_features: {metadata['num_features']}\n")
        f.write(f"num_classes: {metadata['num_classes']}\n")
        f.write(f"class_names: {metadata['class_names']}\n")
        f.write("\n// C# 数组格式:\n")
        f.write(f"int numFeatures = {metadata['num_features']};\n")
        f.write(f"int numClasses = {metadata['num_classes']};\n")
        f.write("\n// 类别名称:\n")
        for i, name in enumerate(metadata['class_names']):
            f.write(f"// class {i}: {name}\n")
        f.write("\n// scaler_mean:\n")
        f.write("float[] scalerMean = new float[] {\n")
        for i, val in enumerate(metadata['scaler_mean']):
            f.write(f"    {val:.6f}f" + (",\n" if i < len(metadata['scaler_mean']) - 1 else "\n"))
        f.write("};\n")
        f.write("\n// scaler_std:\n")
        f.write("float[] scalerStd = new float[] {\n")
        for i, val in enumerate(metadata['scaler_std']):
            f.write(f"    {val:.6f}f" + (",\n" if i < len(metadata['scaler_std']) - 1 else "\n"))
        f.write("};\n")
    
    print(f"手动配置参考文件已创建: {simple_path}")
    
    # 创建Unity示例脚本
    # 使用unity_metadata而不是metadata，因为Unity脚本需要的是_json格式的键
    script_path = create_unity_example_script(fixed_metadata_path, onnx_path, output_dir, unity_metadata)

    print("\n" + "="*60)
    print("导出完成!")
    print("="*60)
    print(f"ONNX模型: {onnx_path}")
    print(f"元数据: {fixed_metadata_path}")
    print(f"手动配置参考: {simple_path}")
    print(f"Unity脚本: {script_path}")
    print(f"导出的模型: {best_model_name} (准确率: {best_accuracy:.4f})")
    print("="*60)

    print("\nUnity集成步骤:")
    print("1. 将以下文件放入Unity项目的Assets/文件夹:")
    print(f"   - {onnx_path} -> Assets/Resources/")
    print(f"   - {fixed_metadata_path} -> Assets/Resources/")
    print(f"   - {script_path} -> Assets/Scripts/")
    print("\n2. 安装Barracuda包:")
    print("   Window > Package Manager > + > Add package from git URL")
    print("   URL: com.unity.barracuda")
    print("\n3. 设置场景:")
    print("   - 创建GameObject，添加EmotionRecognizer脚本")
    print("   - 设置Model Asset和Metadata Asset")
    print("\n4. 运行时调用:")
    print("   recognizer.PredictEmotion(blendShapeWeights)")

if __name__ == "__main__":
    main()
