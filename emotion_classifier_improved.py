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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path, label):
    df = pd.read_csv(file_path)
    features = df.drop('timestamp', axis=1)
    features['label'] = label
    return features

class ImprovedEmotionClassifierNN(nn.Module):
    """
    改进的神经网络模型
    - 增加容量（64->128->64->32）
    - 添加残差连接
    - 更深的网络
    """
    def __init__(self, input_shape, num_classes):
        super(ImprovedEmotionClassifierNN, self).__init__()

        self.layer1 = nn.Linear(input_shape, 128)
        self.dropout1 = nn.Dropout(0.4)

        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)

        self.layer3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)

        self.layer4 = nn.Linear(32, num_classes)

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
        x = self.dropout3(x)

        # 输出层
        x = self.layer4(x)
        x = torch.softmax(x, dim=1)

        return x

def create_improved_neural_network(input_shape, num_classes):
    model = ImprovedEmotionClassifierNN(input_shape, num_classes)
    return model

def train_neural_network_improved(X_train, y_train, X_test, y_test, label_encoder,
                                 epochs=200, batch_size=32):
    """
    改进的训练函数
    - 增加epoch数量
    - 添加学习率调度
    - 使用更深的网络
    """
    print("\n" + "="*60)
    print("训练改进的PyTorch神经网络")
    print("="*60)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(label_encoder.transform(y_train))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(label_encoder.transform(y_test))

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建改进的模型
    input_shape = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = create_improved_neural_network(input_shape, num_classes)

    print(f"\n模型架构:")
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # 训练循环
    print("\n开始训练...")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 50

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

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

        # 学习率调度
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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
    print("\n" + "="*60)
    print("导出PyTorch模型到ONNX")
    print("="*60)
    print(f"输出路径: {output_path}")
    print(f"Opset版本: {opset_version}")

    try:
        dummy_input = torch.randn(1, input_shape)
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

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ 文件大小: {file_size / 1024:.2f} KB")

        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过!")

        return onnx_model

    except Exception as e:
        print(f"✗ ONNX导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)

    emotion_files = {
        'anger': 'EmotionData/anger.csv',
        'disgust': 'EmotionData/disgust.csv',
        'fear': 'EmotionData/fear.csv',
        'happy': 'EmotionData/happy.csv',
        'neutral': 'EmotionData/neutral.csv',
        'sad': 'EmotionData/sad.csv',
        'surprise': 'EmotionData/surprise.csv'
    }

    print("="*60)
    print("改进的多类别表情识别")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    data_frames = []

    for emotion, file_path in emotion_files.items():
        if os.path.exists(file_path):
            df = load_data(file_path, emotion)
            data_frames.append(df)
            print(f"{emotion}: {df.shape[0]} 样本")

    all_data = pd.concat(data_frames, axis=0).reset_index(drop=True)

    # 去除Viseme特征
    X = all_data.drop('label', axis=1)
    viseme_columns = [col for col in X.columns if col.startswith('viseme_')]
    X = X.drop(viseme_columns, axis=1)
    print(f"去除Viseme特征后剩余特征数: {X.shape[1]}")

    y = all_data['label']

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 标准化
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

    # 训练改进的神经网络
    print("\n" + "="*60)
    print("训练改进的神经网络")
    print("="*60)
    nn_model, history, nn_accuracy, y_pred_nn, cm_nn = train_neural_network_improved(
        X_train_scaled, y_train, X_test_scaled, y_test, label_encoder,
        epochs=200, batch_size=32
    )

    # 导出ONNX
    onnx_path = os.path.join(output_dir, 'emotion_classifier_improved.onnx')
    num_features = X_train_scaled.shape[1]
    num_classes = len(label_encoder.classes_)
    onnx_model = export_pytorch_to_onnx(nn_model, onnx_path, num_features, num_classes, opset_version=12)

    # 保存元数据
    metadata = {
        'num_features': num_features,
        'num_classes': num_classes,
        'class_names': label_encoder.classes_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
    }

    metadata_path = os.path.join(output_dir, 'model_metadata_improved.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 创建Unity兼容格式
    unity_metadata = {
        'num_features': metadata['num_features'],
        'num_classes': metadata['num_classes'],
        'scaler_mean_json': str(metadata['scaler_mean']).replace("'", '"'),
        'scaler_std_json': str(metadata['scaler_std']).replace("'", '"'),
        'class_names_json': str(metadata['class_names']).replace("'", '"')
    }

    fixed_metadata_path = os.path.join(output_dir, 'model_metadata_improved_fixed.json')
    with open(fixed_metadata_path, 'w') as f:
        json.dump(unity_metadata, f, indent=2)

    print("\n" + "="*60)
    print("改进训练完成!")
    print("="*60)
    print(f"ONNX模型: {onnx_path}")
    print(f"元数据: {fixed_metadata_path}")
    print(f"测试集准确率: {nn_accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
