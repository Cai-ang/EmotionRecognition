#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试ONNX环境是否正确配置
"""

print("="*60)
print("测试ONNX环境")
print("="*60)

# 测试1: 检查包是否安装
print("\n1. 检查必要的包...")
packages_to_check = ['tensorflow', 'tf2onnx', 'onnx']
for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', '未知')
        print(f"   ✓ {pkg}: {version}")
    except ImportError as e:
        print(f"   ✗ {pkg}: 未安装 ({e})")

# 测试2: 创建一个简单的Keras模型并尝试导出
print("\n2. 测试ONNX导出功能...")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tf2onnx
    import onnx

    # 创建一个简单的模型
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(5,)),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    print("   ✓ Keras模型创建成功")

    # 尝试导出为ONNX
    test_onnx_path = "test_model.onnx"

    try:
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            output_path=test_onnx_path,
            opset=12
        )

        # 验证ONNX模型
        onnx.checker.check_model(onnx_model)

        print(f"   ✓ ONNX导出成功: {test_onnx_path}")

        # 删除测试文件
        import os
        if os.path.exists(test_onnx_path):
            os.remove(test_onnx_path)
            print(f"   ✓ 测试文件已清理")

        print("\n" + "="*60)
        print("ONNX环境测试通过！")
        print("="*60)

    except Exception as e:
        print(f"   ✗ ONNX导出失败: {e}")
        print("\n" + "="*60)
        print("ONNX环境有问题！")
        print("="*60)
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"   ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n建议操作:")
print("1. 如果包未安装，运行: pip install -r requirements.txt")
print("2. 如果ONNX导出失败，可能需要: pip install --upgrade tf2onnx onnx")
