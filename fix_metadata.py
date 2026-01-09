import json
import os

def fix_metadata_format(input_path='onnx_models/model_metadata.json', 
                     output_path='onnx_models/model_metadata_fixed.json'):
    """
    修复metadata格式，使其可被Unity的JsonUtility解析
    """
    # 读取原始metadata
    with open(input_path, 'r') as f:
        metadata = json.load(f)
    
    # 创建Unity兼容的格式
    unity_metadata = {
        'num_features': metadata['num_features'],
        'num_classes': metadata['num_classes'],
        'scaler_mean_json': str(metadata['scaler_mean']).replace("'", '"'),
        'scaler_std_json': str(metadata['scaler_std']).replace("'", '"'),
        'class_names_json': str(metadata['class_names']).replace("'", '"')
    }
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(unity_metadata, f, indent=2)
    
    print(f"已保存Unity兼容的metadata到: {output_path}")
    
    # 同时创建简化的版本用于手动配置
    simple_path = 'onnx_models/metadata_simple.txt'
    with open(simple_path, 'w') as f:
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
    
    print(f"已创建简化版本: {simple_path}")
    print("\n在Unity中，可以:")
    print("1. 使用 model_metadata_fixed.json (推荐)")
    print("2. 或者参考 metadata_simple.txt 手动配置")

if __name__ == "__main__":
    fix_metadata_format()
