using UnityEngine;
using Unity.Barracuda;
using System.IO;

public class EmotionRecognizer : MonoBehaviour
{
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
    {
        // 加载模型
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Compute, runtimeModel);

            // 加载元数据
            LoadMetadata();

            Debug.Log("模型加载成功!");
        }
        else
        {
            Debug.LogError("模型资产未设置!");
        }
    }

    void LoadMetadata()
    {
        // 从Resources加载JSON元数据文件
        TextAsset metadataText = Resources.Load<TextAsset>("model_metadata_fixed");
        if (metadataText != null)
        {
            ModelMetadata metadata = JsonUtility.FromJson<ModelMetadata>(metadataText.text);

            // 解析JSON数组字符串为C#数组
            // scaler_mean_json 格式: "[0.1, 0.2, ...]"
            // scaler_std_json 格式: "[0.3, 0.4, ...]"
            // class_names_json 格式: "[\"anger\", \"happy\"]"

            // 提取并转换scaler_mean
            string meanStr = metadata.scaler_mean_json.Trim('[', ']');
            string[] meanParts = meanStr.Split(',');
            scalerMean = new float[52];
            for (int i = 0; i < 52; i++)
            {
                scalerMean[i] = float.Parse(meanParts[i].Trim());
            }

            // 提取并转换scaler_std
            string stdStr = metadata.scaler_std_json.Trim('[', ']');
            string[] stdParts = stdStr.Split(',');
            scalerStd = new float[52];
            for (int i = 0; i < 52; i++)
            {
                scalerStd[i] = float.Parse(stdParts[i].Trim());
            }

            // 提取并转换class_names
            string classStr = metadata.class_names_json.Trim('[', ']');
            string[] classParts = classStr.Split(',');
            classNames = new string[7];
            for (int i = 0; i < 7; i++)
            {
                classNames[i] = classParts[i].Trim('"', ' ');
            }
        }
        else
        {
            Debug.LogError("无法加载元数据文件! 请确保model_metadata_fixed.json在Resources文件夹中");
        }
    }

    public string PredictEmotion(float[] blendShapeWeights)
    {
        if (worker == null || blendShapeWeights.Length != 52)
        {
            Debug.LogError("模型未初始化或特征维度不匹配");
            return "Unknown";
        }

        // 1. 标准化输入
        float[] normalizedInput = new float[52];
        for (int i = 0; i < 52; i++)
        {
            normalizedInput[i] = (blendShapeWeights[i] - scalerMean[i]) / scalerStd[i];
        }

        // 2. 创建输入张量
        Tensor inputTensor = new Tensor(1, 52, normalizedInput);

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
        {
            if (output[i] > maxProb)
            {
                maxProb = output[i];
                predictedClass = i;
            }
        }

        // 7. 返回结果
        string result = classNames[predictedClass];
        if (resultText != null)
        {
            resultText.text = $"表情: {result}\n概率: {maxProb:F2}";
        }

        return result;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }

    // 元数据结构
    [System.Serializable]
    private class ModelMetadata
    {
        public int num_features;
        public int num_classes;
        public string scaler_mean_json;
        public string scaler_std_json;
        public string class_names_json;
    }
}
