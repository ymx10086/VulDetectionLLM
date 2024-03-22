# VulDetectionLLM

## 环境准备

```shell
conda env create -n vul python=3.8
conda activate vul (新版本采用source activate vul)
pip install -r requirements.txt
```

## 模型下载

1. 前往config.py自行更改模型路径
2. 自行下载模型

## VulDetectionLLM数据集

请将数据集放在项目第一层目录下，更名为benchset


## 运行实验脚本

- 当前支持模型如下：

  ```
  "vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4", "claude-instant-1","claude-2", "palm-2", "gemini-pro", "deepseek-coder", "qwen", "codellama", "chatglm3-6b"
  ```

- 运行实验脚本完成推理

  ```
  python generate.py
  ```

  可使用的参数：

  - --model ：设置模型名称，默认为gemini-pro
  - --scale ：设置数据集规模，默认为2k（可选["2k", "4k", "8k", "16k", "32k"]）
  - --max-n-tokens ：限制模型最长生成长度，默认为500
  - --temperature：设置模型温度，默认为0

- 以gemini-pro为例评测2k代码数据

  ```
  python generate.py --model gemini-pro --scale 2k
  ```

## 额外添加支持模型或api

- 额外添加模型（huggingface支持的）时
  - 在config.py文件中声明模型路径名称和具体路径，并在conversers.py被导入
  - get_model_path_and_template中查阅[FastChat/fastchat/model/model_adapter.py at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py)判断是否可以添加template并对应添加
- 额外添加api时
  - 在langeage_models.py参考其他api的实现封装生成函数api类
  - get_model_path_and_template中查阅[FastChat/fastchat/model/model_adapter.py at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py)判断是否可以添加template并对应添加
  - 在load_indiv_model函数中实现调用
