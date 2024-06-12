# 轻松玩转书生·浦语大模型趣味 Demo
视频来源：https://www.bilibili.com/video/BV1AH4y1H78d/

代码来源：https://github.com/InternLM/Tutorial/blob/camp2/helloworld/hello_world.md
## 任务目标
- 熟悉算力平台的操作和ssh登录
- 了解如何下载和运行大模型
## 2 **部署 `InternLM2-Chat-1.8B` 模型进行智能对话**
### **2.1 配置基础环境**
```bash
studio-conda -o internlm-base -t demo
# 与 studio-conda 等效的配置方案
# conda create -n demo python==3.10 -y
# conda activate demo
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
```bash
conda activate demo
```
安装必须的python包
```bash
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```
### **2.2 下载 `InternLM2-Chat-1.8B` 模型**
```bash
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```
添入download_mini.py文件的内容
```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```
执行命令，下载模型参数文件：
```bash
python /root/demo/download_mini.py
```
### **2.3 运行 cli_demo**  
编辑cli_demo.py文件的内容
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```
输入命令，执行 Demo 程序：
```bash
conda activate demo
python /root/demo/cli_demo.py
```
输入`exit`以退出
## 3 **实战：部署实战营优秀作品 `八戒-Chat-1.8B` 模型**
### 3.1 **简单介绍 `八戒-Chat-1.8B`、`Chat-嬛嬛-1.8B`、`Mini-Horo-巧耳`（实战营优秀作品）**
### 3.2 **配置基础环境**
```bash
conda activate demo
```
```bash
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
# git clone https://github.com/InternLM/Tutorial -b camp2
cd /root/Tutorial
```
### 3.3 **下载运行 Chat-八戒 Demo**
```bash
python /root/Tutorial/helloworld/bajie_download.py
```
```bash
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```
```bash
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```
##  4 **实战：使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型（需要调整至30% A100）**
### 4.1 **初步介绍 Lagent 相关知识**
### 4.2 **配置基础环境（需要调整至30% A100）**
```bash
conda activate demo
```
```bash
cd /root/demo
```
```bash
git clone https://gitee.com/internlm/lagent.git
# git clone https://github.com/internlm/lagent.git
cd /root/demo/lagent
git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
pip install -e . # 源码安装
```
### 4.3 **使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型为内核的智能体**
```bash
cd /root/demo/lagent
```
建立软链接。主要作用是创建一个指向另一个文件或目录的快捷方式（就是windows下的快捷方式，不用每次都输一长串地址）
```bash
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```
71行，修改需要运行的模型为`InternLM2-Chat-7B`
```bash
# 其他代码...
value='/root/models/internlm2-chat-7b'
# 其他代码...
```
保存。运行模型。
```bash

streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006
```

> **语句解析**
> 1. **启动Streamlit应用**：
>    - `streamlit run` 用于启动一个Streamlit应用程序。Streamlit是一种用于构建和共享数据应用的开源Python库，特别适用于机器学习和数据科学项目的快速原型开发。
>
> 2. **指定应用程序文件**：
>    - `/root/demo/lagent/examples/internlm2_agent_web_demo_hf.py` 是要运行的Python脚本文件路径。这个脚本定义了应用程序的界面和逻辑。
>
> 3. **服务器地址**：
>    - `--server.address 127.0.0.1` 指定服务器的地址为 `127.0.0.1`，即本地回环地址。这意味着应用程序将在本地计算机上运行，只有本地计算机可以访问。
>
> 4. **端口号**：
>    - `--server.port 6006` 指定服务器使用的端口号为 `6006`。端口是计算机上用于网络通信的虚拟通道，通过指定端口号，可以避免与其他应用程序的端口冲突。

```bash
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```
> 注意这里使用ssh登录，未设置公钥的情况下会要求提供密码。windows下使用cmd更方便，使用右键就可以粘贴，`ctl+V`会报错  

## 5 **实战：实践部署 `浦语·灵笔2` 模型（需要调整至30% A100）**
### 5.1 **初步介绍 `XComposer2` 相关知识**
### 5.2 **配置基础环境（需要调整至30% A100）**
```bash
conda activate demo
# 补充环境包
pip install timm==0.4.12 sentencepiece==0.1.99 markdown2==2.4.10 xlsxwriter==3.1.2 gradio==4.13.0 modelscope==1.9.5
```
```bash
cd /root/demo
git clone https://gitee.com/internlm/InternLM-XComposer.git
# git clone https://github.com/internlm/InternLM-XComposer.git
cd /root/demo/InternLM-XComposer
git checkout f31220eddca2cf6246ee2ddf8e375a40457ff626
```
```bash
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b /root/models/internlm-xcomposer2-7b
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b /root/models/internlm-xcomposer2-vl-7b
```
### 5.3 **图文写作实战（需要调整至30% A100）**
```bash
cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
--code_path /root/models/internlm-xcomposer2-7b \
--private \
--num_gpus 1 \
--port 6006
```
```bash
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```
### 5.4 **图片理解实战（需要调整至30% A100）**
```bash
conda activate demo

cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  \
--code_path /root/models/internlm-xcomposer2-vl-7b \
--private \
--num_gpus 1 \
--port 6006
```
## 6 **附录**

### 6.1 **模型下载**
#### 6.2.1 **Hugging Face**
```bash
pip install -U huggingface_hub
```
```python
import os
# 下载模型
os.system('huggingface-cli download --resume-download internlm/internlm2-chat-7b --local-dir your_path')
```
```python
import os 
from huggingface_hub import hf_hub_download  # Load model directly 

hf_hub_download(repo_id="internlm/internlm2-7b", filename="config.json")
```
#### 6.2.2 **ModelScope**
```bash
pip install modelscope==1.9.5
pip install transformers==4.35.2
```
```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='your path', revision='master')
```
#### 6.2.3 **OpenXLab**
```python
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
base_path = './local_files'
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/Usr_name/repo_name.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
```
