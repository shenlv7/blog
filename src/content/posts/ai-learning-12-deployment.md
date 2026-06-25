---
title: "AI学习之路(第12期)：AI应用部署与工程化——从Notebook到产品"
slug: ai-learning-12-deployment
pubDate: 2026-06-27
description: "从模型导出到服务部署，从性能优化到监控运维，把训练好的AI模型变成真正可用的产品"
image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200"
tags: ["AI学习", "模型部署", "ONNX", "TensorRT", "Docker", "MLOps", "工程化"]
series: "AI学习之路"
episode: 12
---

# AI学习之路(第12期)：AI应用部署与工程化——从Notebook到产品

![AI部署](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800)

> "一个模型如果没有部署到生产环境，那它只是一个昂贵的实验。"

## 前言

恭喜你走到了AI学习之路的最后一期！前11期，我们从机器学习基础一路走到AI Agent和伦理安全。你可能已经能训练出不错的模型了。但有一个残酷的现实：

**Kaggle上99%的模型从未进入生产环境。**

一个Jupyter Notebook里跑得飞起的模型，要变成用户可以使用的产品，中间隔着一整条工程化的鸿沟：模型格式转换、推理加速、服务封装、负载均衡、监控告警、A/B测试、持续迭代……

这一期，我们补上这最后一块拼图。

---

## 1. 模型导出与格式转换

### 1.1 为什么需要模型转换？

```
训练框架 (PyTorch/TensorFlow)  →  推理引擎 (ONNX/TensorRT/CoreML)

原因：
- 训练框架太重，部署环境装不下
- 推理引擎针对部署优化（量化、算子融合）
- 跨平台需求（服务器/手机/浏览器/嵌入式设备）
```

### 1.2 PyTorch → ONNX

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 加载训练好的模型
model = models.resnet18(pretrained=True)
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("✅ ONNX模型导出成功")
```

### 1.3 ONNX推理

```python
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession("resnet18.onnx")

# 准备输入
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 推理
outputs = session.run(None, {"input": input_data})
predictions = outputs[0]

print(f"预测结果shape: {predictions.shape}")
print(f"Top-1类别: {np.argmax(predictions)}")
```

### 1.4 格式对比

| 格式 | 来源 | 目标平台 | 特点 |
|------|------|---------|------|
| **ONNX** | 通用 | 服务器/边缘 | 跨框架标准格式 |
| **TorchScript** | PyTorch | 服务器 | PyTorch原生序列化 |
| **TensorRT** | NVIDIA | GPU服务器 | 极致推理性能 |
| **CoreML** | Apple | iOS/macOS | Apple设备优化 |
| **TFLite** | TensorFlow | 移动端/IoT | 轻量级部署 |
| **GGML** | 社区 | CPU/边缘 | LLM专用格式 |

---

## 2. 推理加速

### 2.1 模型量化

量化是将浮点数(FP32)转为低精度(INT8/FP16)的过程：

```python
import torch

class ModelQuantizer:
    """模型量化工具"""
    
    @staticmethod
    def dynamic_quantization(model):
        """动态量化：推理时动态计算量化参数"""
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized
    
    @staticmethod
    def static_quantization(model, calibration_loader):
        """静态量化：使用校准数据预先计算量化参数"""
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        prepared = torch.quantization.prepare(model)
        
        # 校准：用真实数据跑一遍
        with torch.no_grad():
            for data, _ in calibration_loader:
                prepared(data)
        
        # 转换
        quantized = torch.quantization.convert(prepared)
        return quantized
    
    @staticmethod
    def compare_size(original, quantized):
        """对比模型大小"""
        import os
        
        torch.save(original.state_dict(), "/tmp/original.pth")
        torch.save(quantized.state_dict(), "/tmp/quantized.pth")
        
        orig_size = os.path.getsize("/tmp/original.pth") / 1024 / 1024
        quant_size = os.path.getsize("/tmp/quantized.pth") / 1024 / 1024
        
        print(f"原始模型: {orig_size:.2f} MB")
        print(f"量化模型: {quant_size:.2f} MB")
        print(f"压缩比: {orig_size/quant_size:.1f}x")

# 使用示例
model = models.resnet18(pretrained=True)
quantizer = ModelQuantizer()

# 动态量化
quantized = quantizer.dynamic_quantization(model)
quantizer.compare_size(model, quantized)
```

### 2.2 模型剪枝

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """结构化剪枝：移除30%不重要的参数"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    # 永久化剪枝
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
    
    return model

# 统计稀疏率
def sparsity_stats(model):
    total = 0
    zeros = 0
    for param in model.parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()
    print(f"稀疏率: {zeros/total:.2%} ({zeros}/{total})")
```

### 2.3 批量推理优化

```python
import torch
import time

class BatchInference:
    """批量推理优化"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
    
    def benchmark(self, input_shape, batch_sizes=[1, 8, 16, 32, 64]):
        """测试不同batch size的吞吐量"""
        results = {}
        
        for bs in batch_sizes:
            dummy = torch.randn(bs, *input_shape).to(self.device)
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    self.model(dummy)
            
            # 计时
            torch.cuda.synchronize()
            start = time.time()
            
            iterations = 100
            for _ in range(iterations):
                with torch.no_grad():
                    self.model(dummy)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            throughput = (bs * iterations) / elapsed
            latency = elapsed / iterations * 1000  # ms
            
            results[bs] = {
                "throughput": throughput,
                "latency_ms": latency
            }
            
            print(f"Batch {bs:3d}: {throughput:.0f} samples/s, {latency:.1f}ms/batch")
        
        return results
```

---

## 3. 服务部署

### 3.1 FastAPI部署

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

app = FastAPI(title="AI模型服务")

# 全局加载模型
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PredictRequest(BaseModel):
    image_base64: str

class PredictResponse(BaseModel):
    class_id: int
    confidence: float
    latency_ms: float

@app.on_event("startup")
async def load_model():
    global model
    import torchvision.models as models
    model = models.resnet18(pretrained=True).eval()
    print("✅ 模型加载完成")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    import time
    start = time.time()
    
    try:
        # 解码图片
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 预处理
        input_tensor = transform(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
        
        latency = (time.time() - start) * 1000
        
        return PredictResponse(
            class_id=predicted.item(),
            confidence=confidence.item(),
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

### 3.2 Docker容器化

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY app.py .
COPY models/ ./models/

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/resnet18.onnx
      - WORKERS=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### 3.3 Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-model
  template:
    metadata:
      labels:
        app: ai-model
    spec:
      containers:
      - name: ai-service
        image: your-registry/ai-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-model-service
spec:
  selector:
    app: ai-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 4. 监控与运维

### 4.1 推理指标监控

```python
import time
from dataclasses import dataclass, field
from typing import List
from collections import deque

@dataclass
class ModelMetrics:
    """模型推理指标"""
    latencies: deque = field(default_factory=lambda: deque(maxlen=10000))
    predictions: deque = field(default_factory=lambda: deque(maxlen=10000))
    errors: int = 0
    total_requests: int = 0
    
    def record(self, latency_ms: float, prediction: int, error: bool = False):
        self.total_requests += 1
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        if error:
            self.errors += 1
    
    def get_stats(self):
        if not self.latencies:
            return None
        
        lats = sorted(self.latencies)
        return {
            "total_requests": self.total_requests,
            "error_rate": self.errors / self.total_requests,
            "latency_p50": lats[len(lats) // 2],
            "latency_p95": lats[int(len(lats) * 0.95)],
            "latency_p99": lats[int(len(lats) * 0.99)],
            "throughput_rps": len(self.latencies) / (max(self.latencies) - min(self.latencies) + 0.001),
            "prediction_distribution": {
                k: list(self.predictions).count(k) 
                for k in set(self.predictions)
            }
        }
    
    def check_drift(self, baseline_distribution, threshold=0.1):
        """检测预测分布漂移"""
        current = self.get_stats()["prediction_distribution"]
        for cls, baseline_rate in baseline_distribution.items():
            current_rate = current.get(cls, 0) / self.total_requests
            if abs(current_rate - baseline_rate) > threshold:
                return True, f"类别{cls}分布漂移: {baseline_rate:.2%} → {current_rate:.2%}"
        return False, "分布稳定"

# 使用示例
metrics = ModelMetrics()
# ... 在推理时记录指标 ...
metrics.record(latency_ms=15.3, prediction=1)
metrics.record(latency_ms=12.1, prediction=0)

stats = metrics.get_stats()
print(f"P50延迟: {stats['latency_p50']:.1f}ms")
print(f"P99延迟: {stats['latency_p99']:.1f}ms")
```

### 4.2 A/B测试框架

```python
import random
from typing import Dict, Any

class ABTestRouter:
    """A/B测试路由器"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.traffic_split: Dict[str, float] = {}
        self.results: Dict[str, list] = {}
    
    def add_model(self, name: str, model, traffic_pct: float):
        """添加模型和流量分配"""
        self.models[name] = model
        self.traffic_split[name] = traffic_pct
        self.results[name] = []
    
    def route(self, input_data):
        """根据流量分配路由请求"""
        rand = random.random()
        cumulative = 0
        
        for model_name, pct in self.traffic_split.items():
            cumulative += pct
            if rand < cumulative:
                return model_name, self.models[model_name]
        
        # 默认返回第一个
        first = list(self.models.keys())[0]
        return first, self.models[first]
    
    def predict(self, input_data):
        """执行预测并记录结果"""
        model_name, model = self.route(input_data)
        
        start = time.time()
        prediction = model(input_data)
        latency = (time.time() - start) * 1000
        
        self.results[model_name].append({
            "prediction": prediction,
            "latency": latency,
            "timestamp": time.time()
        })
        
        return model_name, prediction
    
    def analyze(self):
        """分析A/B测试结果"""
        print("=" * 50)
        print("📊 A/B测试结果")
        print("=" * 50)
        
        for model_name, results in self.results.items():
            if not results:
                continue
            
            latencies = [r["latency"] for r in results]
            print(f"\n模型: {model_name}")
            print(f"  请求数: {len(results)}")
            print(f"  平均延迟: {sum(latencies)/len(latencies):.1f}ms")
            print(f"  P95延迟: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")

# 使用示例
ab_test = ABTestRouter()
ab_test.add_model("resnet18", resnet18_model, traffic_pct=0.5)
ab_test.add_model("efficientnet", efficientnet_model, traffic_pct=0.5)

# 运行测试
for _ in range(1000):
    model_name, prediction = ab_test.predict(test_image)

ab_test.analyze()
```

---

## 5. MLOps工作流

### 5.1 完整的ML管道

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path

class MLPipeline:
    """机器学习管道"""
    
    def __init__(self, experiment_name: str):
        self.experiment = experiment_name
        self.runs_dir = Path(f"experiments/{experiment_name}")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def start_run(self, params: dict) -> str:
        """开始一次实验运行"""
        run_id = hashlib.md5(
            f"{datetime.now().isoformat()}{json.dumps(params)}".encode()
        ).hexdigest()[:8]
        
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # 保存参数
        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        
        self.current_run = run_dir
        return run_id
    
    def log_metric(self, name: str, value: float, step: int = None):
        """记录指标"""
        metrics_file = self.current_run / "metrics.jsonl"
        entry = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_model(self, model, name: str = "model"):
        """保存模型"""
        import torch
        model_path = self.current_run / f"{name}.pth"
        torch.save(model.state_dict(), model_path)
        return str(model_path)
    
    def log_artifact(self, file_path: str, name: str = None):
        """保存产物"""
        import shutil
        src = Path(file_path)
        dst = self.current_run / (name or src.name)
        shutil.copy2(src, dst)
        return str(dst)
    
    def compare_runs(self):
        """对比不同实验运行"""
        runs = []
        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            params_file = run_dir / "params.json"
            metrics_file = run_dir / "metrics.jsonl"
            
            if params_file.exists():
                with open(params_file) as f:
                    params = json.load(f)
                
                metrics = {}
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        for line in f:
                            m = json.loads(line)
                            metrics[m["name"]] = m["value"]
                
                runs.append({
                    "run_id": run_dir.name,
                    "params": params,
                    "metrics": metrics
                })
        
        # 打印对比表
        if runs:
            print("\n实验对比:")
            print("-" * 80)
            for run in runs:
                print(f"Run {run['run_id']}: {run['params']} → {run['metrics']}")
        
        return runs

# 使用示例
pipeline = MLPipeline("resnet-experiments")

# 实验1
run_id = pipeline.start_run({"lr": 0.001, "epochs": 10, "batch_size": 32})
# ... 训练过程 ...
pipeline.log_metric("train_loss", 0.5, step=1)
pipeline.log_metric("val_accuracy", 0.85, step=1)
pipeline.log_model(model)

# 实验2
run_id = pipeline.start_run({"lr": 0.0001, "epochs": 20, "batch_size": 64})
# ... 训练过程 ...

# 对比结果
pipeline.compare_runs()
```

---

## 6. 完整部署示例

### 6.1 从训练到部署的完整流程

```python
# deploy_pipeline.py
"""
完整的AI模型部署流程
"""
import torch
import torch.nn as nn
import torchvision.models as models
import onnxruntime as ort
import numpy as np
from pathlib import Path

class DeploymentPipeline:
    """模型部署管道"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.output_dir = Path(f"deployments/{model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def step1_export_onnx(self, model, input_shape=(1, 3, 224, 224)):
        """步骤1: 导出ONNX"""
        print("📦 步骤1: 导出ONNX模型...")
        
        dummy = torch.randn(*input_shape)
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        
        print(f"  ✅ ONNX导出: {onnx_path}")
        return onnx_path
    
    def step2_validate_onnx(self, onnx_path, model, input_shape=(1, 3, 224, 224)):
        """步骤2: 验证ONNX模型"""
        print("🔍 步骤2: 验证ONNX模型...")
        
        # PyTorch推理
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            torch_output = model(dummy).numpy()
        
        # ONNX推理
        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"input": dummy.numpy()})[0]
        
        # 对比
        max_diff = np.abs(torch_output - onnx_output).max()
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  {'✅ 验证通过' if max_diff < 1e-5 else '❌ 验证失败'}")
        return max_diff < 1e-5
    
    def step3_create_serving_config(self):
        """步骤3: 创建服务配置"""
        print("⚙️ 步骤3: 创建服务配置...")
        
        config = {
            "model_name": self.model_name,
            "model_path": f"{self.model_name}.onnx",
            "input_shape": [3, 224, 224],
            "preprocessing": {
                "resize": [224, 224],
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "serving": {
                "max_batch_size": 32,
                "timeout_ms": 5000,
                "workers": 4
            }
        }
        
        import json
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✅ 配置文件: {config_path}")
        return config
    
    def step4_generate_dockerfile(self):
        """步骤4: 生成Dockerfile"""
        print("🐳 步骤4: 生成Dockerfile...")
        
        dockerfile = f"""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY {self.model_name}.onnx .
COPY config.json .
COPY server.py .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        dockerfile_path = self.output_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)
        
        print(f"  ✅ Dockerfile: {dockerfile_path}")
    
    def run(self, model):
        """执行完整部署流程"""
        print(f"🚀 开始部署模型: {self.model_name}")
        print("=" * 50)
        
        onnx_path = self.step1_export_onnx(model)
        self.step2_validate_onnx(onnx_path, model)
        self.step3_create_serving_config()
        self.step4_generate_dockerfile()
        
        print("=" * 50)
        print(f"✅ 部署准备完成！产物在: {self.output_dir}")
        print("\n下一步:")
        print(f"  cd {self.output_dir}")
        print(f"  docker build -t {self.model_name} .")
        print(f"  docker run -p 8000:8000 {self.model_name}")

# 一键部署
model = models.resnet18(pretrained=True).eval()
pipeline = DeploymentPipeline("resnet18")
pipeline.run(model)
```

---

## 总结与展望

### AI学习之路回顾

```
第1期  机器学习核心概念      ← 基础篇
第2期  深度学习入门
第3期  神经网络架构
第4期  数据预处理与特征工程
第5期  NLP自然语言处理      ← 中级篇
第6期  计算机视觉
第7期  强化学习
第8期  GANs生成对抗网络
第9期  大语言模型(LLM)      ← 进阶篇
第10期 AI Agent与工具使用
第11期 AI伦理与安全
第12期 部署与工程化          ← 你现在在这里
```

### 下一步学习建议

| 方向 | 推荐内容 |
|------|---------|
| **深入LLM** | 微调、RLHF、模型压缩 |
| **Agent开发** | MCP协议、多Agent框架 |
| **CV方向** | 目标检测、图像生成、3D视觉 |
| **NLP方向** | RAG系统、知识图谱 |
| **工程化** | MLOps、特征工程平台 |
| **研究方向** | 阅读论文、复现实验 |

### 推荐资源

- **书籍**: 《Deep Learning》(Goodfellow)、《动手学深度学习》(李沐)
- **课程**: Stanford CS231n、Fast.ai、DeepLearning.AI
- **实践**: Kaggle竞赛、Hugging Face、Papers with Code
- **社区**: GitHub、知乎AI专栏、Twitter/X AI圈

---

**恭喜你完成了AI学习之路全部12期！** 🎉

从机器学习基础到AI Agent，从模型训练到生产部署，你已经建立了完整的AI技术栈。接下来，选一个方向深入，动手做项目，在实践中成长。

记住：**最好的学习方式，是边做边学。**

---

## 参考资料

- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Developer Guide](https://developer.nvidia.com/tensorrt)
- [MLOps Principles](https://ml-ops.org/)
- [Made With ML](https://madewithml.com/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
