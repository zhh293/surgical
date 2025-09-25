# SAM Segmentation Backend Service

这是一个基于FastAPI和Segment Anything Model (SAM)的后端服务，用于接收前端发送的图片和标注点，进行图像分割，并返回分割结果。

## 功能特点

- 提供RESTful API接口接收图片和标注点
- 使用SAM模型进行高精度图像分割
- 支持跨域请求，方便前端集成
- 包含健康检查和错误处理

## 环境要求

- Python 3.8+
- 建议使用CUDA加速（需要NVIDIA GPU）

## 安装步骤

1. 克隆或下载项目代码

2. 安装依赖包:
   ```
   pip install -r requirements.txt
   ```

3. 下载SAM模型权重文件:
   - 从官方网站下载SAM模型权重：https://github.com/facebookresearch/segment-anything#model-checkpoints
   - 支持的模型类型：
     - ViT-H: sam_vit_h_4b8939.pth (推荐)
     - ViT-L: sam_vit_l_0b3195.pth
     - ViT-B: sam_vit_b_01ec64.pth
    
     - # 下载sam预训练模型
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    # 如果想要分割的效果好请使用 sam_vit_h_4b8939.pth 权重
    # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

4. 设置环境变量（可选）:
   ```
   # Linux/Mac
   export SAM_CHECKPOINT=/path/to/your/sam_checkpoint.pth
   export SAM_MODEL_TYPE=vit_h  # 对应你下载的模型类型

   # Windows
   set SAM_CHECKPOINT=C:\path\to\your\sam_checkpoint.pth
   set SAM_MODEL_TYPE=vit_h
   ```

## 运行服务
python main.py
服务将在 http://0.0.0.0:8000 启动

## API文档

服务启动后，可以通过以下地址访问自动生成的API文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API接口说明

### 根路径
- 地址: `/`
- 方法: GET
- 描述: 检查服务是否运行

### 健康检查
- 地址: `/health`
- 方法: GET
- 描述: 检查服务和模型是否正常加载

### 分割预测
- 地址: `/predict`
- 方法: POST
- 描述: 接收图片和标注点，返回分割掩码
- 请求体:
  ```json
  {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "points": [
      {"x": 120, "y": 80, "label": 1},
      {"x": 200, "y": 150, "label": 1}
    ],
    "image_format": "jpeg"
  }
  ```
- 响应体:
  ```json
  {
    "mask": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "success": true,
    "message": null
  }
  ```

## 部署建议

- 生产环境中应限制CORS允许的源地址
- 考虑使用Gunicorn作为生产服务器
- 对于高并发场景，可以考虑添加缓存机制
- 建议使用GPU部署以获得更好的性能
