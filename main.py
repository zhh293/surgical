from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入SAM工具函数
from sam_utils import load_sam_model, predict_mask, initialize_sam

# 初始化FastAPI应用
app = FastAPI(title="SAM Segmentation Service", version="1.0")

# 配置CORS，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应指定具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义请求数据模型
class Point(BaseModel):
    x: int
    y: int
    label: int  # 1表示前景，0表示背景


class SamRequest(BaseModel):
    image: str  # Base64编码的图片
    points: List[Point]
    image_format: Optional[str] = "jpeg"  # 图片格式，默认为jpeg


# 定义响应数据模型
class SamResponse(BaseModel):
    mask: str  # Base64编码的掩码图片
    success: bool = True
    message: Optional[str] = None


# 加载SAM模型（显式调用load_sam_model）
try:
    sam_checkpoint = os.getenv("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
    model_type = os.getenv("SAM_MODEL_TYPE", "vit_b")

    logger.info(f"开始加载SAM模型: {model_type}，权重路径: {sam_checkpoint}")

    # 显式调用load_sam_model加载模型权重
    sam_model = load_sam_model(model_type, sam_checkpoint)

    # 初始化预测器（传入已加载的模型）
    predictor = initialize_sam(model_type, sam_checkpoint, sam_model=sam_model)

    logger.info("SAM模型加载成功")
except Exception as e:
    logger.error(f"SAM模型加载失败: {str(e)}")
    predictor = None


# 根路由
@app.get("/")
async def root():
    return {"message": "SAM Segmentation Service is running", "status": "healthy"}


# 健康检查路由
@app.get("/health")
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=500, detail="SAM model not loaded")
    return {"status": "healthy", "model_loaded": True}


# 分割预测路由
@app.post("/predict", response_model=SamResponse)
async def predict(request: SamRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="SAM model not loaded")

    try:
        # 1. 解析Base64图片
        logger.info("收到分割请求")

        # 去除Base64前缀
        if "," in request.image:
            base64_str = request.image.split(",")[1]
        else:
            base64_str = request.image

        # 解码Base64
        image_bytes = base64.b64decode(base64_str)

        # 转换为PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        logger.info(f"图片解析完成，尺寸: {image.size}")

        # 2. 解析标注点
        points = [(p.x, p.y) for p in request.points]
        labels = [p.label for p in request.points]

        logger.info(f"收到{len(points)}个标注点")

        # 3. 调用SAM模型进行预测
        mask = predict_mask(predictor, image, points, labels)

        # 4. 将掩码转换为Base64
        # 创建RGBA图像，掩码区域为红色半透明
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask == 1] = [255, 0, 0, 128]  # 红色，半透明

        # 转换为PIL Image
        mask_image = Image.fromarray(mask_rgba)

        # 编码为Base64
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

        logger.info("分割预测完成")

        return {"mask": mask_base64, "success": True}

    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
