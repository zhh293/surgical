import os
import numpy as np
import logging
from segment_anything import sam_model_registry, SamPredictor

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_sam_model(model_type: str, checkpoint_path: str):
    """
    加载SAM模型权重文件

    Args:
        model_type: SAM模型类型，如"vit_h", "vit_l", "vit_b"
        checkpoint_path: 模型权重文件路径

    Returns:
        加载好的SAM模型实例

    Raises:
        FileNotFoundError: 如果检查点文件不存在
        ValueError: 如果模型类型不支持
        Exception: 其他加载错误
    """
    try:
        # 检查模型文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM模型权重文件不存在: {checkpoint_path}")

        # 检查模型类型是否支持
        supported_models = sam_model_registry.keys()
        if model_type not in supported_models:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型有: {list(supported_models)}")

        logger.info(f"开始加载SAM模型: {model_type}，权重路径: {checkpoint_path}")

        # 加载模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        logger.info(f"模型加载成功，参数数量: {sum(p.numel() for p in sam.parameters()):,}")
        return sam

    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        raise


def initialize_sam(model_type: str, checkpoint_path: str,
                   sam_model=None, device: str = None) -> SamPredictor:
    """
    初始化SAM预测器（支持传入已加载的模型，避免重复加载）

    Args:
        model_type: SAM模型类型
        checkpoint_path: 模型权重路径（备用，若sam_model为None则重新加载）
        sam_model: 已加载的SAM模型实例（可选）
        device: 运行设备（cuda/cpu）

    Returns:
        SamPredictor: 初始化后的预测器
    """
    try:
        # 若未传入已加载的模型，则调用load_sam_model加载
        if sam_model is None:
            sam_model = load_sam_model(model_type, checkpoint_path)

        # 自动选择设备
        if device is None:
            device = "cuda" if sam_model.device.type == "cuda" else "cpu"

        # 移动模型到指定设备
        sam_model.to(device=device)
        logger.info(f"模型已部署到设备: {device}")

        # 创建预测器
        predictor = SamPredictor(sam_model)
        logger.info("SAM预测器初始化完成")
        return predictor

    except Exception as e:
        logger.error(f"预测器初始化失败: {str(e)}", exc_info=True)
        raise


def predict_mask(
        predictor: SamPredictor,
        image: np.ndarray,
        points: list,
        labels: list,
        multimask_output: bool = False
) -> np.ndarray:
    """
    使用SAM模型预测分割掩码

    Args:
        predictor: 已初始化的SamPredictor
        image: 输入图像（PIL Image或numpy数组）
        points: 标注点坐标列表，格式为[(x1, y1), (x2, y2), ...]
        labels: 标注点标签列表，1表示前景，0表示背景
        multimask_output: 是否输出多个掩码

    Returns:
        np.ndarray: 预测的分割掩码，形状为(H, W)
    """
    try:
        # 将PIL Image转换为numpy数组
        if hasattr(image, 'convert'):  # 检查是否为PIL Image
            image_np = np.array(image)
        else:
            image_np = image.copy()  # 防止修改原始图像

        # 确保图像是RGB格式
        if image_np.ndim == 3 and image_np.shape[-1] == 4:
            logger.warning("输入图像是RGBA格式，自动转换为RGB")
            image_np = image_np[..., :3]

        # 设置图像
        predictor.set_image(image_np)
        logger.debug(f"已设置输入图像，形状: {image_np.shape}")

        # 转换点坐标格式
        input_points = np.array(points, dtype=np.float32)
        input_labels = np.array(labels, dtype=np.int32)

        logger.debug(f"使用标注点: {input_points}, 标签: {input_labels}")

        # 预测掩码
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output
        )

        # 选择最佳掩码（如果是多掩码输出，选择分数最高的）
        if multimask_output and len(masks) > 1:
            best_idx = np.argmax(scores)
            logger.debug(f"从{len(masks)}个掩码中选择最佳，分数: {scores[best_idx]:.4f}")
            return masks[best_idx]

        return masks[0]

    except Exception as e:
        logger.error(f"掩码预测失败: {str(e)}", exc_info=True)
        raise
