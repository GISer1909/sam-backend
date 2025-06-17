import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# 模型配置
MODEL_TYPE = "vit_h"  # 可选: "vit_b", "vit_l", "vit_h", "efficientvit_sam_b"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device)
predictor = SamPredictor(sam)

# 加载图像
image = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# 框选交互（左上角坐标x1,y1, 右下角x2,y2）
input_box = np.array([100, 100, 500, 400])  # 示例框选区域
masks, _, _ = predictor.predict(box=input_box, multimask_output=False)

# 可视化结果
mask_display = masks[0].astype(np.uint8) * 255
cv2.imwrite("output_mask.png", mask_display)