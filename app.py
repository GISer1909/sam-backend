import os
import cv2
import numpy as np
import torch
import uuid
import json
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLOR_PALETTE = np.random.randint(0, 255, (100, 3), dtype=np.uint8)  # 用于不同类别的颜色

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 加载SAM模型
print(f"正在加载SAM模型到{DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)
predictor = SamPredictor(sam)
print("SAM模型加载完成！")

# 当前处理的图像信息
current_image = {
    'filename': None,
    'path': None,
    'height': None,
    'width': None,
    'masks': {},  # 存储所有蒙版 {类别ID: 蒙版数据}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存文件并生成唯一ID
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # 加载图像并设置到预测器
    image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    # 更新当前图像信息
    global current_image
    current_image = {
        'filename': filename,
        'path': filepath,
        'height': image.shape[0],
        'width': image.shape[1],
        'masks': {},
    }
    
    return jsonify({
        'success': True,
        'filename': filename,
        'width': image.shape[1],
        'height': image.shape[0]
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # 检查是否有图像被加载
    if current_image['filename'] is None:
        return jsonify({'error': '没有加载图像'}), 400
    
    class_id = data.get('classId', '1')  # 类别ID，默认为1
    mode = data.get('mode', 'points')  # 预测模式: 'points'、'box' 或 'boxes'
    
    # 根据不同模式预测
    if mode == 'points':
        # 获取点坐标和标签
        points = np.array(data['points'])
        labels = np.array(data['labels'])
        
        # 预测蒙版
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,  # 返回多个可能的蒙版
        )
    elif mode == 'box':
        # 获取框坐标 [x1, y1, x2, y2]
        box = np.array(data['box'])
        
        # 预测蒙版 - 使用框
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=True,
        )
    elif mode == 'boxes':
        # 多框模式
        boxes = data.get('boxes', [])
        if not boxes:
            return jsonify({'error': '未提供boxes参数'}), 400

        combined_mask = None
        best_score = 0.0
        all_boxes = []
        for box in boxes:
            box_np = np.array(box)
            masks, scores, logits = predictor.predict(
                box=box_np,
                multimask_output=True,
            )
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            score = float(scores[best_mask_idx])
            if combined_mask is None:
                combined_mask = mask.astype(bool)
            else:
                combined_mask = np.logical_or(combined_mask, mask.astype(bool))
            best_score = max(best_score, score)
            all_boxes.append(box)
        # 存储合并后的蒙版
        current_image['masks'][class_id] = {
            'mask': combined_mask,
            'score': best_score,
            'boxes': all_boxes
        }
        # 生成可视化图像
        mask_image = combined_mask.astype(np.uint8) * 255
        rgba = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)
        rgba[..., 0:3] = 255
        rgba[..., 3] = mask_image
        _, buffer = cv2.imencode('.png', rgba)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'success': True,
            'mask': mask_base64,
            'score': best_score,
            'class_id': class_id
        })
    else:
        return jsonify({'error': '不支持的预测模式'}), 400
    
    # 选择得分最高的蒙版
    best_mask_idx = np.argmax(scores)
    mask = masks[best_mask_idx]
    score = float(scores[best_mask_idx])
    
    # 存储蒙版
    current_image['masks'][class_id] = {
        'mask': mask.astype(bool),
        'score': score
    }
    
    # 如果有点信息，也存储
    if mode == 'points':
        current_image['masks'][class_id]['points'] = data['points']
        current_image['masks'][class_id]['labels'] = data['labels']
    # 如果是框模式，存储框信息
    elif mode == 'box':
        current_image['masks'][class_id]['box'] = data['box']
    
    # 生成可视化图像
    mask_image = mask.astype(np.uint8) * 255

    # 生成带alpha通道的PNG
    rgba = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)
    rgba[..., 0:3] = 255  # RGB全白（实际前端会覆盖颜色）
    rgba[..., 3] = mask_image  # alpha通道为mask

    # 转换为base64以便在前端显示
    _, buffer = cv2.imencode('.png', rgba)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'mask': mask_base64,
        'score': score,
        'class_id': class_id
    })

@app.route('/get_combined_mask', methods=['GET'])
def get_combined_mask():
    if current_image['filename'] is None or not current_image['masks']:
        return jsonify({'error': '没有加载图像或没有可用的蒙版'}), 400
    
    # 创建彩色分割图像
    height, width = current_image['height'], current_image['width']
    combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 为每个类别添加不同颜色
    for class_id, mask_data in current_image['masks'].items():
        mask = mask_data['mask']
        color = COLOR_PALETTE[int(class_id) % len(COLOR_PALETTE)]
        for c in range(3):
            combined_mask[:, :, c] = np.where(mask, color[c], combined_mask[:, :, c])
    
    # 转换为base64
    _, buffer = cv2.imencode('.png', combined_mask)
    combined_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'combined_mask': combined_base64
    })

@app.route('/save_result', methods=['POST'])
def save_result():
    if current_image['filename'] is None or not current_image['masks']:
        return jsonify({'error': '没有加载图像或没有可用的蒙版'}), 400
    
    data = request.json
    export_type = data.get('exportType', 'semantic')  # 'semantic' 或 'instance'
    
    # 原始图像
    image = cv2.cvtColor(cv2.imread(current_image['path']), cv2.COLOR_BGR2RGB)
    
    # 生成结果文件名
    base_filename = os.path.splitext(current_image['filename'])[0]
    result_dir = os.path.join(RESULTS_FOLDER, base_filename)
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建语义分割图像
    height, width = current_image['height'], current_image['width']
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 处理每个类别的蒙版
    mask_info = {}
    for class_id, mask_data in current_image['masks'].items():
        mask = mask_data['mask']
        class_id_int = int(class_id)
        
        # 更新语义掩码（数字表示类别）
        semantic_mask = np.where(mask, class_id_int, semantic_mask)
        
        # 更新彩色掩码（颜色表示类别）
        color = COLOR_PALETTE[class_id_int % len(COLOR_PALETTE)]
        for c in range(3):
            colored_mask[:, :, c] = np.where(mask, color[c], colored_mask[:, :, c])
        
        # 保存每个类别的单独掩码
        instance_filename = f"instance_{class_id}.png"
        instance_path = os.path.join(result_dir, instance_filename)
        cv2.imwrite(instance_path, mask.astype(np.uint8) * 255)
        
        # 创建掩码信息字典，根据不同标注方式添加不同字段
        mask_info_item = {
            'file': instance_filename,
            'score': mask_data['score']
        }
        
        # 根据存在的键添加对应信息
        if 'points' in mask_data:
            mask_info_item['points'] = mask_data['points']
            
        if 'labels' in mask_data:
            mask_info_item['labels'] = mask_data['labels']
            
        if 'box' in mask_data:
            mask_info_item['box'] = mask_data['box']
        
        # 添加到结果中
        mask_info[class_id] = mask_info_item
    
    # 保存语义分割掩码（灰度）
    semantic_path = os.path.join(result_dir, "semantic_mask.png")
    cv2.imwrite(semantic_path, semantic_mask)
    
    # 保存彩色语义分割掩码
    colored_path = os.path.join(result_dir, "colored_mask.png")
    cv2.imwrite(colored_path, colored_mask)
    
    # 生成半透明叠加效果
    alpha = 0.5
    overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(colored_mask[:, :, c] > 0, 
                                   colored_mask[:, :, c], 
                                   overlay[:, :, c])
    
    # 叠加图像和掩码
    overlay_path = os.path.join(result_dir, "overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 保存元数据
    metadata = {
        'original_image': current_image['filename'],
        'classes': mask_info,
        'export_type': export_type,
        'date': os.path.getmtime(current_image['path'])
    }
    
    with open(os.path.join(result_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return jsonify({
        'success': True,
        'result_folder': base_filename,
        'files': {
            'semantic': "semantic_mask.png",
            'colored': "colored_mask.png",
            'overlay': "overlay.png"
        }
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<path:filepath>')
def result_file(filepath):
    directory, filename = os.path.split(filepath)
    return send_from_directory(os.path.join(RESULTS_FOLDER, directory), filename)

@app.route('/clear', methods=['POST'])
def clear_current():
    global current_image
    current_image = {
        'filename': None,
        'path': None,
        'height': None,
        'width': None,
        'masks': {},
    }
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
