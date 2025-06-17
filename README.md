# SAM 语义分割标注工具

基于 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 的专业级语义分割标注平台，支持多类别、多种交互方式，适合科研与工业数据标注场景。

## 功能亮点

- 🖱️ **点交互标注**：支持正点/负点交互，快速获得高质量分割
- 🟦 **多框标注**：支持同时添加多个框，适合批量目标分割
- 🏷️ **多类别管理**：自定义类别名称与颜色，类别切换便捷
- 🎨 **实时可视化**：分割结果实时预览，支持彩色/叠加/语义掩码多种视图
- 💾 **多格式导出**：一键导出语义掩码、彩色掩码、实例掩码、叠加图及元数据
- 🧩 **现代化界面**：响应式布局，专业美观，操作流畅
- 📝 **标注历史管理**：支持已完成标注的查看与删除

## 快速开始

### 1. 环境准备

- Python 3.8+
- PyTorch 1.7+（建议CUDA环境）
- 下载 [SAM权重文件](https://github.com/facebookresearch/segment-anything#model-checkpoints)（如 `sam_vit_h_4b8939.pth`），放在项目根目录

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python app.py
```

或使用脚本：

```bash
./start.sh
```

### 4. 打开浏览器访问

```
http://localhost:5000
```

## 使用说明

### 标注流程

1. **上传图像**：点击左侧“图像上传”选择图片
2. **添加类别**：输入类别名称、选择颜色，点击“添加”
3. **选择工具**：点标注（正点/负点）或框选
4. **进行标注**：在图像上点击添加点或绘制多个框
5. **生成蒙版**：点击“生成蒙版”按钮
6. **保存结果**：点击“保存结果”导出所有分割结果

### 工具说明

- **正点(+)**：绿色，表示应包含的区域
- **负点(-)**：红色，表示应排除的区域
- **框选**：可添加多个框，适合批量目标分割
- **清除点/框/蒙版**：分别清除当前的点、框、蒙版
- **重置所有**：清空所有标注和结果

### 结果文件说明

保存后，`results/` 目录下会生成：

- `semantic_mask.png`：灰度语义掩码（像素值为类别ID）
- `colored_mask.png`：彩色语义掩码
- `overlay.png`：原图与分割结果叠加图
- `instance_{类别ID}.png`：每个类别的单独掩码
- `metadata.json`：标注元数据（类别、点、框等信息）

## 高级特性

- **多类别标注**：支持任意数量类别，颜色自定义
- **多框支持**：可同时添加多个框，自动合并分割结果
- **实时预览**：右侧面板实时显示所有类别的分割叠加效果
- **标注管理**：可删除已完成的类别分割结果

## 常见问题

- **模型加载慢**：首次加载SAM模型较慢，建议使用GPU
- **浏览器兼容性**：建议使用最新版 Chrome/Edge/Firefox
- **依赖问题**：如遇依赖问题，建议新建虚拟环境

## 目录结构

```
sam-backend/
├── app.py                # Flask后端主程序
├── templates/
│   └── index.html        # 前端主页面
├── uploads/              # 上传图片目录
├── results/              # 标注结果输出目录
├── requirements.txt      # Python依赖
├── README.md             # 项目说明
└── sam_vit_h_4b8939.pth  # SAM模型权重（需手动下载）
```

## 致谢

- [Meta AI Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Bootstrap 5](https://getbootstrap.com/)
- [Font Awesome](https://fontawesome.com/)

---

如有建议或问题，欢迎提交 [Issue](https://github.com/your-repo/issues) 或 PR！
