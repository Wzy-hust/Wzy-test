# Wzy-test

本仓库包含两个 Jupyter Notebook：

- `Untitled.ipynb`：数据加载与预处理示例（当前为基于 Iris 数据集的随机森林实现与评估，包括混淆矩阵、特征重要性可视化）。
- `Untitled2.ipynb`：基于 PyTorch 的 ResNet34 二分类训练与验证流程（含数据集划分、数据增强、训练记录与可视化）。

## 文件说明

### 1. Untitled.ipynb
- 主要内容：
  - 自实现 `DecisionTree` / `RandomForest`
  - 读取 `iris.data`，进行去重、标签编码、训练/测试集划分
  - 输出准确率、绘制混淆矩阵与特征重要性
- 注意：Notebook 中数据路径为本地路径（如 `D:\下载\iris\iris.data`），在其他机器上运行需要改成你自己的数据文件路径。

### 2. Untitled2.ipynb
- 主要内容：
  - 使用 `torchvision.models.resnet34` 构建二分类模型（替换 fc 层）
  - `ImageFolder` 读取本地数据集，按比例拆分训练/验证集（分层抽样）
  - 训练与验证循环、保存最佳模型、绘制训练曲线（输出 `training_history.png`）
- 注意：Notebook 中数据集根目录和模型保存路径为本地路径（如 `D:\QQ\糖足数据...`），运行前需要改成你自己的路径。

## 环境依赖（大致）
- Python
- Jupyter Notebook
- Untitled.ipynb：numpy / pandas / scikit-learn / matplotlib / seaborn
- Untitled2.ipynb：torch / torchvision / sklearn / tqdm / matplotlib
