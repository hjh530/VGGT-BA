# VGGT-BA: Bundle Adjustment for VGGT Camera Poses

对 VGGT 生成的相机位姿进行全局 Bundle Adjustment 优化。

## 环境配置

```bash
# 安装依赖
conda create -n VGGT-BA python=3.10 -y
conda activate VGGT-BA
conda install -c pytorch pytorch torchvision -y
pip install numpy pycolmap roma kornia tqdm pillow matplotlib scipy

# 依赖 vggt-omega 的图像加载模块
git clone https://github.com/Linketic/vggt-omega.git /path/to/vggt-omega
# 将 run_*.py 中 vggt-omega 路径改为实际路径：
# sys.path.insert(0, '/path/to/vggt-omega')
```

## 使用方法

### 1. 准备输入

```
input/<scene_name>/
├── images/              # 原始图像
├── predictions.npz      # VGGT 预测 (depth, extrinsics, intrinsics)
└── sparse/
    ├── 0/               # COLMAP 模型 (stereo SfM 结果)
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    └── 1/               # GT 位姿 (可选，用于评估)
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

### 2. 运行 BA 优化

```bash
conda activate VGGT-BA

# Ceres BA (推荐，二阶 LM 求解器，更精确)
python run_ceres_ba.py [scene_name]
# 默认 scene_name=input/test

# 或 Adam BA (一阶梯下降，速度更快)
python run_adam_ba.py [scene_name]
```

### 3. 评估与可视化

```bash
# 定量评估 (ATE, RRE, RTE, RRA, RTA, AUC)
python evaluate.py [scene_name]

# 轨迹可视化 (3D + XY + 逐帧误差 + 直方图)
python plot_trajectory.py --scene [scene_name]
```

### 4. 输出

```
output/<scene_name>/sparse/0/
├── cameras.bin          # 优化后的相机内参
├── images.bin           # 优化后的相机外参
├── points3D.bin         # 优化后的 3D 点 (带颜色)
├── traj_gt_vggt.png     # GT vs VGGT 轨迹图
├── traj_gt_ours.png     # GT vs 优化结果 轨迹图
└── traj_all.png         # 三合一对比图
```

## 匹配对生成策略

`utils/pairs.py` 提供两种配对模式，自动检测场景类型：

### 自动检测 (`mode='auto'`，默认)

| 场景类型 | 判断条件 | 策略 |
|---------|---------|------|
| **纯空中场景** (如 picture) | 无 `Terr_` 前缀图像 | **穷举法**：N×(N-1)/2 全配对 |
| **空地混合场景** (如 test) | 存在 `Terr_` 前缀图像 | **规则法**：距离最近邻 + 序列窗口 |

### 规则法详情

| 配对类型 | 规则 |
|---------|------|
| Air-Air | 每个 Air 图像匹配最近的 15 个 Air 邻居 |
| Ground-Ground | 滑动窗口 20 帧 + 回环匹配 |
| Air-Ground | 每个 Air 图像匹配最近的 5 个 Ground 图像 |

也可以手动指定模式：修改脚本中 `generate_pairs(img_names, ext_all, mode='exhaustive')` 或 `mode='rules'`。

## 三个 BA 脚本对比

| 脚本 | 优化方法 | 特点 |
|------|---------|------|
| `run_ceres_ba.py` | Ceres Solver (LM) | 二阶优化，精度最高，速度快 |
| `run_adam_ba.py` | Adam (GD) | 一阶优化，需要手动调参 |
| `run_ceres_ba_dense.py` | Ceres Solver (LM) | 密集匹配版本 |

## 依赖

- **PyTorch** >= 2.0 — 神经网络推理 (SuperPoint + LightGlue)
- **pycolmap** — Ceres BA 求解器 + COLMAP RANSAC 验证
- **roma** — 旋转数学 (四元数/旋转矩阵转换)
- **kornia** — 极线几何
- **numpy**, **scipy** — 数值计算
- **pillow** — 图像加载
- **tqdm** — 进度条
- **matplotlib** — 轨迹可视化

### 模型权重

首次运行会自动下载 SP+LG 权重 (`weights/sp_lg_100h.ckpt`)，或手动放置到 `weights/` 目录。

### vggt-omega

需安装 vggt-omega 用于图像预处理。如路径不同，修改脚本中 `sys.path.insert(0, '/path/to/vggt-omega')`。
