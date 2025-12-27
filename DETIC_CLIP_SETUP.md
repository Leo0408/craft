# DETIC + CLIP 检测器设置指南

## 概述

DETIC + CLIP 检测器是 CRAFT++ 框架中推荐的感知方案，相比 MDETR 具有更好的检测性能和鲁棒性。

## 安装步骤

### 1. 安装 DETIC

```bash
# 克隆 DETIC 仓库
git clone https://github.com/facebookresearch/Detic.git
cd Detic

# 安装依赖
pip install -r requirements.txt

# 安装 detectron2（如果还没有安装）
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 2. 下载 DETIC 模型权重

**⚠️ 如果遇到 403 Forbidden 错误**，请参考下面的故障排除部分。

```bash
# 在 Detic 目录下
mkdir -p models
cd models

# ============================================
# 方法 1: 使用 Clash 代理（推荐）
# ============================================
# 如果你的 Clash 运行在本地，通常端口是 7890（HTTP）和 7891（SOCKS5）
# 设置代理环境变量：
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
# 或者使用 SOCKS5（如果 Clash 支持）：
# export HTTP_PROXY=socks5://127.0.0.1:7891
# export HTTPS_PROXY=socks5://127.0.0.1:7891

# 然后使用 wget 下载
wget https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
# 或小模型
wget https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth

# ============================================
# 方法 2: 使用 curl with User-Agent（推荐，通常更可靠）
# ============================================
# curl 通常能更好地处理代理
curl -L -A "Mozilla/5.0" -o detic_LCOCOI21k_CLIP_R50_1x.pth \
  https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth

# ============================================
# 方法 3: 临时禁用代理（如果代理导致问题）
# ============================================
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
wget --user-agent="Mozilla/5.0" \
  https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth

# ============================================
# 方法 4: 使用 Python 脚本（自动处理代理问题）
# ============================================
# 在 craft 目录下运行：
python3 download_detic_model.py small  # 下载小模型
# 或
python3 download_detic_model.py large   # 下载大模型
```

### 3. 安装 CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 4. 安装 ByteTrack（可选，用于跟踪）

**注意**：ByteTrack 是可选的，如果安装失败，检测器仍然可以工作，只是没有跟踪功能。

#### 方法 1: 使用 pip（可能失败，需要编译工具）

```bash
pip install byte-track
```

如果失败，需要先安装编译依赖：

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake

# 然后重试
pip install byte-track
```

#### 方法 2: 从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack

# 安装依赖
pip install -r requirements.txt

# 安装（开发模式）
pip install -e .
```

#### 方法 3: 跳过 ByteTrack（推荐，如果不需要跟踪）

ByteTrack 是可选的。如果安装失败，可以：

1. **跳过安装**：代码会自动检测并禁用跟踪功能
2. **在代码中禁用跟踪**：
   ```python
   detector = DeticClipDetector(
       device=device,
       detic_threshold=0.3,
       clip_threshold=0.25,
       use_tracking=False  # 禁用跟踪
   )
   ```

**注意**：即使没有 ByteTrack，DETIC + CLIP 检测器仍然可以正常工作，只是没有多目标跟踪功能。

## 配置

### 在 demo2.ipynb 中使用

1. 在 Step 4 (Alternative) 中，设置：
   ```python
   DETECTION_METHOD = 'detic_clip'
   ```

2. 调整阈值（如果需要）：
   ```python
   detector = DeticClipDetector(
       device=device,
       detic_threshold=0.3,  # DETIC 检测阈值（0.0-1.0）
       clip_threshold=0.25,  # CLIP 语义相似度阈值（0.0-1.0）
       use_tracking=True     # 是否启用 ByteTrack 跟踪
   )
   ```

## 工作原理

### DETIC 检测阶段

1. **开放词表检测**：DETIC 使用 21k 类别的开放词表进行检测
2. **输出**：边界框、掩码、类别、置信度

### CLIP 过滤阶段

1. **Prompt 扩展**：将对象名称扩展为多个变体
   - "cup" → ["cup", "a cup", "the cup", "coffee cup"]
2. **语义匹配**：使用 CLIP 计算检测结果与目标对象的语义相似度
3. **过滤**：只保留相似度高于阈值的检测结果

### ByteTrack 跟踪阶段

1. **多目标跟踪**：为每个检测到的对象分配稳定的 ID
2. **处理遮挡**：当对象被遮挡时，保持 ID 不变
3. **ID 切换处理**：处理对象 ID 切换的情况

### Environment Memory 集成

1. **时序平滑**：使用跟踪 ID 进行跨帧的对象状态平滑
2. **遮挡预测**：基于跟踪信息预测被遮挡对象的位置
3. **置信度衰减**：当对象长时间未检测到时，降低置信度

## 优势

相比 MDETR 方案：

1. ✅ **更强的检测能力**：DETIC 支持 21k 类别，检测更准确
2. ✅ **更好的鲁棒性**：CLIP 语义过滤减少误检
3. ✅ **Prompt 扩展**：自动扩展对象名称，提高匹配率
4. ✅ **内置跟踪**：ByteTrack 提供稳定的对象跟踪
5. ✅ **更好的集成**：与 Environment Memory 无缝集成

## 故障排除

### 下载模型时遇到 403 Forbidden 错误

**问题**：使用 `wget` 下载 DETIC 模型时出现 `403 Forbidden` 错误。

**原因**：通常是代理配置问题，代理服务器不允许访问该 URL，或者需要正确配置代理。

**解决方案**（按推荐顺序）：

#### 1. **配置 Clash 代理**（如果使用 Clash）

首先确认 Clash 的代理端口（通常在 Clash 界面可以看到）：
- HTTP 代理端口：通常是 `7890`
- SOCKS5 代理端口：通常是 `7891`

```bash
# 设置 HTTP 代理（推荐）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 或者使用 SOCKS5（如果 Clash 支持）
# export HTTP_PROXY=socks5://127.0.0.1:7891
# export HTTPS_PROXY=socks5://127.0.0.1:7891

# 验证代理是否工作
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 然后下载
wget https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
```

**注意**：如果 Clash 设置了认证，可能需要：
```bash
export HTTP_PROXY=http://username:password@127.0.0.1:7890
export HTTPS_PROXY=http://username:password@127.0.0.1:7890
```

#### 2. **使用 curl with User-Agent**（推荐，通常更可靠）

```bash
# curl 通常能更好地处理代理
curl -L -A "Mozilla/5.0" -o detic_LCOCOI21k_CLIP_R50_1x.pth \
  https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
```

#### 3. **临时禁用代理后使用 wget**

如果代理导致问题，可以临时禁用：
```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
wget --user-agent="Mozilla/5.0" \
  https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
```

#### 4. **使用 Python 脚本**（自动处理代理）

```bash
# 在 craft 目录下
python3 download_detic_model.py small  # 下载小模型
python3 download_detic_model.py large   # 下载大模型
```

#### 5. **使用 aria2**（如果已安装，支持多线程下载）

```bash
aria2c -x 16 -s 16 \
  https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
```

#### 6. **手动下载**（如果所有自动方法都失败）
   - 在浏览器中打开以下 URL：
     - 小模型：https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
     - 大模型：https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
   - 下载后，将文件放到 `Detic/models/` 目录
   - 或者使用其他下载工具（如 IDM、迅雷等）

6. **使用镜像源**（如果有）：
   - 某些地区可能有镜像源
   - 检查 DETIC GitHub 仓库的 Issues 或 Wiki

7. **从 DETIC 官方仓库下载**：
   ```bash
   # 克隆 DETIC 仓库，可能包含下载脚本
   git clone https://github.com/facebookresearch/Detic.git
   cd Detic
   # 查看是否有下载脚本
   ls *.sh *.py | grep download
   ```

**注意**：如果持续遇到 403 错误，建议：
- 使用 VPN 或更换网络环境
- 在浏览器中手动下载
- 联系网络管理员检查防火墙设置

### DETIC 加载失败

- 检查 DETIC 是否正确安装
- 确认模型权重文件路径正确
- 检查 detectron2 版本兼容性

### CLIP 加载失败

- 安装 CLIP：`pip install git+https://github.com/openai/CLIP.git`
- 检查网络连接（首次运行需要下载模型）

### ByteTrack 不可用

- 跟踪功能是可选的，不影响检测
- 如需使用，按照安装步骤安装 ByteTrack

## 参考

- DETIC 论文：https://arxiv.org/abs/2201.02605
- DETIC 代码：https://github.com/facebookresearch/Detic
- CLIP 论文：https://arxiv.org/abs/2103.00020
- ByteTrack 论文：https://arxiv.org/abs/2110.06864

