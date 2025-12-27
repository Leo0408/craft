# DETIC 模型手动下载指南

## 问题

如果使用 `wget` 或 `curl` 下载 DETIC 模型时遇到 `403 Forbidden` 错误，请按照以下步骤手动下载。

## 解决方案

### 方法 1: 浏览器下载（最简单）

1. **打开浏览器**，访问以下 URL：

   **小模型**（推荐，更快）：
   ```
   https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
   ```

   **大模型**（更准确，但更大）：
   ```
   https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
   ```

2. **下载文件**到本地

3. **创建目录并移动文件**：
   ```bash
   # 如果 Detic 目录存在
   mkdir -p Detic/models
   mv ~/Downloads/detic_LCOCOI21k_CLIP_R50_1x.pth Detic/models/
   
   # 或者直接放到 models 目录
   mkdir -p models
   mv ~/Downloads/detic_LCOCOI21k_CLIP_R50_1x.pth models/
   ```

### 方法 2: 使用下载工具

如果浏览器下载也失败，可以使用下载工具：

1. **IDM (Internet Download Manager)**
2. **迅雷**
3. **aria2**（命令行工具）：
   ```bash
   # 安装 aria2
   sudo apt-get install aria2  # Ubuntu/Debian
   # 或
   brew install aria2  # macOS
   
   # 使用 aria2 下载
   aria2c -x 16 -s 16 \
     https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
   ```

### 方法 3: 使用 Python requests（如果可用）

```python
import requests

url = "https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth"
output = "detic_LCOCOI21k_CLIP_R50_1x.pth"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers, stream=True)
with open(output, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"✅ 下载完成: {output}")
```

### 方法 4: 检查 DETIC 官方仓库

DETIC 的 GitHub 仓库可能提供其他下载方式：

```bash
git clone https://github.com/facebookresearch/Detic.git
cd Detic
# 查看 README 或 scripts 目录
cat README.md | grep -i download
ls scripts/
```

### 方法 5: 使用代理或 VPN

如果是因为网络限制：

1. **配置代理**：
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   wget https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth
   ```

2. **使用 VPN**：连接到其他地区的 VPN 后重试

## 验证下载

下载完成后，验证文件：

```bash
# 检查文件大小（小模型约 47MB，大模型约 1.2GB）
ls -lh models/detic_*.pth

# 检查文件完整性（如果提供了 checksum）
# md5sum models/detic_LCOCOI21k_CLIP_R50_1x.pth
```

## 文件位置

确保模型文件在正确的位置：

```
Detic/
├── models/
│   ├── detic_LCOCOI21k_CLIP_R50_1x.pth  # 小模型
│   └── detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth  # 大模型
└── ...
```

或者在项目根目录：

```
craft/
├── models/
│   └── detic_LCOCOI21k_CLIP_R50_1x.pth
└── ...
```

## 如果仍然无法下载

1. **检查网络连接**：确保可以访问互联网
2. **检查防火墙**：确保没有阻止访问
3. **联系管理员**：如果是公司/学校网络，可能需要管理员配置
4. **使用其他网络**：尝试使用手机热点或其他网络
5. **查看 DETIC Issues**：在 GitHub 上查看是否有类似问题

## 临时解决方案

如果暂时无法下载模型，可以：

1. **使用 MDETR 方案**（在 demo2.ipynb 中设置 `DETECTION_METHOD = 'mdetr'`）
2. **使用其他检测器**：如 Grounding DINO
3. **等待网络问题解决**后再下载

