#!/usr/bin/env python3
"""
下载 DETIC 模型权重的脚本
支持多种下载方式，自动处理代理问题
"""
import os
import sys
import urllib.request
import urllib.error

def download_file(url, output_path, use_proxy=False):
    """下载文件，支持代理设置"""
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 设置代理（如果需要）
    if use_proxy:
        # 如果设置了代理环境变量，使用它
        proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
        if proxy:
            print(f"使用代理: {proxy}")
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy,
                'https': proxy
            })
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        else:
            print("未检测到代理设置，尝试直接下载...")
    
    try:
        # 下载文件
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ 下载成功: {output_path}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"❌ 403 Forbidden 错误")
            print("可能的原因:")
            print("  1. 代理配置问题")
            print("  2. 需要认证")
            print("  3. URL 访问受限")
            print("\n解决方案:")
            print("  1. 临时禁用代理: unset HTTP_PROXY HTTPS_PROXY")
            print("  2. 使用 curl: curl -L -o <output> <url>")
            print("  3. 手动下载: 在浏览器中打开 URL 下载")
        else:
            print(f"❌ HTTP 错误 {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    # DETIC 模型 URL
    model_urls = {
        "large": "https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        "small": "https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth"
    }
    
    # 选择模型
    model_type = sys.argv[1] if len(sys.argv) > 1 else "small"
    if model_type not in model_urls:
        print(f"未知模型类型: {model_type}")
        print("可用类型: large, small")
        sys.exit(1)
    
    url = model_urls[model_type]
    output_dir = "models"  # 或 "Detic/models" 如果 Detic 目录存在
    output_path = os.path.join(output_dir, os.path.basename(url))
    
    # 尝试不使用代理下载
    print("=" * 60)
    print("尝试不使用代理下载...")
    print("=" * 60)
    if not download_file(url, output_path, use_proxy=False):
        print("\n" + "=" * 60)
        print("尝试使用代理下载...")
        print("=" * 60)
        download_file(url, output_path, use_proxy=True)
