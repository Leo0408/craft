# GitHub推送指南

由于网络代理和GnuTLS的兼容性问题，推荐使用以下方法之一：

## 方法1：使用Personal Access Token（推荐）

### 步骤：

1. **创建Personal Access Token**
   - 访问：https://github.com/settings/tokens
   - 点击 "Generate new token" -> "Generate new token (classic)"
   - 设置名称（如：craft-push）
   - 选择过期时间
   - 勾选 `repo` 权限
   - 点击 "Generate token"
   - **重要**：复制生成的token（只显示一次）

2. **使用token推送**
   ```bash
   cd /home/fdse/zzy/craft
   export http_proxy=http://127.0.0.1:7890
   export https_proxy=http://127.0.0.1:7890
   git config --global http.sslVerify false
   git push -u origin main
   ```
   当提示输入用户名时，输入：`Leo0408`
   当提示输入密码时，输入：**你的Personal Access Token**（不是GitHub密码）

3. **或者直接在URL中包含token**（不推荐，但可以尝试）
   ```bash
   git remote set-url origin https://<TOKEN>@github.com/Leo0408/craft.git
   git push -u origin main
   ```

## 方法2：配置SSH密钥

1. **生成SSH密钥**（如果还没有）
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **添加SSH密钥到GitHub**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   复制输出的公钥，然后：
   - 访问：https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴公钥并保存

3. **使用SSH推送**
   ```bash
   cd /home/fdse/zzy/craft
   git remote set-url origin git@github.com:Leo0408/craft.git
   git push -u origin main
   ```

## 方法3：使用git credential helper存储token

```bash
# 配置credential helper
git config --global credential.helper store

# 推送时会提示输入用户名和token，输入一次后会保存
git push -u origin main
```

## 当前配置状态

- 远程仓库：`https://github.com/Leo0408/craft.git`
- 代理设置：`http://127.0.0.1:7890`
- SSL验证：已禁用（仅用于测试）

## 故障排除

如果仍然遇到问题：
1. 检查clash代理是否正常运行
2. 尝试重启clash
3. 检查clash的代理模式（可能需要设置为全局模式）
4. 考虑使用其他代理工具或VPN

