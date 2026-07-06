# Cloudflare Pages 部署配置

## 方式一：通过 Cloudflare Dashboard 连接 GitHub

1. 登录 https://dash.cloudflare.com → Pages → Create a project → Connect to Git
2. 选择你的 GitHub 仓库
3. 填写以下配置：
   - **Framework preset**: Astro
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
   - **Node version**: 22
4. 点击 Save and Deploy

## 方式二：通过 Wrangler CLI 直接部署

```bash
# 安装 wrangler
npm install -g wrangler

# 登录（首次需要）
wrangler login

# 部署
wrangler pages deploy dist --project-name=personal-site
```

## 方式三：通过 GitHub Actions 自动部署

在仓库 Settings → Secrets 添加：
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

然后在 `.github/workflows/` 添加 Cloudflare 部署 workflow。
