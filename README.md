# 个人网站 (Personal Site)

基于 [Astro](https://astro.build) 构建的个人网站 + 博客。

## 快速开始

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

## 项目结构

```
src/
├── components/        # 组件
│   ├── Nav.astro       # 导航栏
│   └── Footer.astro    # 页脚
├── layouts/           # 布局
│   ├── Base.astro      # 基础布局（含导航 + 页脚）
│   └── BlogPost.astro  # 博客文章布局
├── pages/             # 页面（自动路由）
│   ├── index.astro        # 首页
│   ├── about.astro        # 关于
│   ├── blog/
│   │   ├── index.astro    # 博客列表
│   │   ├── welcome.astro  # 示例文章1
│   │   └── getting-started.astro  # 示例文章2
│   ├── projects.astro     # 项目
│   └── contact.astro      # 联系
└── styles/
    └── global.css     # 全局样式
```

## 写新博客文章

在 `src/pages/blog/` 下新建 `.astro` 文件，参考现有文章格式：

```astro
---
import BlogPost from '../../layouts/BlogPost.astro';

const title = '文章标题';
const date = '2026-07-01';
---

<BlogPost title={title} date={date}>
  正文内容，支持 Markdown 语法。
</BlogPost>
```

别忘了在 `src/pages/blog/index.astro` 的文章列表中添加新文章。

## 自定义内容

替换以下占位内容为你自己的信息：
- `Your Name` → 你的名字（在 Nav.astro、Footer.astro、各页面中）
- 社交链接 → Footer.astro 和 contact.astro 中的 GitHub/Twitter/Email
- `astro.config.mjs` 中的 `site` 字段 → 你的实际域名

## 部署

详见 [DEPLOY.md](./DEPLOY.md)

- **GitHub Pages**: 推送到 GitHub 后通过 Actions 自动部署
- **Cloudflare Pages**: 连接 Git 仓库或用 Wrangler CLI
