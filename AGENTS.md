# Agent 开发规范

所有在本仓库工作的 Agent 必须严格遵守以下规则：

## 1. 文章标题强同步
修改 `src/pages/blog/*.md`（或 `.astro`）的 Frontmatter `title` 时，**必须同步更新** `src/pages/blog/index.astro` 中对应 `slug` 的 `title`，确保列表页与正文标题严格一致。

## 2. 图片/图表标题必须居中
博客正文中的图片说明/图表标题（如 `*图 1：说明文字*` 或 `<figcaption>`）必须居中对齐，严禁左对齐。使用 Markdown 斜体图注语法（`*图注文字*`）或显式添加居中样式 `<p align="center">`。
