# RKNode 技术方案 PPT 重写实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于当前项目实现，将 `docs/RKNode平台介绍与节点数据流.pptx` 重写为面向技术方案汇报的 16 页架构型演示文稿。

**Architecture:** 保留 `scripts/generate_platform_ppt.py` 作为唯一内容生成源，以现有绘图辅助函数构建控制面、训练/转换、Desired Revision、RK3588 数据面、业务分析、输出与部署安全七类图示。增加独立验证脚本从生成后的 PPTX 提取文本、检查标题/关键事实/禁用表述和页面边界，再通过 LibreOffice 渲染 PDF 做视觉验收。

**Tech Stack:** Python 3、python-pptx、Pillow、Playwright、LibreOffice、Poppler/PDF 渲染工具。

---

## 文件结构

- Modify: `scripts/generate_platform_ppt.py`，维护 16 页页面内容、布局和视觉样式。
- Create: `scripts/validate_platform_ppt.py`，验证页面数、标题、关键事实、禁用字符串和形状边界。
- Modify: `docs/ppt-assets/overview.png`，当前平台工作台截图，不包含敏感地址或令牌。
- Modify: `docs/ppt-assets/inference.png`，当前推理节点页面截图，不包含敏感地址或令牌。
- Modify: `docs/RKNode平台介绍与节点数据流.pptx`，最终生成物。
- Create during verification: `/tmp/rknode-ppt-render/`，PDF、逐页 PNG 和总览图；该目录不进入项目交付。

### Task 1: 建立可执行的 PPT 内容契约

**Files:**
- Create: `scripts/validate_platform_ppt.py`
- Inspect: `docs/RKNode平台介绍与节点数据流.pptx`

- [x] **Step 1: 写入对 16 页结构和关键事实的验证器**

验证器使用以下固定契约，并输出所有问题后以非零状态退出：

```python
EXPECTED_TITLES = [
    "RK3588 模型全生命周期平台",
    "平台定位：中央控制面连接异构执行节点",
    "模型生命周期：从数据集到板端业务结果",
    "部署拓扑：中心平台与四类执行节点",
    "直连节点协议：身份、能力、健康、容量与鉴权",
    "训练调度：Torch / Paddle × CPU / CUDA",
    "数据集与训练产物：格式、日志、指标、权重与 ONNX",
    "RKNN 转换：Toolkit2、量化、板端验证与部署就绪",
    "不可变 Release：制品、校验和与运行契约",
    "Desired Revision：多板下发、原子激活与回滚",
    "RK3588 数据面：RKMPP、RKNN 与 ByteTrack",
    "业务分析：二级推理、区域、越线、OSD 与事件媒体",
    "统一输出：JSONL、HTTP、Kafka、ZLM SEI 与预览",
    "可靠性：幂等、缓存清理、恢复与最终收敛",
    "部署与安全：在线/离线镜像、Compose、Token 与隧道",
    "核心价值与技术边界",
]

REQUIRED_TERMS = [
    "direct", "Torch", "Paddle", "CUDA", "RKNN Toolkit2",
    "SHA256", "Desired Revision", "RKMPP", "ByteTrack",
    "二级推理", "区域", "越线", "Kafka", "ZLM SEI",
    "完全离线", "VPN", "SSH 隧道", "目标环境最终验收",
]

FORBIDDEN_TERMS = [
    "2026-08-12", "中心平台固定端口 8080", "浏览器直连板卡",
    "privileged: true", "人脸识别", "dev-admin-token",
]
```

验证逻辑还应检查：页面数恰为 16；每页期望标题在对应页出现；所有文本框和图片位于 `slide_width × slide_height` 内；PPTX 内没有 IPv4 地址、Bearer Token 或私钥头。

- [x] **Step 2: 运行旧版 PPT 验证并确认失败**

Run:

```bash
/tmp/rknode-ppt-env/bin/python scripts/validate_platform_ppt.py docs/RKNode平台介绍与节点数据流.pptx
```

Expected: `FAIL`，至少报告旧日期 `2026-08-12`、缺少 `RKMPP`、`ByteTrack`、`Kafka`、`ZLM SEI` 和 `目标环境最终验收`。

### Task 2: 刷新当前平台截图

**Files:**
- Modify: `docs/ppt-assets/overview.png`
- Modify: `docs/ppt-assets/inference.png`

- [x] **Step 1: 验证前端可访问并获取当前节点状态**

Run:

```bash
curl -fsS http://127.0.0.1:5173/ >/dev/null
curl -fsS -H 'Authorization: Bearer admin' http://127.0.0.1:5173/api/v1/service-endpoints
```

Expected: 首页返回成功；接口中包含 trainer、converter、inference 三种节点，且响应不返回节点 Token。

- [x] **Step 2: 使用 Playwright 写入会话令牌并截图工作台**

Run:

```bash
node -e "const { chromium }=require('playwright'); (async()=>{const b=await chromium.launch({headless:true}); const p=await b.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1}); await p.goto('http://127.0.0.1:5173/'); await p.evaluate(()=>sessionStorage.setItem('rknode.adminToken','admin')); await p.reload(); await p.waitForLoadState('networkidle'); await p.screenshot({path:'docs/ppt-assets/overview.png'}); await b.close();})().catch(e=>{console.error(e);process.exit(1)})"
```

Expected: `overview.png` 为 1600×1000，侧栏显示训练、转换、推理三类节点状态，不显示令牌或基础设施地址。

- [x] **Step 3: 截取推理页作为当前能力证据**

Run:

```bash
node -e "const { chromium }=require('playwright'); (async()=>{const b=await chromium.launch({headless:true}); const p=await b.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1}); await p.goto('http://127.0.0.1:5173/#/inference'); await p.evaluate(()=>sessionStorage.setItem('rknode.adminToken','admin')); await p.reload(); await p.waitForLoadState('networkidle'); await p.screenshot({path:'docs/ppt-assets/inference.png'}); await b.close();})().catch(e=>{console.error(e);process.exit(1)})"
```

Expected: `inference.png` 为 1600×1000，显示当前 RK3588 推理节点；不出现管理员令牌、SSH 配置或节点真实地址。

### Task 3: 重写控制面和生命周期页面（1-5 页）

**Files:**
- Modify: `scripts/generate_platform_ppt.py`

- [x] **Step 1: 更新封面事实与页面元数据**

将封面日期改为 `2026-08-14`，副标题改为“数据集 → 训练 → RKNN 转换 → 多板下发 → 板端业务推理”，并让元数据关键词覆盖 direct、Desired Revision、RKMPP、ByteTrack、Kafka 和 ZLM SEI。

```python
prs.core_properties.subject = "中央控制面、直连节点与 RK3588 推理数据流"
prs.core_properties.keywords = (
    "RK3588, RKNN, direct node, Desired Revision, RKMPP, "
    "ByteTrack, Kafka, ZLM SEI"
)
```

- [x] **Step 2: 按已确认结构重写第 2-5 页**

页面结论固定为：

```python
CONTROL_PLANE_SLIDES = {
    2: "中央控制面统一保存资产、任务、Release、期望状态和审计信息",
    3: "每个生命周期阶段都有明确输入、制品、验证和责任边界",
    4: "Web/API、训练、转换和推理节点可独立部署并按职责横向扩展",
    5: "direct 节点必须同时满足身份、能力、健康、容量与独立鉴权",
}
```

第 4 页 Web 标签使用 `可配置 · 当前 5173`；训练/转换节点标注 `10081`，推理节点标注 `10082`。第 5 页明确 direct 为新节点主线，Worker/Agent 仅兼容；协议图展示 `/health`、`/capabilities`、`dispatch`、`desired revision` 和 `cache cleanup`。

- [x] **Step 3: 运行语法检查**

Run:

```bash
/tmp/rknode-ppt-env/bin/python -m py_compile scripts/generate_platform_ppt.py
```

Expected: exit code 0，无语法错误。

### Task 4: 重写训练、转换和发布页面（6-10 页）

**Files:**
- Modify: `scripts/generate_platform_ppt.py`

- [x] **Step 1: 用训练矩阵和回源时序重写第 6-7 页**

第 6 页必须显示完整矩阵：

```python
TRAIN_MATRIX = [
    ("Torch", "CPU", "YOLO / DeepLabV3+", "trainer-torch"),
    ("Torch", "CUDA", "YOLO / DeepLabV3+", "trainer-torch + CUDA overlay"),
    ("Paddle", "CPU", "PPOCR Det / Rec", "trainer-paddle"),
    ("Paddle", "CUDA", "PPOCR Det / Rec", "trainer-paddle + CUDA overlay"),
]
```

时序固定为“创建任务 → direct dispatcher 下发 jobId → 节点按授权回源数据 → 上报日志/指标 → 上传权重与 ONNX”。第 7 页左侧使用当前训练截图，右侧展示 COCO/VOC/YOLO/MASK/PPOCR、PT/Pdparams、ONNX、日志和指标。

- [x] **Step 2: 用板端验收链路重写第 8 页**

流程固定为：输入 ONNX/量化配置 → 回源下载 → Toolkit2 构建 → RK3588 Runtime 样本推理 → 性能/CPU fallback 审计 → RKNN/验证报告/readiness 回传。必须明确“转换成功不等于部署就绪”。

- [x] **Step 3: 强化第 9-10 页不可变发布与原子下发**

第 9 页包含主模型、全部二级模型、SHA256、manifest、适配器、输入输出契约和验证报告。第 10 页候选版本步骤固定为：

```python
REVISION_STEPS = [
    "下载全部制品", "SHA256 校验", "静态契约检查", "Runtime 探测",
    "模型预热", "排空旧流", "原子激活", "健康上报",
]
```

失败语义写为“候选目录失败即丢弃；current/previous 保持完整；节点重连后继续向 desired revision 收敛”。

### Task 5: 重写 RK3588 数据面和业务输出页面（11-13 页）

**Files:**
- Modify: `scripts/generate_platform_ppt.py`

- [x] **Step 1: 第 11 页画出完整板端数据面**

主图必须按以下顺序显示，并注明未启用节点从图中省略：

```text
RKMPP / OpenCV Capture -> Primary RKNN -> ByteTrack
-> Secondary RKNN -> Analytics -> OSD / Event Media -> Outputs
```

页面同时说明 RKMPP 适用于 RTSP H.264/H.265；OpenCV 保持旧任务兼容；NPU Core 0/1/2 支持 shared/exclusive 策略。

- [x] **Step 2: 第 12 页展示业务分析状态机**

区域/越线使用跟踪框底边中心点和 track ID；二级推理是普通非人脸 YOLO，结果通过 `parentTrackId`/`parentDetectionIndex` 关联；事件媒体包括结构化事件、JPEG 抓拍和 RKMPP 可用时的前后录像。

- [x] **Step 3: 第 13 页展示共用协议的五路输出**

```python
OUTPUTS = [
    ("JSONL", "板端持久化"),
    ("HTTP", "业务系统推送"),
    ("Kafka", "异步有界消息"),
    ("ZLM SEI", "原视频码流无重编码注入"),
    ("Preview", "独立限频 JPEG 旁路"),
]
```

必须写明 JSONL/HTTP/Kafka/ZLM SEI 共享 `schema v2` 结构化协议，sink 失败不改变 NPU 任务健康；浏览器只访问中心预览会话，不直连板卡或 RTSP。

### Task 6: 重写可靠性、部署安全和边界页面（14-16 页）

**Files:**
- Modify: `scripts/generate_platform_ppt.py`

- [x] **Step 1: 第 14 页统一故障和清理语义**

展示节点离线、任务失败、revision 激活失败、任务删除四类场景；底部机制固定为 `health identity 校验 · capacity 计算 · 幂等 dispatch · desired/actual 对账 · node_cleanup 重试 · 有界容器日志`。

- [x] **Step 2: 第 15 页加入完整部署矩阵**

页面包含在线镜像、完全离线镜像包、中央 Compose、CPU/CUDA 训练 overlay、RK3588 转换/推理双服务、节点独立 Token、防火墙、VPN 和 SSH 隧道。安全边界禁止 `privileged`，强调显式映射 NPU/MPP/RGA/dma-heap 设备。

- [x] **Step 3: 第 16 页区分已实现能力和最终验收项**

已实现区覆盖 direct、Torch/Paddle、CPU/CUDA、RKNN/Release/Revision、RKMPP/ByteTrack、业务分析和统一输出。边界区原文使用：

```text
目标环境最终验收：至少一条真实文件流或 RTSP 完成硬解码、NPU 推理、
跟踪、业务分析、预览与结构化输出；外部 Kafka 消费和 ZLM/SEI 端到端联调。
```

### Task 7: 生成、自动验证和视觉验收

**Files:**
- Modify: `docs/RKNode平台介绍与节点数据流.pptx`
- Inspect: `/tmp/rknode-ppt-render/platform.pdf`
- Inspect: `/tmp/rknode-ppt-render/slide-*.png`
- Inspect: `/tmp/rknode-ppt-render/contact-sheet.png`

- [x] **Step 1: 连续生成两次并验证确定性结构**

Run:

```bash
/tmp/rknode-ppt-env/bin/python scripts/generate_platform_ppt.py
/tmp/rknode-ppt-env/bin/python scripts/generate_platform_ppt.py
/tmp/rknode-ppt-env/bin/python scripts/validate_platform_ppt.py docs/RKNode平台介绍与节点数据流.pptx
```

Expected: 两次生成均输出同一路径；验证器输出 `PASS: 16 slides`。

- [x] **Step 2: 通过 LibreOffice 导出 PDF**

Run:

```bash
mkdir -p /tmp/rknode-ppt-render
libreoffice --headless --convert-to pdf --outdir /tmp/rknode-ppt-render docs/RKNode平台介绍与节点数据流.pptx
```

Expected: 生成 `/tmp/rknode-ppt-render/RKNode平台介绍与节点数据流.pdf`，LibreOffice 无解析错误。

- [x] **Step 3: 渲染逐页图片并生成总览**

使用系统可用的 PDF 渲染器；若没有 Poppler，则使用 LibreOffice 导出后的 PDF 配合 Python PDF 库渲染。输出必须是 16 张同尺寸 PNG，并组合为 4×4 总览图。

Expected: 每页非空，分辨率一致；总览图清楚显示 16 页页面结构。

- [x] **Step 4: 人工视觉检查并修复**

逐页检查文字溢出、重叠、截断、图形越界、截图空白、颜色语义和页码。对发现的问题修改生成器后，从 Step 1 重新执行完整验证。

- [x] **Step 5: 提取文本做最终敏感信息审计**

Run:

```bash
/tmp/rknode-ppt-env/bin/python scripts/validate_platform_ppt.py --dump-text docs/RKNode平台介绍与节点数据流.pptx > /tmp/rknode-ppt-render/deck-text.txt
rg -n '2026-08-12|Bearer |BEGIN .*PRIVATE KEY|([0-9]{1,3}\.){3}[0-9]{1,3}|dev-admin-token' /tmp/rknode-ppt-render/deck-text.txt
```

Expected: `rg` 无匹配；PPT 中没有真实 IP、令牌、私钥或旧日期。

### Task 8: 交付核对

**Files:**
- Inspect: `docs/RKNode平台介绍与节点数据流.pptx`
- Inspect: `scripts/generate_platform_ppt.py`
- Inspect: `scripts/validate_platform_ppt.py`

- [x] **Step 1: 运行最终质量门禁**

Run:

```bash
/tmp/rknode-ppt-env/bin/python -m py_compile scripts/generate_platform_ppt.py scripts/validate_platform_ppt.py
/tmp/rknode-ppt-env/bin/python scripts/validate_platform_ppt.py docs/RKNode平台介绍与节点数据流.pptx
```

Expected: 语法检查成功；内容验证输出 `PASS: 16 slides`。

- [x] **Step 2: 记录环境限制**

项目根目录不是 Git 仓库，因此不执行或声称 Git 提交。最终交付应报告 PPTX 路径、页数、验证命令结果、视觉检查结果，以及真实流/Kafka/ZLM 仍属于目标环境验收的边界。
