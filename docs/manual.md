% Articulate — 机械臂代码生成 Agent 工具
% 项目介绍与使用说明书
% v1.0 | 2026-05

\newpage

# 项目概述

## 项目简介

Articulate 是一个机械臂控制代码自动生成工具。用户用自然语言描述任务需求（如"从 (0.3,0,0.2) 抓取物体放到 (0.6,0,0.3)"），Articulate 经过 5 个阶段自动输出部署到真实机械臂的机器人程序。

```
┌─────────────────────────────────────────────────────────┐
│  用户输入: "Pick and place from (0.3,0,0.2) to ..."      │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   5-Stage Pipeline                       │
│                                                         │
│  ① 需求分析 → ② 技术方案 → ③ 代码生成 → ④ 仿真验证 → ⑤ 部署   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│  输出: UR .script / KUKA KRL / ABB RAPID 机器人程序      │
│        ROS2 控制代码包                                   │
│        仿真验证报告 + 安全检清单                           │
└─────────────────────────────────────────────────────────┘
```

整个流程由大语言模型（LLM）驱动，结合预置的机械臂运动学库和代码模板，在自动化程度和输出质量之间取得平衡。

## 核心能力

- **自然语言→机器人代码**：输入日常语言描述，无需手写运动学或轨迹规划
- **多阶段自动流水线**：从需求分析到部署包生成，5 个阶段全自动串联
- **混合决策路由**：标准化子任务走模板（高速可靠），复杂逻辑走 LLM 生成（灵活）
- **仿真预验证**：MuJoCo 仿真环境下验证轨迹安全性，14 项指标体系量化评估，含代码执行验证
- **自动修复循环**：仿真不通过时自动诊断并调整轨迹参数，最多尝试 3 次
- **多品牌部署**：支持 Universal Robots (.script)、KUKA (KRL)、ABB (RAPID) 三种主流品牌
- **断点续行**：流水线状态持久化到项目目录，随时中断后可从当前阶段继续

## 适用场景

| 场景 | 说明 |
|------|------|
| 产线调试 | 快速生成抓取/搬运/装配的机械臂程序原型 |
| 方案验证 | 在仿真环境中验证轨迹可行性后再上真实产线 |
| 技术评估 | 对比不同臂型的运动学性能和轨迹质量 |
| 教学演示 | 展示 LLM + 机器人学的交叉应用 |
| 集成开发 | 为 ROS2 机器人项目生成控制节点骨架代码 |

### 支持的任务类型

- **pick_and_place**: 抓取搬运（最成熟）
- **welding**: 焊接轨迹
- **spraying**: 喷涂轨迹
- **palletizing**: 码垛
- **assembly**: 装配
- **custom**: 自定义任务

### 支持的机械臂构型

3 种预置臂型，基于 DH 参数建模：

| 臂型 | 自由度 | 特点 | 工作半径 | 额定载荷 |
|------|--------|------|---------|---------|
| six_dof_standard | 6-DOF | 标准工业臂 | ~0.75m | ~8kg |
| six_dof_collaborative | 6-DOF | 协作臂（轻载） | ~0.55m | ~5kg |
| seven_dof_standard | 7-DOF | 冗余自由度臂 | ~0.8m | ~6kg |

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 编程语言 | Python 3.10+ |
| CLI 框架 | Typer + Rich |
| 数据模型 | Pydantic + dataclass |
| LLM 接口 | Anthropic SDK + OpenAI-compatible |
| 运动学 | Pinocchio（优先）/ 纯 numpy（降级） |
| 仿真 | MuJoCo 3.0+（优先）/ 纯运动学（降级） |
| 代码模板 | Jinja2 |
| 模板管理 | YAML 定义 + Jinja2 渲染 |
| 包管理 | setuptools / pip |

\newpage

# 系统架构

## 整体架构

Articulate 采用分层架构，核心是 **Skill 门面 + 5 阶段流水线**。

```
┌────────────────────────────────────────────────────────────┐
│                        CLI 层                               │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │ generate │  │  init  │ │  plan  │ │ codegen│ │simulate│ │deploy│
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                    Pipeline 层                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              PipelineOrchestrator                     │  │
│  │  ① Requirement → ② Approach → ③ Codegen → ④ Sim → ⑤ Deploy │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              PipelineState (状态持久化)               │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                    Skill 层（门面）                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ PromptMgr│ │ Router   │ │ Kinematics│ │ Planning     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ROS2Gen   │ │Converters│ │URDFLoader│ │Dynamics      │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                    LLM 层                                    │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │  ClaudeClient   │  │  DeepSeekClient │  ← OpenAI兼容     │
│  └────────────────┘  └────────────────┘                    │
│        + 指数退避重试  + 结构化输出解析                     │
└─────────────────────────────────────────────────────────────┘
```

## 五阶段流水线

每个阶段接收 `StageContext`，执行核心逻辑，更新上下文后传递给下一阶段。

### 阶段 ①：需求分析 (Requirement Analysis)

**输入**：用户自然语言需求字符串

**处理流程**：
1. 加载 `requirement_analysis.yaml` 系统提示词
2. 调用 LLM 的结构化输出接口，从自然语言提取：
   - 任务类型（pick_and_place / welding / ...）
   - 关键路径点（位置 + 可选的姿态）
   - 末端执行器类型
   - 速度和精度要求
   - 环境描述和障碍物信息
   - 缺失信息列表
3. 如果检测到信息不完整（如缺少末端朝向、速度要求），提示用户补充
4. 用户确认需求文档

**输出**：`RequirementDocument` 结构化需求文档

**容错**：LLM 结构化解析失败时自动重试（最多 3 次），重试时附带错误信息指导模型修正输出格式。

### 阶段 ②：技术方案 (Technical Approach)

**输入**：`RequirementDocument`

**处理流程**：
1. 从 3 种预置臂型中自动选择匹配的臂型
2. 加载 `technical_design.yaml` 系统提示词
3. 调用 LLM 设计技术方案，包括：
   - 运动学策略（解析法/数值法/混合法）
   - 轨迹类型组合（PTP/LIN/CIRC/SPLINE）
   - ROS2 架构设计（节点、话题、服务、动作）
   - 风险评估（低/中/高 + 缓解措施）
   - 方案文字描述
4. 风险等级为高时提示用户注意
5. 用户确认技术方案

**输出**：`TechnicalApproach` 完整技术方案

### 阶段 ③：代码生成 (Code Generation)

**输入**：`TechnicalApproach`

**核心组件**：`CodeGenerationEngine`

**处理流程**：
1. **子任务分解**：将技术方案拆解为独立的子任务
   - LLM 分解（优先） → `_default_decomposition`（降级）
   - 典型子任务：运动学求解器、轨迹规划器、ROS2 控制节点、启动文件、包配置
2. **路由决策**：对每个子任务调用 `DecisionRouter`
   - 规则匹配优先：子任务名匹配预定义规则表 → 确定路由路径
   - LLM 分类降级：无规则匹配时由 LLM 判断走哪个路径
   - 低置信度触发用户仲裁
3. **代码生成**：根据路由结果选择生成方式
   - **Library 路径**：调用 Jinja2 模板渲染标准化代码
   - **Prompt 路径**：调用 LLM 生成定制化代码
4. **组装合并**：将所有子任务产出的文件合并为完整包结构
5. **验证**：ast.parse 语法检查、必需文件检查、危险模式检查
6. 展示代码摘要供用户确认

**路由规则表**（14 条预置规则）：

| 子任务模式 | 路由 | 置信度 | 说明 |
|-----------|------|--------|------|
| forward_kinematics | library | 0.95 | 标准 FK |
| inverse_kinematics | library | 0.90 | 标准 IK |
| inverse_kinematics_7dof | library_with_prompt | 0.70 | 7-DOF 带冗余处理 |
| trajectory_planner | library | 0.90 | 通用轨迹规划 |
| trajectory_ptp | library | 0.95 | PTP 轨迹 |
| trajectory_linear | library | 0.90 | LIN 轨迹 |
| trajectory_circular | library | 0.85 | CIRC 轨迹 |
| ros2_controller_node | library | 0.90 | ROS2 控制节点 |
| ros2_launch_file | library | 0.95 | ROS2 启动文件 |
| ros2_package_config | library | 0.95 | ROS2 包配置 |
| obstacle_avoidance | prompt | 0.60 | 避障 |
| custom_control_logic | prompt | 0.50 | 自定义逻辑 |
| force_control | prompt | 0.55 | 力控 |
| dual_arm_coordination | prompt | 0.40 | 双臂协调 |

**输出**：`GeneratedCode` 包含完整 ROS2 包结构

**生成的代码包结构**：

```
ros_ws/src/arm_controller/
├── __init__.py
├── arm_controller/
│   ├── __init__.py
│   ├── arm_controller_node.py     # ROS2 控制节点
│   ├── arm_kinematics.py          # FK/IK 运动学求解器
│   └── trajectory_planner.py      # 轨迹规划器
├── config/
│   ├── controllers.yaml           # ros2_control 控制器配置
│   └── kinematics.yaml            # 运动学参数配置
├── launch/
│   └── arm_controller_bringup.launch.py
├── package.xml
└── setup.py
```

**容错机制**：
- LLM 分解失败 → 降级到预置的默认分解
- LLM 代码生成失败 → 重新生成一次
- 每个生成的文件经过 ast.parse 语法校验

### 阶段 ④：仿真验证 (Simulation Verification)

**输入**：`GeneratedCode`

**处理流程**：
1. 根据臂型 DH 参数和动力学数据，从代码生成 URDF 和 MJCF 模型文件
2. 从生成代码中提取轨迹路径点（三层 fallback：AST waypoints → 代码执行 TrajectoryPlanner → S-curve 默认轨迹）
3. 使用 MuJoCo 执行轨迹仿真（position controller, kp=20, kv=0.5）
4. 14 项指标体系验证（含代码结构验证和 FK/IK 执行验证）
5. 不通过时自动修复循环（LLM 代码修改 + auto-tuning actuator 调优 + 交互式继续询问）

**14 项验证指标**：

| 指标 | 测量内容 | 阈值 | 单位 | 说明 |
|------|---------|------|------|------|
| joint_position_error | 关节位置平滑度 | ≤ 0.85 | ratio | 使用范围/可用半范围，检查突变 |
| joint_velocity_overshoot | 关节速度峰值 | ≤ 20.0 | ratio | 位置控制可超额定 20x |
| joint_acceleration_peak | 关节加速度峰值 | ≤ 40.0 | ratio | 位置控制加速度可超 40x |
| joint_torque_peak | 关节力矩峰值 | ≤ 0.95 | ratio | 不超过额定力矩 95% |
| self_collision_distance | 连杆最小间距 | ≥ 5.0 | mm | 跳过相邻链接碰撞 |
| joint_limit_margin | 关节限位裕度 | ≥ 0.05 | rad | 距离硬限位裕度 |
| path_jerk | 轨迹急动度 | ≤ 100.0 | ratio | 归一化急动度 |
| condition_number | 奇异位形接近度 | ≤ 1000.0 | - | Jacobian 条件数（P95，容忍 ≤5% 帧离群） |
| tcp_position_error | TCP 位置精度 | ≤ 100.0 | ratio | 步长一致性 |
| payload_ratio | 负载率 | ≤ 0.9 | ratio | 不超过额定 90% |
| code_syntax_valid | 代码语法 | = 0 | errors | 所有 .py 文件语法校验 |
| code_required_symbols | 必需符号 | = 0 | missing | 必需类/方法是否存在 |
| kinematics_fk | 正运动学 | passed | bool | 执行 FK 验证返回 4×4 矩阵 |
| kinematics_ik_roundtrip | 逆运动学往返 | < 0.001 | m | FK→IK→FK 往返位置误差 |

**自动修复循环**：
- 仿真不通过时进入修复循环
- 调用 LLM 分析失败根因并修改代码（传入失败指标和当前代码）
- 前 3 次自动修复；第 4 次起询问用户是否继续
- 用户可选择"继续修复"、"跳过到 Stage 5"、"停留在 Stage 4"
- LLM 修复无效时自动调整 actuator 参数（kp 渐进减小 20→12→7→5，kv 渐进增大 0.5→1.0→2.0→4.0）
- 每次修复后重新验证代码执行指标

**降级策略**：MuJoCo 未安装时自动切换到纯运动学仿真

**输出**：`SimulationReport` 含通过/失败、各项指标值、修复历史

### 阶段 ⑤：部署 (Deployment)

**输入**：`TechnicalApproach` + `GeneratedCode`

**处理流程**：
1. 用户选择目标品牌（默认 ur）
2. 从生成代码或技术方案中提取轨迹路径点（多层 fallback）
3. 调用品牌对应的 `BaseConverter` 实现类生成机器人程序
4. 生成部署指南文档
5. 生成安全操作检清单
6. 生成部署元数据 JSON

**支持品牌及文件格式**：

| 品牌 | 文件格式 | 路径规划指令 |
|------|---------|-------------|
| UR (Universal Robots) | `.script` | movej / movel / movec |
| KUKA | `.src` + `.dat` | PTP / LIN / CIRC |
| ABB | `.mod` (RAPID) | MoveAbsJ / MoveL / MoveC |

**输出**：`DeploymentPackage` 包含：

```
deploy/<brand>/
├── articulate_program.script     # 机器人程序
├── DEPLOYMENT_GUIDE.md           # 部署操作指南
├── SAFETY_CHECKLIST.md           # 安全检清单
└── deployment_metadata.json      # 部署元数据
```

## 混合决策路由

这是项目中最重要的设计模式。核心思路：**能用模板的走模板，不能用模板的让 LLM 生成**。

```
子任务名 + 上下文
       │
       ▼
┌──────────────┐
│  规则表匹配   │ ← 子串匹配（最长匹配优先）
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
  匹配    不匹配
   │       │
   │   ┌──▼────────┐
   │   │  LLM 分类  │ ← 需配置 LLM
   │   └──┬─────────┘
   │   ┌──┴──┐
   │   │     │
   │  成功   失败/无LLM
   │   │     │
   │   │  ┌──▼────────┐
   │   │  │  默认 prompt 路由 │
   │   │  └─────────────┘
   │   │
   ▼   ▼
┌──────────────┐
│  置信度检查   │ ← threshold=0.7
└──────┬───────┘
   ┌───┴───┐
   │       │
 ≥0.7    <0.7
   │       │
   │   ┌──▼────────┐
   │   │ 用户仲裁   │ → 选 library/prompt/cancel
   │   └────────────┘
   │
   ▼
执行路由（library/prompt）
```

规则匹配使用**子串匹配**（不区分大小写），匹配度以模式串长度为准——更具体的匹配优先。例如 `inverse_kinematics_7dof` 比 `inverse_kinematics` 更具体，优先匹配。

## Skill 门面模式

`ArticulateSkill` 是统一访问所有子模块的门面。流水线各阶段通过 `self.skill` 访问：

```python
self.skill.prompt_mgr.render(name, **context)   # 渲染提示词
self.skill.router.route(sub_task, context)       # 路由决策
self.skill.kinematics.fk(dh_params, angles)      # 正运动学
self.skill.kinematics.ik(dh_params, target)      # 逆运动学
self.skill.planning.plan_ptp(start, goal)        # PTP 规划
self.skill.ros2_gen.generate_package(approach)   # ROS2 代码生成
self.skill.urdf_loader.load(path)                # URDF 加载
self.skill.converters.get_converter(brand)       # 品牌转换器
```

## 提示词管理系统

`PromptManager` 管理 9 个提示词文件（6 个系统提示词 + 3 个上下文文档）：

**系统提示词**（用于 LLM 结构化输出）：

| 文件 | 用途 |
|------|------|
| requirement_analysis.yaml | 需求分析阶段 |
| technical_design.yaml | 技术方案设计 |
| code_generation.yaml | 代码生成 |
| deployment_planning.yaml | 部署规划 |
| failure_analysis.yaml | 仿真失败根因诊断 |
| risk_assessment.yaml | 风险评估 |

**上下文文档**（注入到代码生成提示词中作为背景知识）：

| 文件 | 用途 |
|------|------|
| ros2_humble_context.md | ROS2 Humble 编码规范 |
| dh_parameter_context.md | DH 参数标准说明 |
| safety_guidelines.md | 安全编码约束（硬约束 + 代码模式 + ROS2 安全规范） |

每个 YAML 模板支持 Jinja2 变量替换，渲染时传入上下文变量即可。

## 状态持久化

流水线状态序列化到项目目录的 `.articulate/state.json` 文件中。每次执行完一个阶段自动持久化。包含：

- 当前阶段索引（支持断点续行）
- 用户输入
- 各阶段输出（requirement_doc, technical_approach 等）
- 目标品牌

状态文件仅存储结构化数据，不存生成的代码内容（代码存于 `ros_ws/` 目录）。

\newpage

# 安装指南

## 环境要求

- **Python**: 3.10 或更高版本
- **操作系统**: Linux / macOS / Windows
- **可选依赖**:
  - MuJoCo 3.0+（仿真验证，不装则降级到纯运动学仿真）
  - Pinocchio（运动学加速，不装则使用纯 numpy 实现）
  - ROS2 Humble（编译运行生成的代码）

## 安装步骤

```bash
# 1. 克隆或解压项目
cd articulate

# 2. 安装项目及依赖
pip install -e .

# 3. （可选）安装仿真和运动学加速
pip install -e ".[robotics]"
```

## API Key 配置

Articulate 依赖 LLM API（支持 Anthropic Claude 或 OpenAI 兼容接口）。配置方式：

### 方式一：环境变量

```bash
# Linux / macOS
export ARTICULATE_ANTHROPIC_API_KEY=sk-...

# Windows (CMD)
set ARTICULATE_ANTHROPIC_API_KEY=sk-...

# Windows (PowerShell)
$env:ARTICULATE_ANTHROPIC_API_KEY="sk-..."
```

### 方式二：交互式输入

直接运行 `articulate generate`，未配置 API Key 时会提示输入。

### 方式三：配置文件

项目支持 `.env` 文件，在项目根目录创建 `.env`：

```
ARTICULATE_ANTHROPIC_API_KEY=sk-...
```

### 验证安装

```bash
# 查看所有可用命令
articulate --help

# 查看一键生成命令的帮助
articulate generate --help
```

\newpage

# 使用指南

## 一键生成（推荐）

对于首次使用或不熟悉分步流程的用户，`articulate generate` 命令自动完成全部 5 个阶段。

### 基本用法

```bash
articulate generate "目标描述" --brand 品牌
```

### 示例

```bash
# 从 A 点抓取放到 B 点，目标 UR 机器人
articulate generate "Pick and place from (0.3,0,0.2) to (0.6,0,0.3)"

# 用 KUKA 机器人执行简单搬运
articulate generate "从 (0.5,0,0.1) 搬运到 (0.8,0,0.4)，末端朝下" -b kuka

# 带速度和精度要求
articulate generate "抓取 (0.2,0.1,0.3) 处的零件，放置到 (0.5,-0.1,0.2)，速度 0.5m/s，精度 ±1mm"

# 指定 ABB 机器人
articulate generate "码垛任务：从传送带抓取到托盘 (0,0,0.5)，共 3 层" -b abb
```

### 执行流程

运行后自动经历：

```
1/5  Requirement Analysis   需求分析（从自然语言提取结构化需求）
    ↓
2/5  Technical Approach     方案设计（选择臂型、运动学策略、轨迹规划）
    ↓
3/5  Code Generation        代码生成（子任务分解→路由→生成→组装验证）
    ↓
4/5  Simulation Verification 仿真验证（MuJoCo 执行 + 14 项指标评估 + 自动修复）
    ↓
5/5  Deployment Package      部署包生成（机器人程序 + 指南 + 安全检清单）
```

每个阶段自动确认，无需人工干预。最后输出汇总结果。

### 验证失败的处理

如果阶段 ④ 仿真验证不通过，Articulate 会：

1. 自动诊断失败根因（LLM 分析是哪项指标超限）
2. 调用 LLM 修改代码或调整 actuator 参数
3. 重新仿真验证
4. 前 3 轮自动修复，之后询问用户是否继续
5. 验证失败时，用户可选择跳过到部署阶段

实际使用中，建议将仿真未通过的轨迹先用 10% 速度在真实机器人上测试。

### 指定项目目录

```bash
# 在特定目录下创建和执行项目
articulate generate "需求" --dir ./my_project
```

未指定 `--dir` 时，自动在当前目录创建项目文件夹（名称从需求文本派生）。

## 分步执行

对于需要逐阶段审查和干预的场景，也可以分步执行。

### 初始化项目

```bash
# 创建一个新项目
articulate init my_project
```

创建的项目结构：

```
my_project/
├── .articulate/          # 流水线状态和配置
│   └── state.json
├── ros_ws/src/           # ROS2 工作空间
├── deploy/               # 部署包输出
└── assets/               # 模型和资源文件
```

### 需求分析 + 技术方案

```bash
articulate plan "Pick and place from (0.3,0,0.2) to (0.6,0,0.3)"
```

执行阶段 1 和 2，交互式确认每一步。

### 代码生成

```bash
articulate codegen
```

执行阶段 3，基于已批准的技术方案生成 ROS2 代码。

### 仿真验证

```bash
articulate simulate
```

执行阶段 4，运行 MuJoCo 仿真并验证 14 项指标。

### 生成部署包

```bash
# 生成 UR 程序
articulate deploy --brand ur

# 生成 KUKA 程序
articulate deploy --brand kuka

# 生成 ABB 程序
articulate deploy --brand abb
```

## 查看结果

### 当前状态

```bash
articulate status
```

显示当前项目进度（第几个阶段）、项目路径、状态文件位置。

### 生成报告

```bash
# Markdown 格式
articulate report -f md

# HTML 格式（可浏览器打开查看）
articulate report -f html
```

### 项目目录结构

完整运行后的项目目录：

```
my_project/
├── .articulate/
│   └── state.json                    # 流水线状态
├── ros_ws/src/arm_controller/
│   ├── package.xml
│   ├── setup.py
│   ├── __init__.py
│   └── arm_controller/
│       ├── __init__.py
│       ├── arm_controller_node.py    # ROS2 控制节点
│       ├── arm_kinematics.py         # 运动学求解器
│       └── trajectory_planner.py     # 轨迹规划器
├── config/
│   ├── controllers.yaml              # 控制器配置
│   └── kinematics.yaml               # 运动学参数
├── launch/
│   └── arm_controller_bringup.launch.py
├── deploy/
│   ├── validation_report.md          # 仿真验证报告
│   ├── validation_report.html        # HTML 格式报告
│   └── ur/
│       ├── articulate_program.script # 机器人程序
│       ├── DEPLOYMENT_GUIDE.md       # 部署指南
│       ├── SAFETY_CHECKLIST.md       # 安全检清单
│       └── deployment_metadata.json  # 部署元数据
└── assets/
    ├── arm.urdf                      # URDF 模型
    └── arm.mjcf                      # MuJoCo 模型
```

\newpage

# 安全机制

Articulate 在四个层面保障生成代码的安全性：

## 第一层：提示词级安全约束

代码生成提示词中注入了 `safety_guidelines.md` 安全规范，包括：

**硬约束**：
- 所有轨迹必须保持在关节限位范围内
- 指令速度不得超过额定值的 90%
- 指令力矩不得超过额定值的 95%
- 路径必须包含奇异点检测和规避

**代码模式约束**：
- 所有运动指令需包含前置校验
- 控制节点必须实现超时/看门狗机制
- 每次运动前检查急停信号
- 路径预计算完成后再执行
- 所有数值运算使用 numpy 保证类型安全

**部署约束**：
- 首次运行必须以 10% 速度空载测试
- 工作空间边界必须在首次运动前验证
- 自动运动前必须测试急停
- 必须包含安全预检程序

## 第二层：仿真验证

MuJoCo 仿真环境下执行轨迹并验证 14 项指标：

- **位置安全**：关节位置限位检查、限位裕度、TCP 精度
- **速度安全**：关节速度峰值（< 额定 110%）
- **力矩安全**：关节力矩峰值（< 额定 95%）、负载率（< 90%）
- **轨迹质量**：加速度峰值、急动度（平滑性）
- **运动学安全**：奇异位形检测、自碰撞检测（> 5mm）
- **代码验证**：语法检查、必需符号检查、FK/IK 代码实际执行验证

任一项不通过时，自动进入修复循环（3 次自动 + 交互式继续）。

## 第三层：部署安全检清单

部署阶段随机器人程序输出 `SAFETY_CHECKLIST.md`，包含品牌特定的安全检查项：

**UR 检清单示例**：
- Verify TCP flange is properly mounted and payload is correctly set
- Confirm tool center point (TCP) is calibrated
- Verify safety zones and speed limits are configured
- Test emergency stop before first run
- Run at reduced speed (10%) for initial validation

**KUKA 检清单示例**：
- Verify base and tool calibration data
- Check software limit switches are enabled
- Verify payload data matches actual tooling
- Test ESTOP functionality
- First run at reduced override (< 10%)

**ABB 检清单示例**：
- Verify tool data and work object coordinates
- Check axis configuration and software limits
- Verify safety zones are correctly configured
- Test emergency stop and safety stops
- First automatic run at low speed (< 10%)

检清单包含操作员、主管、安全负责人签名栏，形成闭环管理。

## 第四层：代码级安全检查

- **语法检查**：所有生成 `.py` 文件通过 `ast.parse` 语法校验
- **危险模式检测**：扫描 `exec(`、`eval(`、`__import__(`、`subprocess.call`、`os.system` 等危险模式并告警
- **必需文件检查**：确保 `package.xml`、`__init__.py` 等关键文件存在

\newpage

# 技术详解

## LLM 客户端

Articulate 设计了统一的 LLM 抽象接口，支持多种模型后端。

### ClaudeClient

```python
class ClaudeClient:
    # 核心方法
    async def complete(system, messages, **kwargs) -> LLMResponse
    async def complete_structured(system, messages, output_model, **kwargs) -> T
```

**特性**：
- 指数退避重试：RateLimitError（限流）、5xx（服务器错误）、APIConnectionError（网络）自动重试
- 可配置重试次数和初始延迟：`llm_max_retries`（默认 3）、`llm_retry_base_delay`（默认 2.0s）
- 结构化输出：自动在 user message 末尾追加 JSON Schema 约束 → 解析 Pydantic 模型
- 解析失败重试：不符合 JSON 格式时，将失败信息和错误原因反馈给 LLM 重新生成（最多 3 次）

### DeepSeekClient

OpenAI 兼容接口封装，接口与 `ClaudeClient` 一致，可无缝切换。

```python
client = DeepSeekClient(api_key=key, model="deepseek-v4-flash")
# 使用方法与 ClaudeClient 完全相同
await client.complete_structured(system, messages, output_model=MyModel)
```

切换模型只需替换客户端实例，业务代码无需改动。

## 决策路由

`DecisionRouter` 采用双阶段路由策略：

### 第一阶段：规则匹配

```python
def _match_rule(self, sub_task: str) -> Optional[Rule]:
```

- 子串匹配（大小写不敏感）
- 最长匹配优先（更具体的规则优先）
- 14 条预置规则覆盖标准子任务

### 第二阶段：LLM 分类（规则不匹配时）

```python
async def _llm_classify(self, sub_task, context) -> RoutingResult
```

- LLM 判断走 library / prompt / library_with_prompt
- 输出置信度评分
- 置信度 < 0.7 时触发用户仲裁

### 仲裁机制

低置信度时用户介入选择：
- `library`：使用预置模板
- `prompt`：使用 LLM 生成
- `cancel`：取消该子任务

## 代码生成引擎

`CodeGenerationEngine` 是阶段 3 的核心，负责从技术方案到完整代码包的转换。

### 子任务分解

**LLM 分解**（默认）：
```python
response = await llm.complete_structured(
    system="You are a robotics code architect...",
    output_model=DecompositionSchema,
)
```

**降级策略**：LLM 失败时使用预置分解方案，包含 6 个标准子任务。

### 模板渲染

ROS2 代码使用 Jinja2 模板生成，模板包含：

| 模板文件 | 生成内容 |
|---------|---------|
| ros2_node.py.j2 | ROS2 Python 控制节点类 |
| launch.py.j2 | ROS2 launch 启动文件 |
| package_xml.j2 | package.xml 包配置 |
| controllers_yaml.j2 | ros2_control 控制器配置 |

模板变量通过 Python dict 注入，支持发布者/订阅者、参数配置、关节命名等自定义。

### 文件组装

合并所有子任务的产出，遵循：
- 后产出的文件覆盖先产出的同名文件（具体 > 通用）
- 确保必需文件存在（`__init__.py`、`package.xml`）

## 仿真验证指标

14 项指标继承自 `BaseMetric` 抽象基类，各自实现 `compute(data: SimulationData) -> float` 方法，通过统一 `evaluate()` 方法输出 `MetricResult(passed, value, threshold)`。

```python
class BaseMetric(ABC):
    name: str = ""
    unit: str = ""
    threshold: float = 0.0
    
    @abstractmethod
    def compute(self, data: SimulationData) -> float: ...
    
    def evaluate(self, data: SimulationData) -> MetricResult:
        value = self.compute(data)
        passed = value <= self.threshold if self.threshold > 0
                else value >= abs(self.threshold)
        return MetricResult(name=..., passed=passed, value=value, ...)
```

`SimulationData` 包含仿真全状态：时间序列、关节位置/速度/加速度/力矩、TCP 位置/姿态、自碰撞距离、条件数。

## 部署转换器

三种品牌转换器继承自 `BaseConverter`：

```python
class BaseConverter(ABC):
    @property
    def brand(self) -> str: ...
    def convert(self, trajectory, output_dir) -> Dict[str, str]: ...
    def generate_safety_checklist(self) -> List[str]: ...
    def generate_deployment_guide(self, trajectory) -> str: ...
```

通过 `ConverterFactory.get_converter(brand)` 工厂方法获取实例，新增品牌只需实现 `BaseConverter` 子类并注册到工厂。

\newpage

# 项目结构

## 目录说明

```
articulate/
├── articulate_core/                 # 核心代码
│   ├── cli/                         # CLI 界面层
│   │   ├── main.py                  # Typer 入口，8 个子命令
│   │   ├── console.py               # Rich 控制台配置
│   │   └── commands/                # 各命令实现
│   │       ├── init.py              # 项目初始化
│   │       ├── plan.py              # 阶段 1-2
│   │       ├── codegen.py           # 阶段 3
│   │       ├── simulate.py          # 阶段 4
│   │       ├── deploy.py            # 阶段 5
│   │       └── generate.py          # 一键生成
│   ├── config/
│   │   └── settings.py              # ArticulateConfig（Pydantic Settings）
│   ├── exceptions.py                # 异常层次结构
│   ├── llm/
│   │   └── client.py                # ClaudeClient + LLMResponse
│   ├── pipeline/                    # 流水线实现
│   │   ├── models.py                # 所有 DTO 定义（6 个阶段产出物 + StageContext）
│   │   ├── state.py                 # PipelineState 序列化
│   │   ├── orchestrator.py          # PipelineOrchestrator + BaseStage
│   │   ├── stage1_requirement.py    # 需求分析
│   │   ├── stage2_approach.py       # 技术方案
│   │   ├── codegen_engine.py        # 代码生成引擎
│   │   ├── stage3_generation.py     # 代码生成阶段编排
│   │   ├── stage4_simulation.py     # 仿真验证 + 自动修复
│   │   ├── deployment_manager.py    # 部署包生成
│   │   └── stage5_deployment.py     # 部署阶段编排
│   ├── simulation/                  # 仿真
│   │   ├── metrics.py               # 14 项验证指标
│   │   ├── validation_engine.py     # 验证引擎 + 报告生成
│   │   └── launch_mujoco.py         # MuJoCo 仿真执行器
│   ├── skill/                       # 领域知识门面
│   │   ├── __init__.py              # ArticulateSkill 门面类
│   │   ├── prompt_manager.py        # 提示词管理加载/缓存/渲染
│   │   ├── decision_router.py       # 混合路由引擎
│   │   ├── router_rules.yaml        # 14 条路由规则
│   │   ├── prompts/                 # 提示词资源
│   │   │   ├── system/              # 6 个系统提示词 YAML
│   │   │   ├── analysis/            # 2 个分析提示词 YAML
│   │   │   └── context/             # 3 个上下文文档 Markdown
│   │   ├── templates/               # Jinja2 模板
│   │   │   ├── ros2_node.py.j2
│   │   │   ├── launch.py.j2
│   │   │   ├── package_xml.j2
│   │   │   └── controllers_yaml.j2
│   │   ├── models/
│   │   │   ├── dh_template.py       # DH 参数 + ArmModel + FK
│   │   │   ├── preset_arms.py       # 3 种预置臂型定义
│   │   │   └── urdf_loader.py       # URDF 加载/MJCF 转换
│   │   ├── library/
│   │   │   ├── kinematics.py        # FK/IK（Pinocchio/numpy）
│   │   │   ├── planning.py          # 轨迹规划
│   │   │   ├── dynamics.py          # 动力学（Newton-Euler）
│   │   │   └── ros2_gen.py          # ROS2 代码生成器
│   │   └── converters/              # 品牌转换器
│   │       ├── base.py              # BaseConverter + Trajectory
│   │       ├── factory.py           # ConverterFactory
│   │       ├── ur_script.py         # UR (.script)
│   │       ├── krl.py               # KUKA (KRL)
│   │       └── rapid.py             # ABB (RAPID)
│   └── reporting/                   # 报告生成
│       ├── html_reporter.py
│       └── markdown_reporter.py
├── tests/                           # 测试
│   ├── unit/                        # 单元测试
│   ├── integration/                 # 集成测试
│   ├── e2e/                         # 端到端测试（需要 API key）
│   └── fixtures/                    # 测试夹具
│       ├── recorded_llm.py          # RecordedLLM（录制回放测试）
│       └── *.json                   # 预录响应数据（7 个场景）
├── run_deepseek.py                  # DeepSeek 完整流水线运行脚本
├── pyproject.toml                   # 项目配置 + entry point
└── .gitignore
```

## 核心模块关系图

```
CLI 命令 (main.py)
    │
    ├── init → 创建项目目录结构
    │
    ├── plan → PipelineOrchestrator → Stage 1 + 2
    │                                       ├── RequirementStage
    │                                       └── TechnicalApproachStage
    │
    ├── codegen → PipelineOrchestrator → Stage 3
    │                                       └── CodeGenerationStage
    │                                             └── CodeGenerationEngine
    │                                                   ├── decompose()
    │                                                   ├── generate_subtask()
    │                                                   ├── assemble()
    │                                                   └── validate()
    │
    ├── simulate → PipelineOrchestrator → Stage 4
    │                                       └── SimulationStage
    │                                             ├── MuJoCoSimulator
    │                                             ├── ValidationEngine (10 metrics)
    │                                             └── auto-repair loop
    │
    ├── deploy → PipelineOrchestrator → Stage 5
    │                                       └── DeploymentStage
    │                                             └── DeploymentManager
    │                                                   ├── ConverterFactory → UR/KUKA/ABB
    │                                                   ├── Safety Checklist
    │                                                   └── Deployment Guide
    │
    ├── status → PipelineState
    │
    └── report → HTMLReportGenerator / MarkdownReportGenerator
```

\newpage

# 已知限制

## 功能限制

1. **机械臂构型**：当前仅支持 6-DOF 串联臂 + 平行夹爪。SCARA、Delta、移动操作臂不在支持范围内。
2. **末端执行器**：仅支持平行夹爪，不支持吸盘、焊枪、喷涂枪等专用末端。
3. **ROS2 版本**：生成的代码基于 ROS2 Humble，不兼容 ROS1 或其他 ROS2 发行版。
4. **品牌转换器**：UR、KUKA、ABB 三种品牌。其他品牌需要自行实现 `BaseConverter` 子类。
5. **多机器人协同**：不支持单任务多机械臂协同工作。
6. **力控/阻抗控制**：当前仅支持位置控制模式，不支持力控。

## LLM 依赖限制

1. **API Key 必需**：所有阶段依赖 LLM 调用，无 API Key 则无法运行。
2. **输出不确定性**：LLM 生成的结构化输出可能包含格式错误或逻辑错误，依靠重试和降级机制缓解。
3. **质量与模型相关**：输出质量取决于所使用的 LLM 模型能力（Claude Sonnet / DeepSeek V4 等）。
4. **Token 消耗**：复杂任务可能需要大量 token，特别是代码生成和仿真诊断阶段。

## 仿真限制

1. **物理精度**：使用 MuJoCo 的仿真精度取决于模型参数。生成的简易 URDF/MJCF 可能无法完全反映真实机器人的物理特性。
2. **碰撞检测**：当前简化模型使用胶囊体几何表示连杆，碰撞检测精度有限。
3. **动力学模型**：力矩计算基于简化 Newton-Euler 算法，未考虑摩擦、柔性等非线性因素。
4. **未安装 MuJoCo**：降级到纯运动学仿真时，仅计算轨迹运动学，不做物理验证。

## 代码生成限制

1. **语法正确 ≠ 逻辑正确**：`ast.parse` 仅保证 Python 语法正确，不保证运动学/轨迹逻辑正确。
2. **ROS2 编译**：生成的 ROS2 代码包需用户自备 ROS2 环境编译运行。
3. **模板覆盖度**：Library 路径的模板覆盖标准场景，高度定制化的控制逻辑需要使用 Prompt 生成路径。

\newpage

# 常见问题

**Q: 生成的代码能在真实机器人上直接运行吗？**

A: 生成的机器人程序（如 UR .script）可以直接在对应品牌的机器人上运行，但建议：
1. 先检查 `SAFETY_CHECKLIST.md` 并逐项确认
2. 首次以 10% 速度空载运行验证轨迹
3. ROS2 控制代码需要 ROS2 Humble 环境编译

**Q: 如果仿真验证不通过怎么办？**

A: 系统会自动尝试修复（最多 3 轮）。
如果最终仍未通过，检查：
1. 需求是否合理（例如目标点是否在机械臂工作范围内）
2. 是否选择了合适的臂型
3. 可以降低速度或增加路径点来改善轨迹质量
验证报告保存在 `deploy/validation_report.html`，可查看具体哪项指标不合格。

**Q: 生成的代码 ROS2 包如何编译使用？**

A: 需要自备 ROS2 Humble 环境：
```bash
cp -r ros_ws/src /path/to/your/ros2_ws/
cd /path/to/your/ros2_ws
colcon build --packages-select arm_controller
source install/setup.bash
ros2 launch arm_controller arm_controller_bringup.launch.py
```

**Q: 如何更换 LLM 模型？**

A: 设置环境变量 `ARTICULATE_ANTHROPIC_MODEL` 指定模型名：
```bash
export ARTICULATE_ANTHROPIC_MODEL=claude-sonnet-4-20250514
```
更换 DeepSeek 等 OpenAI 兼容模型需要修改运行脚本中的客户端实现。

**Q: 如何添加新的机械臂品牌支持？**

A: 在 `skill/converters/` 下创建新文件，继承 `BaseConverter` 实现三个方法：
```python
class MyRobotConverter(BaseConverter):
    @property
    def brand(self) -> str: return "my_robot"
    def convert(self, trajectory, output_dir) -> Dict[str, str]: ...
    def generate_safety_checklist(self) -> List[str]: ...
```
然后在 `ConverterFactory` 中注册。

**Q: 如何添加预置臂型？**

A: 在 `skill/models/preset_arms.py` 中添加新的 `ArmModel` 实例，定义 DH 参数、关节限位和动力学参数，注册到 `PRESET_ARMS` 字典即可。

**Q: 每个步骤都要人工确认吗？**

A: 分步执行（`plan`/`codegen`/`simulate`/`deploy`）时需要。一键生成（`generate`）自动确认所有交互提示，无需人工参与。

\newpage

# 附录

## A. 配置项说明

`ArticulateConfig` 全部配置项（可通过环境变量或 `.env` 文件覆盖）：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| anthropic_api_key | `ARTICULATE_ANTHROPIC_API_KEY` | "" | LLM API Key |
| anthropic_model | `ARTICULATE_ANTHROPIC_MODEL` | claude-sonnet-4-20250514 | 模型名称 |
| llm_max_retries | `ARTICULATE_LLM_MAX_RETRIES` | 3 | LLM 调用最大重试次数 |
| llm_retry_base_delay | `ARTICULATE_LLM_RETRY_BASE_DELAY` | 2.0 | 重试初始延迟（秒） |
| confidence_threshold | `ARTICULATE_CONFIDENCE_THRESHOLD` | 0.7 | 路由置信度阈值 |
| project_dir | - | cwd | 项目目录 |
| sim_max_retries | `ARTICULATE_SIM_MAX_RETRIES` | 3 | 仿真修复最大次数 |
| sim_timeout | `ARTICULATE_SIM_TIMEOUT` | 120.0 | 仿真超时（秒） |

## B. 异常层次

```python
ArticulateError          # 所有异常的基类
├── LLMError             # LLM 调用/解析失败
├── ConfigError          # 配置错误
├── StageError           # 流水线阶段执行错误
├── RoutingError         # 路由决策错误
├── GenError             # 代码生成错误
├── SimError             # 仿真错误
├── ValidationError      # 验证错误
├── ConversionError      # 品牌转换错误
└── UserCancelledError   # 用户取消操作
```

## C. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1.0 | 2026-05 | 初始版本 |

## D. 第三方依赖

```
typer>=0.9.0         CLI 框架
anthropic>=0.30.0    Claude API SDK
pydantic>=2.0.0      数据模型/配置
pydantic-settings>=2.0.0  环境变量配置
pyyaml>=6.0          YAML 解析
jinja2>=3.0          模板引擎
numpy>=1.24          数值计算
rich>=13.0           终端 UI

# 可选（pip install -e ".[robotics]"）
pinocchio>=3.0       运动学加速
ompl>=1.6            运动规划
mujoco>=3.0          物理仿真

# 开发
pytest>=7.0
pytest-asyncio>=0.21
pytest-mock>=3.10
```
