# Articulate — 机械臂代码生成 Agent 技术规范

> **版本**: v0.1
> **状态**: Implemented
> **最后更新**: 2026-05-23

---

## 1. 项目概述

### 1.1 定位

Articulate 是一个面向**产线工程师/集成商**的机械臂代码生成 Agent。用户以自然语言描述需求，Articulate 全流程自动完成**方案设计 → 代码生成 → 仿真验证 → 部署准备**，最终输出可在真实机械臂上运行的代码和部署包。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| **全栈代码生成** | 从运动规划到 ROS2 节点框架，含正逆解、轨迹插补、任务编排 |
| **混合智能** | 标准操作（FK/IK/标准轨迹）→ 库调用；定制需求 → 提示词生成 |
| **仿真验证** | MuJoCo 完整运动学 + 动力学验证，含力矩/碰撞/平滑度检查 |
| **多臂适配** | 参数化 DH 模板 + URDF 自动解析，支持多种机械臂形态 |
| **逐级确认** | 关键阶段暂停展示结果，用户确认后继续 |
| **脚本部署** | 生成目标品牌格式的部署脚本（UR .script、KRL、RAPID） |

### 1.3 目标用户画像

- **角色**: 产线工程师、机器人集成商、自动化工程师
- **技术背景**: 熟悉工艺流程，了解机器人基本概念，但不一定精通运动学算法或 ROS2
- **核心需求**: 快速将工艺需求转化为可运行的机器人程序，减少手写代码和调试时间

### 1.4 非目标

- ⛔ 不生成电机驱动/固件级代码（PID 调节、FOC、电流环）
- ⛔ 不直接与机械臂控制器建立实时通信（通过生成的脚本间接部署）
- ⛔ 不替代成熟的工业离线编程软件（RoboDK、RoboMaster 等）的全部功能

---

## 2. 系统架构

### 2.1 整体架构

```
┌────────────────────────────────────────────────────────┐
│                     User (CLI)                         │
└──────────────────┬─────────────────────────────────────┘
                   │
┌──────────────────▼─────────────────────────────────────┐
│                  articulate CLI                         │
│  ┌────────────┬────────────┬────────────┬────────────┐  │
│  │ articulate │ articulate │ articulate │ articulate  │  │
│  │   plan     │  codegen   │  simulate  │   deploy   │  │
│  └─────┬──────┴─────┬──────┴──────┬─────┴──────┬─────┘  │
│        │            │             │            │        │
│  ┌─────▼────────────▼─────────────▼────────────▼─────┐  │
│  │               Pipeline Orchestrator               │  │
│  │  Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          │                              │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │            Skill 核心 (Domain Logic)              │  │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────────┐  │  │
│  │  │ Dec.Router │ │ Library  │ │ Prompt Manager │  │  │
│  │  └────────────┘ └──────────┘ └────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│            MuJoCo 仿真 + 验证引擎                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ MuJoCo   │  │ Metric   │  │ Validation       │  │
│  │ Physics  │  │ Engine   │  │ Reporter         │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| **CLI 框架** | Typer + Rich | 命令路由和参数解析，彩色终端输出 |
| **LLM 接口** | Claude API + OpenAI 兼容接口 | 核心智能引擎，支持多模型后端 |
| **数据模型** | Pydantic + dataclass | 配置管理 + 结构化数据验证 |
| **仿真器** | MuJoCo 3.0+ | 物理引擎，全链路仿真（可选，有 numpy 降级） |
| **机器人库** | Pinocchio（可选）/ numpy | 运动学计算，Pinocchio 优先 + numpy fallback |
| **模板引擎** | Jinja2 | 代码模板渲染 |
| **ROS2** | Humble Hawksbill (LTS) | 代码生成目标框架（不要求编译运行） |
| **模型格式** | URDF + MJCF | 双模型分别用于展示和仿真 |
| **报告** | Markdown / HTML | 验证结果和部署指南 |

### 2.3 CLI 命令设计

```
articulate init <project-name>          # 初始化项目目录结构
articulate plan "<requirement>"         # Stage 1-2: 需求+方案
articulate codegen                      # Stage 3: 代码生成
articulate simulate                     # Stage 4: 仿真验证
articulate deploy [--brand ur/kuka/abb] # Stage 5: 部署准备
articulate generate "<requirement>"     # 一键执行全部5个阶段
articulate status                       # 查看当前项目状态
articulate report                       # 生成全流程报告
```

---

## 3. 五阶段 Pipeline 详细设计

### 3.1 Stage 1: 需求理解 (Requirement Analysis)

**输入**: 用户自然语言描述
**输出**: 结构化需求文档 (RequirementDocument)

**流程**:
1. 用户输入需求（如 "Pick and place from (0.3, 0, 0.2) to (0.6, 0, 0.3)"）
2. 调用 LLM 结构化输出接口，分析需求并提取：
   - 任务类型（pick_and_place / welding / spraying / palletizing / assembly / custom）
   - 关键路径点和姿态
   - 末端执行器类型
   - 速度和精度要求
   - 环境和障碍物信息
   - 缺失信息列表
3. 检测到信息不完整时提示用户补充
4. 生成结构化需求文档
5. **暂停 → 用户确认**：需求是否准确完整

**异常处理**:
- LLM 结构化解析失败 → 重试（最多 3 次），附带错误信息指导模型修正输出格式

### 3.2 Stage 2: 技术方案 (Technical Approach)

**输入**: 结构化需求文档
**输出**: 技术方案文档 (TechnicalApproach)

**流程**:
1. 从预置臂型库中自动选择匹配的臂型（基于任务类型和 DOF 需求）
2. 调用 LLM 设计技术方案：
   - 运动学策略（解析法/数值法/混合法）
   - 轨迹类型组合（PTP/LIN/CIRC/SPLINE）
   - ROS2 节点架构设计
   - 风险评估（低/中/高 + 缓解措施）
3. 风险等级为高时提示用户注意
4. **暂停 → 用户确认**：技术方案是否合理

### 3.3 Stage 3: 代码生成 (Code Generation)

**输入**: 技术方案文档 + 机械臂模型参数
**输出**: 完整 ROS2 包结构 (GeneratedCode)

#### 3.3.1 CodeGenerationEngine

核心引擎 `CodeGenerationEngine` 负责将技术方案转换为代码：

1. **子任务分解**：将技术方案拆解为独立子任务
   - LLM 分解优先 → 失败时降级到预置 6 子任务默认分解
   - 典型子任务：运动学求解器、轨迹规划器、ROS2 控制节点、启动文件、包配置
2. **路由决策**：对每个子任务调用 `DecisionRouter` 确定生成路径
3. **代码生成**：Library 路径走 Jinja2 模板渲染，Prompt 路径调用 LLM 生成
4. **组装合并**：所有子任务产出合并为完整包结构（后产出覆盖先产出同名文件）
5. **验证**：ast.parse 语法检查 + 必需文件检查 + 危险模式扫描
6. **子任务降级**：LLM 代码生成或库模板生成失败 → 自动尝试对换路由；均失败时对已知模式生成最小可用桩代码（control_logic、obstacle_avoidance、ros2_control_node）

#### 3.3.2 决策路由 (Decision Router)

路由分类逻辑：

| 子任务模式 | 路由 | 置信度 |
|-----------|------|--------|
| forward_kinematics | library | 0.95 |
| inverse_kinematics | library | 0.90 |
| inverse_kinematics_7dof | library_with_prompt | 0.70 |
| trajectory_planner | library | 0.90 |
| trajectory_ptp | library | 0.95 |
| trajectory_linear | library | 0.90 |
| trajectory_circular | library | 0.85 |
| ros2_controller_node | library | 0.90 |
| ros2_launch_file | library | 0.95 |
| ros2_package_config | library | 0.95 |
| obstacle_avoidance | prompt | 0.60 |
| custom_control_logic | prompt | 0.50 |
| force_control | prompt | 0.55 |
| dual_arm_coordination | prompt | 0.40 |

路由使用**子串匹配**（大小写不敏感），最长模式优先匹配。置信度 < 0.7 时触发**用户仲裁**。

#### 3.3.3 代码模板

以下 Jinja2 模板用于 Library 路径的代码生成：

| 模板文件 | 生成内容 |
|---------|---------|
| ros2_node.py.j2 | ROS2 Python 控制节点（发布者/订阅者/定时器） |
| launch.py.j2 | ROS2 launch 启动文件 |
| package_xml.j2 | package.xml 包配置 |
| controllers_yaml.j2 | ros2_control 控制器配置（含关节限位） |
| kinematics_yaml.j2 | 运动学参数配置文件 |

#### 3.3.4 生成产物

```
<project>/ros_ws/src/arm_controller/
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

### 3.4 Stage 4: 仿真验证 (Simulation Verification)

**输入**: GeneratedCode + 臂型参数
**输出**: 验证报告 + 通过/不通过判定

#### 3.4.1 仿真准备

1. 根据臂型 DH 参数和动力学数据，从代码生成 URDF 和 MJCF 模型文件
2. 从生成代码中提取轨迹路径点（三层 fallback：Phase 1 AST waypoints 提取 → Phase 1.5 实际执行 TrajectoryPlanner 代码 → Phase 3 S-curve 默认轨迹）
3. 初始化 MuJoCo 仿真环境（MuJoCo 未安装时降级到纯运动学仿真）

#### 3.4.2 仿真执行

1. 将轨迹路径点转换为五次 S-curve 平滑轨迹（5s 持续，0.2 rad 范围）
2. 使用 position controller 执行轨迹（kp=50, kv=50）
3. 逐帧采集状态数据：关节位置/速度/加速度/力矩、TCP 位姿、自碰撞距离、条件数
4. 每步执行多个 mj_step 子步以匹配轨迹时间步长

#### 3.4.3 验证指标体系

| 检查项 | 指标名 | 阈值 | 单位 | 说明 |
|--------|--------|------|------|------|
| 关节位置 | joint_position_error | ≤ 0.85 | ratio | 关节使用范围/可用半范围，无突变 |
| 关节速度 | joint_velocity_overshoot | ≤ 20.0 | ratio | 位置控制可超额定 20x，宽松阈值 |
| 关节加速度 | joint_acceleration_peak | ≤ 40.0 | ratio | 位置控制下加速度可超 40x |
| 关节力矩 | joint_torque_peak | ≤ 0.95 | ratio | 不超过额定力矩的 95% |
| 自碰撞 | self_collision_distance | ≥ 5.0 | mm | 连杆最小间距，跳过相邻链接 |
| 工作空间 | joint_limit_margin | ≥ 0.05 | rad | 距离关节硬限位的裕度 |
| 路径平滑度 | path_jerk | ≤ 100.0 | ratio | 归一化急动度（max_jerk / typical_jerk） |
| 奇异点 | condition_number | ≤ 5000.0 | - | Jacobian 条件数（平移雅可比 3×6） |
| 末端精度 | tcp_position_error | ≤ 100.0 | ratio | TCP 步长一致性（max_dev / mean_step） |
| 负载 | payload_ratio | ≤ 0.9 | ratio | 最大力矩 / 额定力矩 |
| 代码语法 | code_syntax_valid | = 0 | errors | 所有 .py 文件 ast.parse 语法检查 |
| 代码符号 | code_required_symbols | = 0 | missing | 必需类/方法是否存在（TrajectoryPlanner、ArmKinematics 等） |
| 正运动学 | kinematics_fk | passed | bool | 实际导入执行 ArmKinematics.forward()，验证返回 4×4 矩阵 |
| 逆运动学 | kinematics_ik_roundtrip | < 0.001 | m | 执行 FK→IK→FK 往返误差，验证 IK 收敛精度 |

**阈值约定**: `threshold > 0` → 值 ≤ 阈值判定通过（上限检查）；`threshold ≤ 0` → 值 ≥ |阈值| 判定通过（下限检查，如自碰撞和限位裕度）。

#### 3.4.4 自动修复循环

1. 任一项指标未通过 → 进入修复循环
2. 调用 LLM 分析失败根因，生成代码修改（失败指标、代码内容传入提示词）
3. LLM 返回修改后的文件 + 执行器参数调整建议（kp/kv/forcerange_scale）
4. 重新生成 MJCF 模型 + 重新提取轨迹 + 重新仿真验证
5. 代码执行指标也在修复后重新验证（确保修复不破坏代码正确性）
6. 前 3 次为自动修复；第 4 次起询问用户是否继续修复
7. 用户可选择"继续修复"、"跳过到 Stage 5"、"停留在 Stage 4"
8. LLM 修复无效时，自动执行 actuator 参数调优（kp 渐进减小 20→12→7→5，kv 渐进增大 0.5→1.0→2.0→4.0）

#### 3.4.5 报告生成

验证引擎自动生成：

- **Markdown 报告**: 指标通过/失败状态表 + 汇总 + 修复建议 + 失败详情
- **HTML 报告**: 彩色编码指标状态（绿/红），失败项高亮显示

#### 3.4.6 降级策略

MuJoCo 未安装时自动降级到**纯运动学仿真**：
- 从轨迹命令通过数值微分计算速度和加速度
- TCP 位置简化为前 3 关节累积和
- 力矩估算为 `|vel| × 5 + |acc| × 2`
- 自碰撞距离和条件数使用默认值

代码执行验证（`_execute_trajectory_planner`、`_validate_kinematics`）不依赖 MuJoCo，降级后仍然执行。

### 3.5 Stage 5: 部署执行 (Deployment)

**输入**: 验证通过的代码 + 目标机械臂品牌
**输出**: 部署脚本包 + 部署指南 + 安全检清单

**流程**:
1. 用户指定目标品牌（ur / kuka / abb，默认 ur）
2. 从技术方案或生成代码中提取轨迹路径点（多层 fallback）
3. 调用品牌对应的 `BaseConverter` 将轨迹转换为目标格式：
   - UR: `.script` (URScript) — movej/movel/movec
   - KUKA: `.src` + `.dat` (KRL) — PTP/LIN/CIRC
   - ABB: `.mod` (RAPID) — MoveAbsJ/MoveL/MoveC
4. 生成部署指南（参数说明、安装步骤、安全注意事项）
5. 生成安全检清单（品牌特定检查项，含签名栏）
6. **暂停 → 用户确认**: 确认部署包完整

**输出目录结构**:
```
deploy/<brand>/
├── articulate_program.script     # 机器人程序
├── DEPLOYMENT_GUIDE.md           # 部署操作指南
├── SAFETY_CHECKLIST.md           # 安全检清单
└── deployment_metadata.json      # 部署元数据
```

---

## 4. Skill 封装设计

### 4.1 定位

`articulate_core.skill` 是 Articulate 的领域知识核心，封装了所有与机械臂编程相关的专业逻辑。它通过 `ArticulateSkill` 门面类统一对外暴露接口。

### 4.2 Skill 模块结构

```
articulate_core/skill/
├── __init__.py                       # ArticulateSkill 门面类
├── decision_router.py                # 混合决策路由（规则匹配 + LLM 分类）
├── router_rules.yaml                 # 14 条路由规则配置
├── prompt_manager.py                 # 提示词加载/缓存/渲染（Jinja2）
│
├── library/                          # ── 库调用模块 ──
│   ├── __init__.py
│   ├── kinematics.py                 # FK/IK（Pinocchio 优先 + numpy fallback）
│   ├── planning.py                   # 轨迹规划（PTP/LIN/CIRC）
│   ├── ros2_gen.py                   # ROS2 代码模板渲染引擎
│   └── dynamics.py                   # 动力学计算（Newton-Euler，负载校验）
│
├── prompts/                          # ── 提示词模板 ──
│   ├── system/                       # 系统级提示词
│   │   ├── requirement_analysis.yaml
│   │   ├── technical_design.yaml
│   │   ├── code_generation.yaml
│   │   └── deployment_planning.yaml
│   ├── analysis/                     # 分析专用提示词
│   │   ├── failure_analysis.yaml     # 仿真失败根因分析
│   │   └── risk_assessment.yaml      # 风险评估
│   └── context/                      # 上下文注入模板
│       ├── ros2_humble_context.md    # ROS2 Humble 规范
│       ├── dh_parameter_context.md   # DH 参数说明
│       └── safety_guidelines.md      # 安全编码规范
│
├── templates/                        # ── Jinja2 代码模板 ──
│   ├── ros2_node.py.j2
│   ├── launch.py.j2
│   ├── package_xml.j2
│   ├── controllers_yaml.j2
│   └── kinematics_yaml.j2
│
├── converters/                       # ── 部署转换器 ──
│   ├── __init__.py
│   ├── base.py                       # BaseConverter 抽象基类 + Trajectory
│   ├── factory.py                    # ConverterFactory
│   ├── ur_script.py                  # UR (.script)
│   ├── krl.py                        # KUKA (KRL)
│   └── rapid.py                      # ABB (RAPID)
│
└── models/                           # ── 机械臂模型库 ──
    ├── __init__.py
    ├── dh_template.py                # DHParameter, JointLimit, ArmModel, FK
    ├── preset_arms.py                # 3 种预置臂型定义
    └── urdf_loader.py                # URDF 加载 / MJCF 生成
```

### 4.3 门面接口

```python
class ArticulateSkill:
    self.prompt_mgr       # PromptManager — 渲染提示词
    self.router           # DecisionRouter — 路由决策
    self.kinematics       # KinematicsLibrary — FK/IK
    self.planning         # PlanningLibrary — 轨迹规划
    self.dynamics         # DynamicsLibrary — 动力学计算
    self.ros2_gen         # ROS2Generator — 代码生成
    self.urdf_loader      # URDFLoader — 模型加载
    self.converters       # ConverterFactory — 品牌转换器
```

---

## 5. 机械臂模型管理

### 5.1 参数化模板

DH 参数模板是适配不同机械臂的核心机制，定义在 `preset_arms.py` 中。每种臂型包含：
- DH 参数（a, alpha, d, theta）
- 关节限位（lower, upper, velocity, torque）
- 动力学参数（mass, inertia, friction, damping）

### 5.2 预置臂型

| 臂型 | DOF | 类型 | 适用场景 |
|------|-----|------|---------|
| six_dof_standard | 6 | 标准工业臂 | 搬运、码垛、焊接 |
| six_dof_collaborative | 6 | 协作臂 | 装配、精密操作 |
| seven_dof_standard | 7 | 冗余自由度臂 | 避障、狭窄空间 |

### 5.3 URDF/MJCF 生成

仿真阶段自动从臂型参数生成 URDF 和 MJCF 模型：
- **URDF**: 用于 ROS2 代码展示，包含 `<robot>` → `<link>` → `<joint>` 层次结构
- **MJCF**: 用于 MuJoCo 仿真，包含 `<mujoco>` → `<worldbody>` → `<body>` → `<geom>`
  - 胶囊体几何表示连杆，球体表示关节
  - position actuator（forcerange 设力矩限）驱动关节
  - 力矩传感器（jtorque）采集关节力矩
  - 碰撞检测 geoms 独立于视觉 geoms

---

## 6. 安全设计

### 6.1 四层安全护栏

```
Layer 1: 提示词级安全约束
 ├── 关节限位、速度上限、力矩上限写入代码
 ├── 奇异点规避代码自动插入
 └── 安全编码规范注入生成提示词

Layer 2: 仿真验证安全
 ├── 14 项指标体系验证（见 3.4.3）
 ├── 自碰撞检测（非相邻连杆间距 > 5mm）
 └── 超时保护（仿真超时自动终止）

Layer 3: 部署安全
 ├── 部署指南含详细安全警告
 ├── 强制要求空载首轮测试（10% 速度）
 └── 品牌特定的安全检清单（含签名栏）

Layer 4: 代码级安全检查
 ├── ast.parse 语法校验
 ├── 危险模式检测（exec/eval/subprocess 等）
 └── 必需文件检查
```

### 6.2 人类确认点

Pipeline 中预设 5 个确认点（一键生成模式自动确认）：

1. **需求确认** → 用户确认 Agent 对需求的理解准确
2. **方案确认** → 用户确认技术方案可行
3. **代码确认** → 用户审查生成代码
4. **仿真结果确认** → 用户确认验证报告可接受
5. **部署确认** → 用户确认准备部署到真实硬件

### 6.3 错误处理策略

| 错误类型 | 处理策略 | 最终失败行为 |
|----------|---------|------------|
| LLM 调用失败 | 重试 3 次，指数退避 | 终止并提示网络问题 |
| 代码生成失败 | 路径切换（library ↔ prompt） | 终止 + 诊断报告 |
| 结构化解析失败 | 重试 3 次，反馈错误修正输出 | 终止 + 诊断报告 |
| 仿真失败 | 自动修复最多 3 次 | 标记验证失败但不阻塞部署 |
| 路由判断失误 | 用户仲裁兜底 | — |
| 用户取消 | 保存当前状态 | 输出部分产物 |

---

## 7. 项目目录结构

```
Articulate/
├── SPEC.md                           # 本文档
├── README.md                         # 项目介绍
├── pyproject.toml                    # Python 包配置 + entry point
├── .gitignore
│
├── articulate_core/                  # 核心 Python 包
│   ├── __init__.py
│   │
│   ├── cli/                          # CLI 入口
│   │   ├── main.py                   # Typer app，8 个子命令
│   │   ├── console.py                # Rich 控制台配置
│   │   └── commands/
│   │       ├── init.py               # 项目初始化
│   │       ├── plan.py               # 阶段 1-2
│   │       ├── codegen.py            # 阶段 3
│   │       ├── simulate.py           # 阶段 4
│   │       ├── deploy.py             # 阶段 5
│   │       └── generate.py           # 一键生成
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py               # ArticulateConfig (pydantic-settings)
│   │
│   ├── exceptions.py                 # 异常层次结构
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                 # ClaudeClient + LLMResponse
│   │
│   ├── pipeline/                     # Pipeline 编排与阶段实现
│   │   ├── __init__.py
│   │   ├── models.py                 # 所有 DTO 定义
│   │   ├── state.py                  # PipelineState 序列化
│   │   ├── orchestrator.py           # PipelineOrchestrator + BaseStage
│   │   ├── stage1_requirement.py     # 需求分析
│   │   ├── stage2_approach.py        # 技术方案
│   │   ├── codegen_engine.py         # 代码生成引擎
│   │   ├── stage3_generation.py      # 代码生成编排
│   │   ├── stage4_simulation.py      # 仿真验证 + 自动修复
│   │   ├── deployment_manager.py     # 部署包生成
│   │   └── stage5_deployment.py      # 部署阶段编排
│   │
│   ├── simulation/                   # 仿真验证
│   │   ├── __init__.py
│   │   ├── launch_mujoco.py          # MuJoCo 仿真执行器
│   │   ├── metrics.py                # 14 项验证指标
│   │   └── validation_engine.py      # 验证引擎 + 报告生成
│   │
│   ├── skill/                        # Skill 门面 (领域知识)
│   │   └── ...                       # 见第 4 节
│   │
│   └── reporting/                    # 报告生成
│       ├── __init__.py
│       ├── html_reporter.py
│       └── markdown_reporter.py
│
├── docs/
│   └── manual.md                     # 使用说明书
│
└── tests/
    ├── __init__.py
    ├── test_config.py                # 配置测试
    ├── test_llm_client.py            # LLM 客户端测试
    ├── test_pipeline_models.py       # 数据模型测试
    ├── unit/                         # 单元测试
    │   ├── test_converters.py
    │   ├── test_deployment_manager.py
    │   ├── test_mujoco_simulator.py
    │   ├── test_simulation_metrics.py
    │   ├── test_simulation_stage.py
    │   ├── test_validation_engine.py
    │   └── test_with_recorded_llm.py
    ├── integration/
    │   └── test_pipeline_stages.py   # 流水线集成测试
    ├── e2e/
    │   └── test_real_llm.py          # 端到端测试（需 API key）
    └── fixtures/                     # 测试夹具（录制的 LLM 响应）
        ├── recorded_llm.py
        ├── requirement_analysis.json
        ├── technical_design.json
        ├── code_generation.json
        ├── failure_analysis.json
        ├── risk_assessment.json
        ├── route_response.json
        └── subtask_decomposition.json
```

---

## 8. 技术决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 代码栈 | Python + ROS2 Humble | 生态成熟，产线工程师熟悉度高 |
| 仿真器 | MuJoCo | 轻量快速，Python API 友好，适合 CI 集成，有 numpy 降级路径 |
| 运动学库 | Pinocchio（可选）+ numpy fallback | Pinocchio 加速 FK/IK，numpy 保证无依赖可用 |
| 规划器 | 内置 numpy 实现（无 OMPL 依赖） | 标准 PTP/LIN/CIRC 轨迹 numpy 足够，降低依赖复杂度 |
| 模型格式 | URDF + MJCF 双模型 | URDF 用于 ROS2，MJCF 用于 MuJoCo，各自最优 |
| 代码模板 | Jinja2 | 轻量灵活，Python 原生，条件渲染能力强 |
| 部署接口 | 脚本格式转换 | 无需控制器运行 ROS2，兼容性最广 |
| LLM 接口 | Claude API + OpenAI 兼容 | Claude 主推，DeepSeek 等兼容接口可替换 |
| 容器化 | 宿主机优先 | MVP 阶段降低环境复杂度 |
| 自动修复 | 轨迹幅度递减 | 简单可靠，无需 LLM 诊断依赖 |

---

## 9. 实现状态

### Phase 1: Foundation ✅
- [x] CLI 框架搭建（命令路由、参数解析）
- [x] Pipeline 编排骨架（5 阶段流程定义、状态管理）
- [x] 配置管理系统 (pydantic-settings)
- [x] Claude API 集成封装（含重试和结构化输出）

### Phase 2: Core Skill ✅
- [x] 决策路由模块（14 条规则 + LLM 分类）
- [x] 运动学库封装（Pinocchio FK/IK + numpy fallback）
- [x] 提示词模板（需求分析 + 方案设计 + 代码生成 + 部署规划 + 根因分析 + 风险评估）
- [x] 代码模板（ROS2 节点/启动文件/包配置/Jinja2）

### Phase 3: Code Generation ✅
- [x] CodeGenerationEngine（子任务分解 → 路由 → 生成 → 组装 → 验证）
- [x] 子任务分解（LLM 优先 + 默认分解降级）
- [x] 代码生成（Library 模板路径 + Prompt 生成路径）
- [x] ast.parse 语法验证 + 危险模式检测

### Phase 4: Simulation ✅
- [x] MuJoCo 集成（模型加载、位置控制、轨迹执行）
- [x] 双模型生成（URDF + MJCF 自动生成）
- [x] 14 项指标体系（关节位置/速度/加速度/力矩、自碰撞、工作空间、平滑度、奇异点、TCP 精度、负载、代码语法/符号检查、FK/IK 执行验证）
- [x] 自动修复循环（LLM 代码修改 + auto-tuning actuator 调优 + 交互式继续询问）
- [x] 代码执行验证 Phase 1.5（实际导入运行 TrajectoryPlanner 代码）
- [x] 纯运动学降级（MuJoCo 未安装时 numpy 回退）

### Phase 5: Deployment ✅
- [x] UR .script 转换器
- [x] KUKA KRL 转换器
- [x] ABB RAPID 转换器
- [x] 部署包生成 + 部署指南 + 安全检清单
- [x] 报告生成器（HTML/Markdown）
- [x] 一键生成命令 `articulate generate`
- [x] 端到端测试（录制回放 + 真实 API）

---

## 附录

### A. 配置项说明

`ArticulateConfig` 全部配置项：

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

### B. 异常层次

```
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

### C. 第三方依赖

```
# 核心
typer>=0.9.0            CLI 框架
anthropic>=0.30.0       Claude API SDK
pydantic>=2.0.0         数据模型/配置
pydantic-settings>=2.0.0  环境变量配置
pyyaml>=6.0             YAML 解析
jinja2>=3.0             模板引擎
numpy>=1.24             数值计算
rich>=13.0              终端 UI

# 可选（pip install -e ".[robotics]"）
pinocchio>=3.0          运动学加速

# 可选（pip install -e ".[simulation]"）
mujoco>=3.0             物理仿真

# 开发（pip install -e ".[dev]"）
pytest>=7.0
pytest-asyncio>=0.21
pytest-mock>=3.10
```

### D. 术语表

| 术语 | 含义 |
|------|------|
| DH 参数 | Denavit-Hartenberg 参数，机械臂运动学建模标准方法 |
| FK | Forward Kinematics，正运动学 |
| IK | Inverse Kinematics，逆运动学 |
| PTP | Point-to-Point，点到点运动 |
| LIN | Linear，直线运动 |
| CIRC | Circular，圆弧运动 |
| TCP | Tool Center Point，工具中心点 |
| MJCF | MuJoCo XML 模型格式 |
| URDF | Unified Robot Description Format，ROS 标准模型格式 |
| ros2_control | ROS2 控制框架 |
| S-curve | 五次多项式 S 形速度曲线，平滑加减速 |
