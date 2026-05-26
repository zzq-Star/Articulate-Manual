# Articulate

**机械臂代码生成 Agent CLI 工具** — 输入自然语言需求，自动完成需求分析、技术方案设计、ROS2 代码生成、MuJoCo 仿真验证、部署脚本生成全流程。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

```
User: "Pick and place from (0.3, 0, 0.2) to (0.6, 0, 0.3)"
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              5-Stage Automated Pipeline                  │
│                                                         │
│  ① Requirement Analysis  →  ② Technical Approach        │
│  ③ Code Generation      →  ④ Simulation Verification    │
│  ⑤ Deployment Package                                   │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
            UR .script / KUKA KRL / ABB RAPID
            ROS2 control code package
            Validation report + Safety checklist
```

## Quick Start

```bash
# Install
pip install -e .

# Set your API key
export ARTICULATE_ANTHROPIC_API_KEY=sk-...

# One-command generation (all 5 stages)
articulate generate "Pick and place from (0.3, 0, 0.2) to (0.6, 0, 0.3)"
可选参数：
参数	说明
-m, --model	LLM 模型名称（如 deepseek-chat、claude-sonnet-4-20250514），不指定则使用默认模型
-p, --provider	LLM 提供商（anthropic / deepseek / openai），不指定时从 --model 自动推断
--dir	项目输出目录，不指定则自动创建

# Or step by step
articulate init my_project
cd my_project
articulate plan "Pick and place from (0.3, 0, 0.2) to (0.6, 0, 0.3)"
articulate codegen
articulate simulate
articulate deploy --brand ur
```

## Pipeline

| Stage | Command | Output |
|-------|---------|--------|
| 1. Requirement Analysis | `articulate plan "..."` | Structured requirement document |
| 2. Technical Approach | `articulate plan "..."` | Arm selection, kinematics strategy, ROS2 architecture |
| 3. Code Generation | `articulate codegen` | ROS2 package (10 files) |
| 4. Simulation Verification | `articulate simulate` | MuJoCo validation report (10 metrics) |
| 5. Deployment | `articulate deploy -b ur/kuka/abb` | Robot program + safety checklist + guide |

## Features

- **Natural Language → Robot Code**: Describe the task in plain language, get deployable robot programs
- **Hybrid Decision Routing**: Standard sub-tasks use library templates (fast + reliable), complex logic uses LLM generation (flexible)
- **MuJoCo Simulation Verification**: 10 validation metrics (joint position/velocity/acceleration/torque, self-collision, workspace, jerk, singularity, TCP accuracy, payload)
- **Auto-Repair Loop**: Failed metrics trigger automatic trajectory scaling (3 attempts max)
- **Multi-Brand Deployment**: Universal Robots (.script), KUKA (KRL), ABB (RAPID)
- **Checkpoint Resume**: Pipeline state persisted to project directory, resume from any stage
- **Graceful Degradation**: MuJoCo optional — falls back to kinematic simulation
- **Multi-Model LLM**: Claude API (primary) + OpenAI-compatible (DeepSeek, etc.)

## Supported Brands

| Brand | Format | Motion Types |
|-------|--------|-------------|
| Universal Robots | `.script` | movej / movel / movec |
| KUKA | `.src` + `.dat` | PTP / LIN / CIRC |
| ABB | `.mod` (RAPID) | MoveAbsJ / MoveL / MoveC |

## Preset Arm Models

| Arm Type | DOF | Application |
|----------|-----|-------------|
| six_dof_standard | 6 | Pick & place, palletizing, welding |
| six_dof_collaborative | 6 | Assembly, precision operation |
| seven_dof_standard | 7 | Obstacle avoidance, confined spaces |

## Installation

```bash
# Core dependencies
pip install -e .

# Optional: simulation support
pip install -e ".[simulation]"

# Optional: robotics acceleration (Pinocchio)
pip install -e ".[robotics]"

# Optional: development
pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- LLM API key (Claude or OpenAI-compatible)

## Documentation

- [SPEC.md](SPEC.md) — Technical specification
- [docs/manual.md](docs/manual.md) — User manual (中文)

## Architecture

```
CLI (Typer) → PipelineOrchestrator → 5 Stages → ArticulateSkill (Facade)
                                              ├── PromptManager
                                              ├── DecisionRouter
                                              ├── KinematicsLibrary
                                              ├── PlanningLibrary
                                              ├── DynamicsLibrary
                                              ├── ROS2Generator
                                              ├── URDFLoader
                                              └── ConverterFactory (UR/KUKA/ABB)
```

## Project Structure

```
articulate_core/
├── cli/              # CLI commands (Typer)
├── config/           # Configuration (pydantic-settings)
├── llm/              # LLM client (Claude + OpenAI-compatible)
├── pipeline/         # 5-stage pipeline orchestration
├── simulation/       # MuJoCo simulation + 10 validation metrics
├── skill/            # Domain logic (router, kinematics, converters, templates)
└── reporting/        # HTML/Markdown report generators
```

## License

MIT
