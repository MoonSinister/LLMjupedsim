"""Prompt construction for LLM-based pedestrian route planning."""

from __future__ import annotations

import json

from agent_model import DEFAULT_AGENT_ROLES


SYSTEM_PROMPT = "你是行人疏散仿真的路径规划器。不要推理，只返回严格 JSON。"


def build_agent_routing_messages(agent_infos, region_infos, exit_infos):
    prompt = {
        "task": "为 JuPedSim 中每个 agent 独立规划高层行为路线。",
        "policy": [
            "必须为 agents 列表中的每一个 agent_id 返回且只返回一个计划。",
            "当前 agents 列表是一个批次，只规划本批次，不要猜测或生成批次外的 agent。",
            "每个计划只对对应 agent_id 生效，不允许用出生区域级别的统一计划代替。",
            "参考每个 agent 的 profile 生成 role、intent、activities、final_exit 和 desired_speed_mps。",
            "必须优先参考 profile.subtype、profile.preferences、profile.personal_variation 来体现个体差异。",
            "activities 是 0 到 3 个中间活动点；每个 activity 必须包含 region、action、wait_seconds。",
            "activity.region 只能使用 regions 列表中的 name；final_exit 只能使用该 agent 的 candidate_exits 中列出的出口名。",
            "wait_seconds 表示到达该中间地点后的停留时间，通勤者通常 0 到 10 秒，购物者/访客/工作人员可为 10 到 120 秒。",
            "角色应符合地图语义：通勤者更可能去火车站/停车场/出口，购物者更可能去商店/餐厅，工作人员更可能去服务通道/办公电梯。",
            "同一 role 的 agent 不应机械复制同一路线；如果 subtype 或 variation 不同，应在活动区域、停留时间、出口选择或路线长度上做合理变化。",
            "同一出生区域的 agent 可以有不同角色、不同意图和不同路径；路线应大致符合空间顺序，避免明显绕远。",
            "输出必须是 JSON，不要包含解释文本。",
        ],
        "roles": DEFAULT_AGENT_ROLES,
        "regions": region_infos,
        "exits": exit_infos,
        "agents": agent_infos,
        "output_schema": {
            "plans": [
                {
                    "agent_id": "agent_0001",
                    "role": "commuter|shopper|staff|visitor",
                    "subtype": "profile 中的 subtype",
                    "intent": "一句话说明该 agent 的意图",
                    "desired_speed_mps": 1.25,
                    "activities": [
                        {
                            "region": "region_1",
                            "action": "browse shops",
                            "wait_seconds": 45,
                        }
                    ],
                    "final_exit": "exit_0",
                }
            ]
        },
    }

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think\n" + json.dumps(prompt, ensure_ascii=False)},
    ]
