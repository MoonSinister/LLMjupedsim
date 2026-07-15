"""Prompt construction for LLM-based pedestrian route planning."""

from __future__ import annotations

import json
import hashlib


PROMPT_VERSION = "1.0"

from jupedsim_mall.profiles.agent_model import DEFAULT_AGENT_ROLES


SYSTEM_PROMPT = (
    "你是室内行人仿真的高层路线规划器。"
    "只返回严格 JSON，不要输出解释、推理过程或 Markdown。"
)


def build_agent_routing_messages(agent_infos, region_infos, exit_infos):
    prompt = {
        "task": "为 JuPedSim 室内仿真中的每个 agent 规划高层行为路线。",
        "policy": [
            "必须为 agents 列表中的每个 agent_id 返回且只返回一个计划。",
            "当前 agents 列表是一个批次，只规划本批次，不要生成批次外的 agent。",
            "每个计划只对对应 agent_id 生效，不能用出生区域级别的统一计划替代个体计划。",
            "参考每个 agent 的 profile 生成 role、subtype、intent、activities、final_exit 和 desired_speed_mps。",
            "优先使用 profile.subtype、profile.preferences、profile.personal_variation、profile.motivation 来体现个体差异。",
            "如果 profile 中包含 history_hint、intent_hint 或 memory_hint，应把它们作为软约束，而不是硬性复制。",
            "activities 是 0 到 3 个中间活动点；每个 activity 必须包含 region、action、wait_seconds。",
            "activity.region 只能使用 regions 列表中的 name。",
            "final_exit 只能使用该 agent 的 candidate_exits 中列出的出口名称。",
            "wait_seconds 表示到达中间地点后的停留时间；通勤者通常 0 到 10 秒，购物者、访客、工作人员可为 10 到 120 秒。",
            "角色应符合地图语义：通勤者更可能走向车站、停车、直接出口；购物者更可能访问商店或餐饮；工作人员更可能访问服务通道或办公电梯。",
            "同一 role 的 agent 不应机械复制同一路线；如果 subtype 或 variation 不同，应在活动区域、停留时间、出口选择或路线长度上合理变化。",
            "同一出生区域的 agent 可以有不同角色、意图和路线；路线应大致符合空间顺序，避免明显绕远。",
            "desired_speed_mps 必须在 0.8 到 1.8 之间。",
            "输出必须是 JSON 对象，不要包含解释文本。",
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
                    "subtype": "profile 中的 subtype 或合理细分类别",
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


def prompt_hash(messages) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
