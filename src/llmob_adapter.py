"""Lightweight LLMob-style persona and activity helpers for the mall scene."""

from __future__ import annotations

import random


MALL_PERSONAS = [
    {
        "role": "commuter",
        "subtypes": [
            {
                "name": "train_connection",
                "preferences": ["train corridor", "direct exit", "wide corridor"],
                "motivation": "Reach the train-station corridor as efficiently as possible.",
                "default_wait_seconds": 0,
                "speed_delta": 0.05,
            },
            {
                "name": "parking_connection",
                "preferences": ["parking escalator", "stairs", "direct exit"],
                "motivation": "Move toward parking or lower-level access without unnecessary stops.",
                "default_wait_seconds": 0,
                "speed_delta": 0.0,
            },
            {
                "name": "hurried_passenger",
                "preferences": ["shortest route", "nearest exit", "no stops"],
                "motivation": "Leave quickly and avoid all optional detours.",
                "default_wait_seconds": 0,
                "speed_delta": 0.12,
            },
        ],
        "attribute": "You are passing through the building and prefer efficient routes.",
        "motivation": "Reach the train-station corridor, parking connection, or a nearby exit quickly.",
        "preferences": ["train corridor", "parking escalator", "direct exit"],
        "desired_speed_mps": 1.45,
        "default_wait_seconds": 0,
    },
    {
        "role": "shopper",
        "subtypes": [
            {
                "name": "window_shopper",
                "preferences": ["shops", "shopfronts", "main corridor"],
                "motivation": "Browse shop areas before leaving.",
                "default_wait_seconds": 50,
                "speed_delta": -0.08,
            },
            {
                "name": "restaurant_visitor",
                "preferences": ["restaurant", "dining entrance", "food area"],
                "motivation": "Visit a restaurant or dining area, then exit.",
                "default_wait_seconds": 75,
                "speed_delta": -0.12,
            },
            {
                "name": "quick_buyer",
                "preferences": ["convenience store", "nearby shop", "nearest exit"],
                "motivation": "Make a short purchase and leave promptly.",
                "default_wait_seconds": 20,
                "speed_delta": 0.05,
            },
        ],
        "attribute": "You are visiting shops and may browse before leaving.",
        "motivation": "Visit shop or restaurant areas, then leave through a convenient exit.",
        "preferences": ["shops", "restaurant", "convenience store"],
        "desired_speed_mps": 1.15,
        "default_wait_seconds": 45,
    },
    {
        "role": "staff",
        "subtypes": [
            {
                "name": "office_worker",
                "preferences": ["office elevator", "service corridor", "upper-floor stairs"],
                "motivation": "Head toward work-related access points and then continue through the building.",
                "default_wait_seconds": 25,
                "speed_delta": 0.0,
            },
            {
                "name": "retail_employee",
                "preferences": ["shop area", "service corridor", "convenience store"],
                "motivation": "Move between shop/service areas with practical, familiar routes.",
                "default_wait_seconds": 35,
                "speed_delta": -0.02,
            },
            {
                "name": "maintenance_staff",
                "preferences": ["service corridor", "stairs", "utility access"],
                "motivation": "Use service-oriented paths and avoid unnecessary public-area browsing.",
                "default_wait_seconds": 15,
                "speed_delta": 0.08,
            },
        ],
        "attribute": "You work in or around the building and know service corridors.",
        "motivation": "Move toward service routes, office elevators, or nearby functional areas.",
        "preferences": ["service corridor", "office elevator", "convenience store"],
        "desired_speed_mps": 1.30,
        "default_wait_seconds": 20,
    },
    {
        "role": "visitor",
        "subtypes": [
            {
                "name": "first_time_visitor",
                "preferences": ["visible landmarks", "shops", "stairs"],
                "motivation": "Explore a visible landmark or shop area before choosing an exit.",
                "default_wait_seconds": 35,
                "speed_delta": -0.08,
            },
            {
                "name": "lost_visitor",
                "preferences": ["open area", "escalator", "multiple landmarks"],
                "motivation": "Move through recognizable areas while finding a suitable way out.",
                "default_wait_seconds": 25,
                "speed_delta": -0.15,
            },
            {
                "name": "meeting_visitor",
                "preferences": ["restaurant", "open area", "shop area"],
                "motivation": "Stop briefly at a likely meeting or browsing area, then leave.",
                "default_wait_seconds": 55,
                "speed_delta": -0.04,
            },
        ],
        "attribute": "You are exploring the building and may stop at visible landmarks.",
        "motivation": "Explore shops, restaurants, stairs, or escalators before choosing an exit.",
        "preferences": ["shops", "restaurant", "stairs", "escalator"],
        "desired_speed_mps": 1.20,
        "default_wait_seconds": 30,
    },
]


def build_agent_profile(agent_id, spawn_name, spawn_order, seed=2026):
    rng = random.Random(f"{seed}-{agent_id}-{spawn_name}-{spawn_order}")
    persona = rng.choice(MALL_PERSONAS).copy()
    subtype = rng.choice(persona.pop("subtypes"))
    persona["subtype"] = subtype["name"]
    persona["profile_id"] = f"profile_{agent_id}"
    persona["motivation"] = subtype["motivation"]
    persona["preferences"] = subtype["preferences"]
    persona["default_wait_seconds"] = subtype["default_wait_seconds"]
    persona["personal_variation"] = rng.choice([
        "prefers shortest plausible path",
        "accepts a small detour for interesting places",
        "avoids unnecessary stops",
        "may pause briefly at the intended destination",
        "prefers a different stop than nearby agents with the same role",
        "chooses a less crowded-looking alternative when plausible",
    ])
    persona["desired_speed_mps"] = round(
        max(0.8, min(1.8, persona["desired_speed_mps"] + subtype["speed_delta"] + rng.uniform(-0.12, 0.12))),
        2,
    )
    return persona


def normalize_wait_seconds(value, fallback):
    try:
        wait = float(value)
    except (TypeError, ValueError):
        wait = fallback
    return int(max(0, min(180, wait)))
