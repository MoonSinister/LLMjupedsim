"""Movement model construction and parameter validation."""

from __future__ import annotations

from dataclasses import dataclass

import jupedsim as jps


@dataclass(frozen=True)
class MovementConfig:
    model_name: str
    agent_radius: float
    agent_time_gap: float
    neighbor_repulsion_strength: float
    neighbor_repulsion_range: float
    geometry_repulsion_strength: float
    geometry_repulsion_range: float
    cfsv3_range_x_scale: float
    cfsv3_range_y_scale: float
    cfsv3_theta_max: float
    cfsv3_agent_buffer: float
    avm_wall_buffer_distance: float
    avm_anticipation_time: float
    avm_reaction_time: float

    @classmethod
    def from_args(cls, args):
        config = cls(
            model_name=args.movement_model,
            agent_radius=args.agent_radius,
            agent_time_gap=args.agent_time_gap,
            neighbor_repulsion_strength=args.neighbor_repulsion_strength,
            neighbor_repulsion_range=args.neighbor_repulsion_range,
            geometry_repulsion_strength=args.geometry_repulsion_strength,
            geometry_repulsion_range=args.geometry_repulsion_range,
            cfsv3_range_x_scale=args.cfsv3_range_x_scale,
            cfsv3_range_y_scale=args.cfsv3_range_y_scale,
            cfsv3_theta_max=args.cfsv3_theta_max,
            cfsv3_agent_buffer=args.cfsv3_agent_buffer,
            avm_wall_buffer_distance=args.avm_wall_buffer_distance,
            avm_anticipation_time=args.avm_anticipation_time,
            avm_reaction_time=args.avm_reaction_time,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_name not in {"cfs", "cfsv2", "cfsv3", "avm"}:
            raise ValueError(f"Unsupported movement model: {self.model_name}")
        positive = {
            "agent_radius": self.agent_radius,
            "agent_time_gap": self.agent_time_gap,
            "neighbor_repulsion_strength": self.neighbor_repulsion_strength,
            "neighbor_repulsion_range": self.neighbor_repulsion_range,
            "geometry_repulsion_strength": self.geometry_repulsion_strength,
            "geometry_repulsion_range": self.geometry_repulsion_range,
            "cfsv3_range_x_scale": self.cfsv3_range_x_scale,
            "cfsv3_range_y_scale": self.cfsv3_range_y_scale,
            "cfsv3_theta_max": self.cfsv3_theta_max,
            "avm_anticipation_time": self.avm_anticipation_time,
            "avm_reaction_time": self.avm_reaction_time,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"movement parameters must be positive: {', '.join(invalid)}")
        if self.cfsv3_agent_buffer < 0 or self.avm_wall_buffer_distance < 0:
            raise ValueError("movement buffer distances must be non-negative")


def create_movement_model(model_name):
    if model_name == "cfs":
        return jps.CollisionFreeSpeedModel()
    if model_name == "cfsv2":
        return jps.CollisionFreeSpeedModelV2()
    if model_name == "cfsv3":
        return jps.CollisionFreeSpeedModelV3()
    if model_name == "avm":
        return jps.AnticipationVelocityModel()
    raise ValueError(f"Unsupported movement model: {model_name}")


def create_agent_parameters(model_name, *, journey_id, stage_id, position, desired_speed, args):
    config = MovementConfig.from_args(args)
    common = {
        "journey_id": journey_id, "stage_id": stage_id, "position": position,
        "radius": config.agent_radius, "desired_speed": desired_speed,
        "time_gap": config.agent_time_gap,
    }
    if model_name == "cfs":
        return jps.CollisionFreeSpeedModelAgentParameters(**common)
    if model_name == "cfsv2":
        return jps.CollisionFreeSpeedModelV2AgentParameters(
            **common, strength_neighbor_repulsion=config.neighbor_repulsion_strength,
            range_neighbor_repulsion=config.neighbor_repulsion_range,
            strength_geometry_repulsion=config.geometry_repulsion_strength,
            range_geometry_repulsion=config.geometry_repulsion_range,
        )
    if model_name == "cfsv3":
        return jps.CollisionFreeSpeedModelV3AgentParameters(
            **common, strength_neighbor_repulsion=config.neighbor_repulsion_strength,
            range_neighbor_repulsion=config.neighbor_repulsion_range,
            strength_geometry_repulsion=config.geometry_repulsion_strength,
            range_geometry_repulsion=config.geometry_repulsion_range,
            range_x_scale=config.cfsv3_range_x_scale, range_y_scale=config.cfsv3_range_y_scale,
            theta_max_upper_bound=config.cfsv3_theta_max, agent_buffer=config.cfsv3_agent_buffer,
        )
    if model_name == "avm":
        return jps.AnticipationVelocityModelAgentParameters(
            **common, strength_neighbor_repulsion=config.neighbor_repulsion_strength,
            range_neighbor_repulsion=config.neighbor_repulsion_range,
            wall_buffer_distance=config.avm_wall_buffer_distance,
            anticipation_time=config.avm_anticipation_time, reaction_time=config.avm_reaction_time,
        )
    raise ValueError(f"Unsupported movement model: {model_name}")
