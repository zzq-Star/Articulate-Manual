"""Base converter interface and standard trajectory representation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Trajectory:
    """Standard trajectory representation for converter input."""
    waypoints: List[Dict]           # [{positions: [...], type: "PTP"|"LIN"|"CIRC", ...}]
    speed: float = 0.25             # m/s or rad/s
    blend: float = 0.0              # blend radius (m)
    tool_name: str = "tool0"
    payload_kg: float = 0.0
    acceleration: float = 1.0       # m/s² or rad/s²
    metadata: Dict = field(default_factory=dict)


class BaseConverter(ABC):
    """Abstract base for brand-specific script converters."""

    @property
    @abstractmethod
    def brand(self) -> str:
        """Brand identifier: 'ur', 'kuka', 'abb'."""
        ...

    @abstractmethod
    def convert(self, trajectory: Trajectory, output_dir: str) -> Dict[str, str]:
        """Convert trajectory to brand-specific script files.

        Args:
            trajectory: Standard trajectory to convert.
            output_dir: Target directory for generated files.

        Returns:
            Dict mapping relative file paths to file contents.
        """
        ...

    @abstractmethod
    def generate_safety_checklist(self) -> List[str]:
        """Return brand-specific safety checklist items."""
        ...

    def generate_deployment_guide(self, trajectory: Trajectory) -> str:
        """Generate a human-readable deployment guide."""
        lines = [
            f"# Deployment Guide — {self.brand.upper()}",
            "",
            f"## Overview",
            f"Tool: {trajectory.tool_name}",
            f"Payload: {trajectory.payload_kg} kg",
            f"Speed: {trajectory.speed}",
            f"Waypoints: {len(trajectory.waypoints)}",
            "",
            "## Waypoints",
        ]
        for i, wp in enumerate(trajectory.waypoints):
            pos = wp.get("positions", [])
            wp_type = wp.get("type", "PTP")
            lines.append(f"  {i+1}. [{wp_type}] {pos}")
        lines.extend([
            "",
            "## Safety",
            *[f"  - {item}" for item in self.generate_safety_checklist()],
        ])
        return "\n".join(lines)
