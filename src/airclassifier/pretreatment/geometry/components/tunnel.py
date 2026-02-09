"""
Attenuation Tunnel Geometry
===========================

RF attenuation tunnels at the infeed and outfeed of the oven.
These prevent RF leakage while allowing material to pass through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import hollow_box_mesh

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class TunnelParams:
    """Attenuation tunnel parameters based on GP-15 measurements."""

    length: float = 0.245       # [m] Tunnel length (X direction) - 24.5cm
    height: float = 0.258       # [m] Internal height (Y direction) - 25.8cm
    width: float = 0.76         # [m] Internal width (Z direction) - 76cm
    wall_thickness: float = 0.02

    tunnel_type: Literal["infeed", "outfeed"] = "infeed"

    # Position (corner at infeed end)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config, tunnel_type: str = "infeed") -> "TunnelParams":
        """Create params from MachineConfig."""
        return cls(
            width=config.belt_width_m + 0.1,  # Slightly wider than belt
            tunnel_type=tunnel_type,
        )


class TunnelGeometry:
    """Attenuation tunnel for RF leakage prevention.

    The tunnels are hollow rectangular ducts open at both ends
    (along the X axis) to allow material flow on the conveyor.
    """

    def __init__(self, params: Optional[TunnelParams] = None):
        self.params = params or TunnelParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate hollow tunnel mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": f"{self.params.tunnel_type}_tunnel"}

        p = self.params
        cx, cy, cz = p.center

        # Hollow box open on X ends
        self._vertices, self._triangles = hollow_box_mesh(
            cx, cy, cz,
            p.length, p.height, p.width,
            p.wall_thickness,
        )

        return self._vertices, self._triangles, {"type": f"{p.tunnel_type}_tunnel"}

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Inlet and outlet rectangular ports.

        - inlet: Outer end (away from oven)
        - outlet: Inner end (connects to oven)
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        cx, cy, cz = p.center

        if p.tunnel_type == "infeed":
            # Infeed: material enters at x=0 (inlet), exits at x=length (outlet to oven)
            return {
                'inlet': ConnectionPort(
                    position=(cx, cy + p.height / 2, cz + p.width / 2),
                    direction=(-1.0, 0.0, 0.0),  # Points outward (away from oven)
                    width=p.width,
                    height=p.height,
                    port_type=PortType.RECTANGULAR,
                    name="infeed_tunnel_inlet",
                ),
                'outlet': ConnectionPort(
                    position=(cx + p.length, cy + p.height / 2, cz + p.width / 2),
                    direction=(1.0, 0.0, 0.0),  # Points into oven
                    width=p.width,
                    height=p.height,
                    port_type=PortType.RECTANGULAR,
                    name="infeed_tunnel_outlet",
                ),
            }
        else:  # outfeed
            # Outfeed: material enters at x=0 (inlet from oven), exits at x=length (outlet)
            return {
                'inlet': ConnectionPort(
                    position=(cx, cy + p.height / 2, cz + p.width / 2),
                    direction=(-1.0, 0.0, 0.0),  # Points back toward oven
                    width=p.width,
                    height=p.height,
                    port_type=PortType.RECTANGULAR,
                    name="outfeed_tunnel_inlet",
                ),
                'outlet': ConnectionPort(
                    position=(cx + p.length, cy + p.height / 2, cz + p.width / 2),
                    direction=(1.0, 0.0, 0.0),  # Points outward
                    width=p.width,
                    height=p.height,
                    port_type=PortType.RECTANGULAR,
                    name="outfeed_tunnel_outlet",
                ),
            }
