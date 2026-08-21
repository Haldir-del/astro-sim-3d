"""Matplotlib 3D viewport for orbit visualization.

Encapsulates all figure handling so the GUI layer stays declarative.
Distances are handled internally in kilometers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.orbital import WGS84_A, WGS84_F, EARTH_RADIUS_MEAN_M

_BG_FIG = "#2b2b2b"
_BG_AX = "#1e1e1e"
_GRID = "#444444"
_ORBIT = "#badc58"
_EARTH = "#22a6b3"
_EDGE = "#130cb7"


class OrbitViewport:
    """Dark-themed 3D viewport with a WGS-84 ellipsoid and trajectory plot."""

    def __init__(self, master):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt

        self._plt = plt
        self.fig = plt.figure(figsize=(7, 7), facecolor=_BG_FIG)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(_BG_AX)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.clear()

    # ------------------------------------------------------------------
    def _style_axes(self, title: str) -> None:
        ax = self.ax
        ax.set_facecolor(_BG_AX)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color=_GRID)
        ax.set_title(title, color="white", fontsize=12, weight="bold")
        ax.set_xlabel("X (km)", color="white", fontsize=9)
        ax.set_ylabel("Y (km)", color="white", fontsize=9)
        ax.set_zlabel("Z (km)", color="white", fontsize=9)

    def _draw_earth(self) -> None:
        """WGS-84 ellipsoid mesh scaled to km."""
        a = WGS84_A / 1000.0
        b = a * (1.0 - WGS84_F)
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        ex = a * np.cos(u) * np.sin(v)
        ey = a * np.sin(u) * np.sin(v)
        ez = b * np.cos(v)
        self.ax.plot_surface(ex, ey, ez, color=_EARTH, alpha=0.35,
                             edgecolor=_EDGE, linewidth=0.25, shade=True)

    def clear(self) -> None:
        self.ax.clear()
        self._style_axes("AstroSim 3D — Vista de Trayectoria Orbital")
        self._draw_earth()
        lim = EARTH_RADIUS_MEAN_M / 1000.0 * 2.0
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except AttributeError:  # older matplotlib
            pass
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def show_trajectory(self, positions_m: np.ndarray,
                        title: Optional[str] = None,
                        status: str = "completed") -> None:
        """Plot a trajectory given positions in meters (n, 3)."""
        pos_km = np.asarray(positions_m, dtype=float) / 1000.0
        if pos_km.ndim != 2 or pos_km.shape[1] != 3 or pos_km.shape[0] < 1:
            raise ValueError("positions must be (n, 3)")

        self.ax.clear()
        self._style_axes(title or "Trayectoria Orbital (RKF45 + J2 + Drag)")
        self._draw_earth()

        color = _ORBIT if status != "impact" else "#ff7675"
        label = {"impact": "Trayectoria de impacto",
                 "escape": "Trayectoria de escape",
                 "completed": "Órbita simulada"}.get(status, "Trayectoria")

        self.ax.plot(pos_km[:, 0], pos_km[:, 1], pos_km[:, 2],
                     color=color, linewidth=1.8, label=label)

        # Start (green) and end (red) markers.
        self.ax.scatter(*pos_km[0], color="#55efc4", s=45,
                        label="Inicio", depthshade=False)
        self.ax.scatter(*pos_km[-1], color="red", s=45,
                        label="Fin", depthshade=False)

        span = float(np.max(np.abs(pos_km)))
        lim = max(span * 1.15, EARTH_RADIUS_MEAN_M / 1000.0 * 1.6)
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        self.ax.legend(facecolor=_BG_FIG, edgecolor="none",
                       labelcolor="white", fontsize=8, loc="upper left")
        self.canvas.draw_idle()
