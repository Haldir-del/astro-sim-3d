"""AstroSim 3D — GUI principal.

Front-end CustomTkinter sobre el motor nativo RKF45 (via app.engine.simulate).
La simulación corre en un hilo secundario para no bloquear la interfaz.

Física incluida:
  * Gravedad newtoniana de masa puntual (WGS-84 GM)
  * Perturbación J2 por achatamiento terrestre (precesión nodal/apsidal real)
  * Arrastre atmosférico con atmósfera co-rotante (U.S. Standard Atmosphere 1976)
"""
from __future__ import annotations

import threading
import tkinter as tk
from typing import Optional

import customtkinter as ctk
import numpy as np

from app.engine import Forces, SimulationResult, simulate
from app.orbital import (
    EARTH_MU,
    EARTH_RADIUS_MEAN_M,
    circular_velocity,
    escape_velocity,
    j2_nodal_rate_rad_s,
    launch_state,
    orbital_elements,
)
from app.visualizer import OrbitViewport

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

_OK = "#55efc4"
_WARN = "#fdcb6e"
_ERR = "#ff7675"
_INFO = "#74b9ff"


class AstroSimApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AstroSim 3D — Simulador Orbital de Alta Fidelidad")
        self.geometry("1280x760")
        self.minsize(1100, 700)

        self._sim_thread: Optional[threading.Thread] = None
        self._result: Optional[SimulationResult] = None

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self.viewport = OrbitViewport(self.plot_frame)

    # ==================================================================
    # Interfaz
    # ==================================================================
    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkScrollableFrame(self, width=330, corner_radius=15)
        self.sidebar.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="AstroSim 3D Engine",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(
            padx=14, pady=(8, 2))
        ctk.CTkLabel(self.sidebar,
                     text="RKF45 (NASA TR R-287) · J2 · USSA76",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(
            padx=14, pady=(0, 10))

        # ---- Condiciones iniciales ----
        ctk.CTkLabel(self.sidebar, text="CONDICIONES INICIALES",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").pack(fill="x", padx=14, pady=(6, 2))
        self.altitude = self._entry("Altitud inicial (km)", "400")
        self.velocity = self._entry("Velocidad tangencial (m/s)", "7672")
        self.inclination = self._entry("Inclinación (deg)", "51.6")
        self.raan = self._entry("RAAN (deg)", "0")
        self.duration = self._entry("Duración (s)", "20000")

        # ---- Modelo físico ----
        ctk.CTkLabel(self.sidebar, text="MODELO FÍSICO",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        self.sw_j2 = ctk.CTkSwitch(self.sidebar, text="Perturbación J2 "
                                   "(achatamiento)")
        self.sw_j2.select()
        self.sw_j2.pack(fill="x", padx=14, pady=3)

        self.sw_drag = ctk.CTkSwitch(self.sidebar, text="Arrastre atmosférico "
                                     "(USSA76)")
        self.sw_drag.pack(fill="x", padx=14, pady=3)

        drag_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        drag_frame.pack(fill="x", padx=14, pady=(2, 0))
        drag_frame.grid_columnconfigure((0, 1), weight=1)
        self.cd = self._mini_entry(drag_frame, "Cd", "2.2", 0)
        self.area = self._mini_entry(drag_frame, "Área (m²)", "20", 1)
        self.mass = self._mini_entry(drag_frame, "Masa (kg)", "1000", 0)

        # ---- Botón y estado ----
        self.btn_run = ctk.CTkButton(self.sidebar, height=40,
                                     text="⚡ Simular órbita",
                                     font=ctk.CTkFont(size=15, weight="bold"),
                                     command=self.launch_simulation)
        self.btn_run.pack(fill="x", padx=14, pady=14)

        self.lbl_status = ctk.CTkLabel(
            self.sidebar, text="Sistema listo", text_color="gray",
            wraplength=290, justify="left")
        self.lbl_status.pack(fill="x", padx=14, pady=(0, 8))

        # ---- Telemetría ----
        tele = ctk.CTkFrame(self.sidebar, corner_radius=10,
                            fg_color="#1e1e1e")
        tele.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        ctk.CTkLabel(tele, text="📡 TELEMETRÍA POST-SIMULACIÓN",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#badc58").pack(pady=(6, 2))
        self.tele_box = ctk.CTkTextbox(tele, font=ctk.CTkFont(
            family="Courier", size=12), fg_color="#111111")
        self.tele_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tele_box.insert("1.0",
                             "Ejecuta una simulación para ver elementos "
                             "orbitales,\nderiva de RAAN medida vs analítica, "
                             "y eventos.")

    def _entry(self, label: str, default: str) -> ctk.CTkEntry:
        ctk.CTkLabel(self.sidebar, text=label, anchor="w").pack(
            fill="x", padx=14, pady=(8, 0))
        e = ctk.CTkEntry(self.sidebar)
        e.insert(0, default)
        e.pack(fill="x", padx=14, pady=(2, 0))
        return e

    def _mini_entry(self, frame, label: str, default: str,
                    col: int) -> ctk.CTkEntry:
        ctk.CTkLabel(frame, text=label, anchor="w",
                     font=ctk.CTkFont(size=11)).grid(
            row=0, column=col, padx=6, pady=(6, 0), sticky="ew")
        e = ctk.CTkEntry(frame, placeholder_text=default, height=26)
        e.insert(0, default)
        e.grid(row=1, column=col, padx=6, pady=(0, 4), sticky="ew")
        return e

    def _set_status(self, text: str, color: str) -> None:
        self.lbl_status.configure(text=text, text_color=color)

    # ==================================================================
    # Ejecución de la simulación
    # ==================================================================
    def launch_simulation(self) -> None:
        if self._sim_thread is not None and self._sim_thread.is_alive():
            return
        try:
            cfg = self._read_inputs()
        except ValueError as exc:
            self._set_status(f"❌ Entrada inválida: {exc}", _ERR)
            return

        self.btn_run.configure(state="disabled", text="⏳ Integrando…")
        self._set_status("Integrando ecuaciones de movimiento…", _WARN)
        self._sim_thread = threading.Thread(
            target=self._worker, args=(cfg,), daemon=True)
        self._sim_thread.start()
        self.after(50, self._poll_worker)

    def _read_inputs(self) -> dict:
        alt_km = float(self.altitude.get())
        vel = float(self.velocity.get())
        inc = float(self.inclination.get())
        raan = float(self.raan.get())
        dur = float(self.duration.get())
        if alt_km <= 0 or dur <= 0:
            raise ValueError("altitud y duración deben ser positivas")
        if not (-90 <= inc <= 90):
            inc = max(-90.0, min(90.0, inc))
        cd = float(self.cd.get()) if self.cd.get() else 2.2
        area = float(self.area.get()) if self.area.get() else 20.0
        mass = float(self.mass.get()) if self.mass.get() else 1000.0
        return {
            "state": launch_state(alt_km, vel, inc, raan),
            "duration": dur,
            "forces": Forces(cd=max(cd, 0.0), area_m2=max(area, 0.0),
                             mass_kg=max(mass, 1e-9),
                             enable_j2=bool(self.sw_j2.get()),
                             enable_drag=bool(self.sw_drag.get())),
        }

    def _worker(self, cfg: dict) -> None:
        try:
            self._result = simulate(cfg["state"], cfg["duration"],
                                    dt_max=1.0, rel_tol=1e-10,
                                    forces=cfg["forces"])
        except Exception as exc:  # noqa: BLE001 - surfaced on the UI
            self._result = exc

    def _poll_worker(self) -> None:
        if self._sim_thread is None:
            return
        if self._sim_thread.is_alive():
            self.after(50, self._poll_worker)
            return
        self.btn_run.configure(state="normal", text="⚡ Simular órbita")
        result = self._result
        self._sim_thread = None
        self._result = None

        if isinstance(result, Exception):
            self._set_status(f"❌ Error del motor: {result}", _ERR)
            return
        assert isinstance(result, SimulationResult)
        try:
            self._present(result)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"❌ Error presentando resultados: {exc}", _ERR)

    # ==================================================================
    # Presentación de resultados
    # ==================================================================
    def _present(self, res: SimulationResult) -> None:
        from app.orbital import radius as state_radius, speed as state_speed

        status_msg = {
            "completed": ("✅ Simulación completada", _OK),
            "impact": ("💥 Impacto contra el cuerpo central", _ERR),
            "escape": ("🚀 Trayectoria de escape hiperbólica", _INFO),
            "buffer_full": ("⚠️ Búfer agotado antes de la duración pedida",
                            _WARN),
        }.get(res.status, ("Estado desconocido", _WARN))
        self._set_status(status_msg[0], status_msg[1])

        self.viewport.show_trajectory(res.positions, status=res.status,
                                      title=self._plot_title())

        lines = []
        n_samples = res.t.size
        r0, v0 = state_radius(res.state[0]), state_speed(res.state[0])
        lines.append(f"Muestras integradas : {n_samples}")
        lines.append(f"r inicial          : {r0/1000:,.1f} km")
        lines.append(f"v inicial          : {v0:,.1f} m/s")

        el0 = orbital_elements(res.state[0], EARTH_MU)
        if el0.is_elliptic:
            lines.append(f"Semieje mayor a    : {el0.semi_major_axis/1000:,.1f} km")
            lines.append(f"Excentricidad e    : {el0.eccentricity:.5f}")
            lines.append(f"Inclinación i      : {el0.inclination_deg:.2f} deg")
            if el0.period:
                lines.append(f"Periodo T          : "
                             f"{el0.period/60:.2f} min")
            hp = el0.periapsis_radius - EARTH_RADIUS_MEAN_M
            lines.append(f"Alt. perigeo       : {hp/1000:,.1f} km")
            if el0.apoapsis_radius:
                ha = el0.apoapsis_radius - EARTH_RADIUS_MEAN_M
                lines.append(f"Alt. apogeo        : {ha/1000:,.1f} km")

        # Deriva de RAAN medida en la simulación vs tasa analítica J2.
        if bool(self.sw_j2.get()) and el0.is_elliptic and el0.eccentricity < 0.9:
            measured = self._measure_raan_rate(res, el0)
            analytic = j2_nodal_rate_rad_s(el0.semi_major_axis,
                                           el0.eccentricity,
                                           el0.inclination_deg)
            lines.append("")
            lines.append("Precesión nodal (J2):")
            if measured is not None:
                lines.append(f"  Ω̇ simulada : {measured:+.4f} deg/día")
                lines.append(f"  Ω̇ analítica: "
                             f"{np.degrees(analytic)*86400:+.4f} deg/día")
            else:
                lines.append(f"  Ω̇ analítica: "
                             f"{np.degrees(analytic)*86400:+.4f} deg/día"
                             " (no medible)")

        if bool(self.sw_drag.get()):
            peak = float(np.max(res.acc_pert))
            mean = float(np.mean(res.acc_pert))
            lines.append("")
            lines.append(f"Acel. perturbadora pico : {peak*1000:.4f} mm/s²")
            lines.append(f"Acel. perturbadora media: {mean*1000:.4f} mm/s²")

        if res.status == "impact":
            r_impact = state_radius(res.state[-1])
            lines.append("")
            lines.append(f"Punto de impacto |r| = {r_impact/1000:,.2f} km "
                         f"(t = {res.t[-1]:,.1f} s)")

        self.tele_box.delete("1.0", "end")
        self.tele_box.insert("1.0", "\n".join(lines))

    def _plot_title(self) -> str:
        parts = ["RKF45 adaptativo"]
        if bool(self.sw_j2.get()):
            parts.append("J2")
        if bool(self.sw_drag.get()):
            parts.append("drag USSA76")
        return "AstroSim 3D — " + " + ".join(parts)

    def _measure_raan_rate(self, res: SimulationResult,
                           el0) -> Optional[float]:
        """Fit dΩ/dt (rad/s→deg/day) from sampled osculating elements."""
        from app.orbital import EARTH_MU as MU
        n = res.t.size
        if n < 20:
            return None
        idx = np.unique(np.linspace(0, n - 1, min(n, 400)).astype(int))
        times, raans = [], []
        prev = None
        offset = 0.0
        for i in idx:
            el = orbital_elements(res.state[i], MU)
            if not el.is_elliptic:
                continue
            ra = el.raan_deg
            if prev is not None:
                while ra + offset < prev:
                    offset += 360.0
            prev = ra + offset
            times.append(res.t[i])
            raans.append(np.deg2rad(prev))
        if len(times) < 10:
            return None
        t = np.asarray(times)
        w = np.asarray(raans)
        A = np.vstack([t, np.ones_like(t)]).T
        slope, _ = np.linalg.lstsq(A, w, rcond=None)[0]
        return float(np.degrees(slope) * 86400.0)


if __name__ == "__main__":
    app = AstroSimApp()
    app.mainloop()
