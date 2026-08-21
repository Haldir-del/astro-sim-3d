# 📋 Pendientes — AstroSim 3D

Estado: motor C++ reescrito (RKF45 + J2 + arrastre USSA76), bindings y GUI
reescritos. Falta compilar, terminar tests y documentar.

## Críticos (bloquean ejecutar el proyecto)

- [x] **Compilar la nueva DLL** — `g++ -shared -O2 -std=c++17 ...` ✅ compila
- [ ] **🐛 BUG ACTIVO — deriva de energía en dos cuerpos**: con J2/drag
      desactivados la energía relativa deriva ~1e-4 en 100 s y ~2 % en 5560 s,
      y crece con dt_max (síntoma de integrador, no del modelo de fuerzas).
      Revisar `rkf45_step()` / control de paso en `simulate_trajectory()`
      (core/orbital_engine.cpp). Los coeficientes RKF45 del tableau parecen
      correctos vs TR R-287; sospecha: lógica de crecimiento de paso o el
      clamp `h = duration - t` combinado con el growth factor.
- [ ] **Escribir `tests/test_engine.py`**: conservación de energía y momento
      angular (2 cuerpos), cierre de órbita circular tras 1 periodo, precesión
      nodal J2 medida vs analítica (`j2_nodal_rate_rad_s`), decaimiento
      semieje por arrastre, detección de impacto (con refinamiento) / escape /
      búfer lleno, validación de argumentos.
- [ ] **Ejecutar la suite completa** `pytest` y corregir lo que falle.
- [ ] **Actualizar `README.md`**: nueva arquitectura, modelo físico con
      ecuaciones, referencias oficiales (NASA TR R-287, USSA76 NTRS 19770009539,
      GDC Orbit Primer, NGA WGS 84).

## Verificación

- [ ] Smoke test de GUI: importar `app.gui_app`, instanciar sin `mainloop`,
      correr una simulación corta end-to-end vía `engine.simulate`.
- [ ] Verificar telemetría: Ω̇ medida ≈ Ω̇ analítica (<5 %) en órbita ISS-like.
- [ ] Comprobar continuidad del modelo atmosférico en los 86 km (C++ ↔ Python).

## Mejoras futuras (no bloqueantes)

- [ ] Trazas de tierra (ground track) ECEF con GMST.
- [ ] Perturbación de tercer cuerpo Luni-Solar.
- [ ] Atmósfera NRLMSISE-00 opcional (requiere índices solares).
- [ ] Export CSV/KML de trayectorias.
- [ ] Empaquetado (PyInstaller) y CI (GitHub Actions que compile la DLL y corra pytest).
