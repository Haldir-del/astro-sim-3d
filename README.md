# 🚀 AstroSim 3D - High-Performance Hybrid Orbital Simulator

AstroSim 3D is a high-performance desktop application engineered to model and visualize 3D orbital mechanics and satellite trajectories in real-time. By coupling a low-level computational core with a modern graphical user interface, the system achieves microsecond-level ordinary differential equation (ODE) solving alongside smooth 3D rendering.

## 🏗️ Architecture Overview

The software implements a hybrid architecture combining the raw computational speed of native compiled languages with the rapid GUI deployment of high-level scripting languages:

* **Computational Core (C++):** Executes the resource-intensive physics simulation loop. It computes gravitational vector fields and updates the state vectors using a custom **Runge-Kutta 4th Order (RK4)** numerical integrator.
* **Interoperability Layer (ctypes):** Facilitates low-latency, zero-serialization data binding by passing pointers to pre-allocated data arrays directly across the C++/Python memory space boundary.
* **Front-End Interface (Python):** Renders an interactive 3D spatial viewport with locked cubic box aspects to mathematically guarantee accurate, undistorted spherical Earth geometry and tracking diagnostics.

---

## 🕹️ Mathematical & Physics Foundation

The backend solver leverages **Newton's Law of Universal Gravitation** mapped into a continuous 3D coordinate system:

$$\mathbf{a} = -G \frac{M}{|\mathbf{r}|^3} \mathbf{r}$$

To step through continuous time without cumulative floating-point drift, the system utilizes the classical **RK4 Integration** method, executing four directional slope evaluations per time increment ($dt = 1.0\text{s}$):

* $k_1 = f(t_n, y_n)$
* $k_2 = f(t_n + \frac{dt}{2}, y_n + \frac{dt}{2}k_1)$
* $k_3 = f(t_n + \frac{dt}{2}, y_n + \frac{dt}{2}k_2)$
* $k_4 = f(t_n + dt, y_n + dt \cdot k_3)$
* $y_{n+1} = y_n + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

---

## 🛠️ Tech Stack & Requirements

* **Languages:** C++ (GCC v11+), Python (v3.10+)
* **GUI Framework:** CustomTkinter (Modern Dark-Mode UI Toolkit)
* **Data Processing:** NumPy (Fast Vector Array Manipulations)
* **Visualization Engine:** Matplotlib (Interactive `mplot3d` Viewport)

### Dependencies Installation

Ensure your Python virtual environment has all required graphics libraries installed:

pip install -r requirements.txt

⚙️ Compilation & Deployment Guide
To deploy or modify the high-speed calculation core, rebuild the shared binary dynamic link library (DLL) using a C++ compiler from the project root:

1. Compile the C++ Engine
g++ -shared -o orbital_engine.dll core/orbital_engine.cpp

2. Execute the Application
Launch the interface event loop through the main application entry point:
py app/gui_app.py

🛰️ Engineering Test Scenarios
Once the interface launches, inputs can be configured to validate specific aerospace trajectories:

Stable Low Earth Orbit (LEO): Set Altitude to 400 km, Velocity to 7660 m/s, and Duration to 6000 s. Generates a perfect circular path mirroring the International Space Station (ISS).

Ballistic Re-entry / Collision: Lower Velocity to 5000 m/s. The solver immediately computes a sub-orbital decay trajectory resulting in an atmospheric impact zone intersection.

Hyperbolic Escape Trajectory: Maximize Velocity to 11500 m/s. The trajectory line shifts into an open-ended hyperbola as kinetic energy overcomes the Earth's gravitational binding energy.