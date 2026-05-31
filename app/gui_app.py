import ctypes
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Correct modern import for 3D projections
import os

# ---- Visual Appearance Configuration ----
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class AstroSimApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ---- Window Configuration ----
        self.title("🚀 AstroSim 3D - High-Performance Orbital Simulator")
        self.geometry("1200x700")
        self.resizable(False, False)

        # ---- Load C++ Core Math Engine ----
        try:
            # Construct absolute path to the DLL relative to this Python file
            dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "orbital_engine.dll"))
            self.libc = ctypes.CDLL(dll_path)
        except OSError:
            # Failure handling if DLL is missing or in wrong path
            self.libc = None
            print("Warning: C++ core engine (orbital_engine.dll) not found. Calculation functionality disabled.")

        # ==========================================
        # INTERFACE STRUCTURE (2-Column Layout)
        # ==========================================
        self.grid_columnconfigure(0, weight=1)  # Control Panel (Left)
        self.grid_columnconfigure(1, weight=4)  # 3D Viewport (Right)
        self.grid_rowconfigure(0, weight=1)

        # ---- LEFT COLUMN: CONTROL PANEL Sidebar ----
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=15)
        self.sidebar.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Sidebar Title
        self.title_label = ctk.CTkLabel(self.sidebar, text="AstroSim 3D Engine", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        # Input 1: Initial Altitude (in Kilometers)
        self.lbl_altitude = ctk.CTkLabel(self.sidebar, text="Initial Altitude (km above Earth):", anchor="w")
        self.lbl_altitude.pack(fill="x", padx=20, pady=(10, 2))
        self.txt_altitude = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. 400")
        self.txt_altitude.insert(0, "400") # Default: ISS altitude
        self.txt_altitude.pack(fill="x", padx=20, pady=5)

        # Input 2: Tangential Velocity (in m/s)
        self.lbl_velocity = ctk.CTkLabel(self.sidebar, text="Initial Orbital Velocity (m/s):", anchor="w")
        self.lbl_velocity.pack(fill="x", padx=20, pady=(10, 2))
        self.txt_velocity = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. 7660")
        self.txt_velocity.insert(0, "7660") # Standard circular velocity for default altitude
        self.txt_velocity.pack(fill="x", padx=20, pady=5)

        # Input 3: Simulation Time (Steps in seconds)
        self.lbl_steps = ctk.CTkLabel(self.sidebar, text="Simulation Duration (seconds):", anchor="w")
        self.lbl_steps.pack(fill="x", padx=20, pady=(10, 2))
        self.txt_steps = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. 6000")
        self.txt_steps.insert(0, "6000") # Approx one full orbit duration
        self.txt_steps.pack(fill="x", padx=20, pady=5)

        # Telemetry Display Box
        self.telemetry_frame = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#1e1e1e")
        self.telemetry_frame.pack(fill="x", padx=20, pady=15)
        
        self.lbl_telemetry_title = ctk.CTkLabel(self.telemetry_frame, text="📡 LIVE TELEMETRY", font=ctk.CTkFont(size=12, weight="bold"), text_color="#badc58")
        self.lbl_telemetry_title.pack(pady=5)
        
        self.lbl_realtime_vel = ctk.CTkLabel(self.telemetry_frame, text="Current Vel: --- m/s", font=ctk.CTkFont(family="Courier"))
        self.lbl_realtime_vel.pack(pady=2)

        # Simulation Run/Fire Button
        self.btn_simulate = ctk.CTkButton(self.sidebar, text="⚡ Render 3D Orbit", font=ctk.CTkFont(weight="bold"), command=self.run_physics_engine)
        self.btn_simulate.pack(fill="x", padx=20, pady=20)

        # Status Label for Client Feedback
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="System Ready • Awaiting Inputs", text_color="gray")
        self.lbl_status.pack(side="bottom", pady=15)

        # ---- RIGHT COLUMN: 3D GRAPH PANEL ----
        self.plot_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.plot_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        # Initialize empty 3D viewport canvas
        self.setup_empty_3d_plot()

    # ==========================================
    # METHODS (Correctly Indented)
    # ==========================================

    def setup_empty_3d_plot(self):
        """Initializes a dark-themed 3D space viewport with a spherical Earth"""
        self.fig = plt.figure(figsize=(7, 7), facecolor="#2b2b2b") # Dark grey figure background
        
        # Define the subplot explicitly as a 3D projection using modern syntax
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#1e1e1e") # Pitch black space background
        
        # Modern Matplotlib 3D pane styling (Fixes the w_xaxis AttributeError)
        self.ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.grid(True, color="#444444")
        
        # Axis labeling
        self.ax.set_title("Real-Time 3D Space Viewport", color="white", fontsize=14, weight="bold")
        self.ax.set_xlabel("X Distance (km)", color="white")
        self.ax.set_ylabel("Y Distance (km)", color="white")
        self.ax.set_zlabel("Z Distance (km)", color="white")
        
        # Generate 3D Earth Mesh Coordinates (Realistic radius ~6371km)
        earth_radius = 6371
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        ex = earth_radius * np.cos(u) * np.sin(v)
        ey = earth_radius * np.sin(u) * np.sin(v)
        ez = earth_radius * np.cos(v)
        
        # Plot Earth as a 3D surface mesh
        self.ax.plot_surface(ex, ey, ez, color='#22a6b3', alpha=0.3, edgecolor='#130cb7', linewidth=0.3)
        
        # Force uniform scaling across all axes to prevent elliptical distortion
        max_lim = 12000
        self.ax.set_xlim(-max_lim, max_lim)
        self.ax.set_ylim(-max_lim, max_lim)
        self.ax.set_zlim(-max_lim, max_lim)
        self.ax.set_box_aspect((1, 1, 1)) # Explicitly force a clean cubic 3D bounding frame

        # Integrate Matplotlib directly into CustomTkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_physics_engine(self):
        """Fetches GUI inputs, executes RK4 math in C++, and renders 3D trajectory"""
        if not self.libc:
            self.lbl_status.configure(text="❌ Error: Core Engine Missing", text_color="red")
            return

        try:
            # 1. Retrieve and process inputs from GUI
            altitude_km = float(self.txt_altitude.get())
            velocity_ms = float(self.txt_velocity.get())
            steps = int(self.txt_steps.get())

            # Physical conversions for backend math (Standard Earth variables)
            init_x = 0.0
            init_y = 6371000.0 + (altitude_km * 1000) 
            init_vx = velocity_ms
            init_vy = 0.0
            dt = 1.0  # Numerical integration time step (1 second)

            # 2. Reserve C-Types memory arrays for data returning from C++
            out_x = (ctypes.c_double * steps)()
            out_y = (ctypes.c_double * steps)()

            self.lbl_status.configure(text="🔄 Processing Vectors...", text_color="orange")
            self.update_idletasks()

            # 3. High-speed Call to the C++ Core Math Engine via DLL
            self.libc.simulate_orbit(
                ctypes.c_double(init_x), ctypes.c_double(init_y),
                ctypes.c_double(init_vx), ctypes.c_double(init_vy),
                ctypes.c_double(dt), ctypes.c_int(steps),
                out_x, out_y
            )

            # 4. Post-processing: Convert meters back to kilometers for 3D plotting
            x_coords = np.array(out_x) / 1000
            y_coords = np.array(out_y) / 1000
            z_coords = np.zeros_like(x_coords) # Flat plane projection relative to the equator

            # ---- Update Live Telemetry Readings ----
            self.lbl_realtime_vel.configure(text=f"Current Vel: {velocity_ms:,.2f} m/s", text_color="#badc58")

            # ---- Re-render Viewport ----
            self.ax.clear()
            self.ax.set_facecolor("#1e1e1e")
            self.ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.tick_params(colors='white')
            self.ax.grid(True, color="#444444")

            # Redraw Earth Sphere surface
            earth_radius = 6371
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
            ex = earth_radius * np.cos(u) * np.sin(v)
            ey = earth_radius * np.sin(u) * np.sin(v)
            ez = earth_radius * np.cos(v)
            self.ax.plot_surface(ex, ey, ez, color='#22a6b3', alpha=0.3, edgecolor='#130cb7', linewidth=0.3)

            # Plot Simulated 3D Trajectory Vector path line
            self.ax.plot(x_coords, y_coords, z_coords, color="#badc58", linewidth=2.5, label="Satellite Path Vector")
            
            # Draw tracking point at final coordinates
            self.ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color="red", s=40, label="Final Satellite Node")

            # Format labels, dynamic zoom boundary and lock box aspect ratio
            self.ax.set_title("Real-Time 3D Orbital Trajectory Engine (RK4 Hybrid)", color="white", fontsize=12, weight="bold")
            max_lim = max(np.max(np.abs(x_coords)), 10000) 
            self.ax.set_xlim(-max_lim, max_lim)
            self.ax.set_ylim(-max_lim, max_lim)
            self.ax.set_zlim(-max_lim, max_lim)
            self.ax.set_box_aspect((1, 1, 1)) # Keeps the Earth perfectly round during re-renders
            self.ax.legend(facecolor="#2b2b2b", edgecolor="none", labelcolor="white")

            # Finalize Render on Canvas
            self.canvas.draw()
            self.lbl_status.configure(text="✅ 3D Render Complete", text_color="green")

        except ValueError:
            self.lbl_status.configure(text="❌ Error: Invalid numerical inputs", text_color="red")
        except Exception as e:
            self.lbl_status.configure(text="❌ Rendering Error", text_color="red")
            print(f"Error: {e}")

if __name__ == "__main__":
    app = AstroSimApp()
    app.mainloop()