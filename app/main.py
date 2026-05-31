import ctypes
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Configuración del tema visual de la interfaz
ctk.set_appearance_mode("System")  # Detecta automáticamente si usas modo oscuro o claro
ctk.set_default_color_theme("blue")

class AstroSimApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚀 AstroSim - Advanced Orbital Simulator")
        self.geometry("1100x650")
        self.resizable(False, False)

        # ---- Intentar cargar el Core de C++ ----
        try:
            self.libc = ctypes.CDLL('./orbital_engine.dll')
        except OSError:
            self.libc = None

        # ==========================================
        # ESTRUCTURA DE LA INTERFAZ (Layout de 2 Columnas)
        # ==========================================
        self.grid_columnconfigure(0, weight=1)  # Panel de Control (Izquierda)
        self.grid_columnconfigure(1, weight=3)  # Gráfica de Ingeniería (Derecha)
        self.grid_rowconfigure(0, weight=1)

        # ---- COLUMNA IZQUIERDA: PANEL DE CONTROL ----
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=15)
        self.sidebar.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Título del Panel
        self.title_label = ctk.CTkLabel(self.sidebar, text="AstroSim Engine v1.0", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        # Input 1: Altura Inicial (en Kilómetros)
        self.lbl_altitude = ctk.CTkLabel(self.sidebar, text="Altitud Inicial (km sobre la Tierra):", anchor="w")
        self.lbl_altitude.pack(fill="x", padx=20, pady=(10, 2))
        self.txt_altitude = ctk.CTkEntry(self.sidebar, placeholder_text="Ej. 400")
        self.txt_altitude.insert(0, "400") # Valor por defecto (Estación Espacial Internacional)
        self.txt_altitude.pack(fill="x", padx=20, pady=5)

        # Input 2: Velocidad Tangencial (en m/s)
        self.lbl_velocity = ctk.CTkLabel(self.sidebar, text="Velocidad Orbital Inicial (m/s):", anchor="w")
        self.lbl_velocity.pack(fill="x", padx=20, pady=(10, 2))
        self.txt_velocity = ctk.CTkEntry(self.sidebar, placeholder_text="Ej. 7660")
        self.txt_velocity.insert(0, "7660")
        self.txt_velocity.pack(fill="x", padx=20, pady=5)

        # Input 3: Tiempo de simulación (Pasos en segundos)
        self.lbl_steps = ctk.CTkLabel(self.sidebar, text="Duración de Simulación (segundos):", anchor="w")
        self.lbl_steps.pack(fill="x", padx=20, pady=(10, 2))
        self.txt_steps = ctk.CTkEntry(self.sidebar, placeholder_text="Ej. 6000")
        self.txt_steps.insert(0, "6000")
        self.txt_steps.pack(fill="x", padx=20, pady=5)

        # Botón de Disparo/Simulación
        self.btn_simulate = ctk.CTkButton(self.sidebar, text="⚡ Ejecutar Simulación C++", font=ctk.CTkFont(weight="bold"), command=self.run_physics_engine)
        self.btn_simulate.pack(fill="x", padx=20, pady=30)

        # Etiqueta de estatus para el cliente
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Sistema Listo • Esperando datos", text_color="gray")
        self.lbl_status.pack(side="bottom", pady=20)

        # ---- COLUMNA DERECHA: PANEL DE LA GRÁFICA ----
        self.plot_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.plot_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        # Inicializar el lienzo gráfico vacío
        self.setup_empty_plot()

    def setup_empty_plot(self):
        """Genera el área donde se dibujará la física orbital"""
        self.fig, self.ax = plt.subplots(figsize=(6, 6), facecolor="#2b2b2b") # Fondo oscuro para Matplotlib
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color="#444444")
        self.ax.set_title("Visor de Trayectoria Orbital")
        
        # Dibujar una Tierra estática de referencia
        earth = plt.Circle((0, 0), 6371, color='#22a6b3', fill=True, alpha=0.3, label="Tierra")
        self.ax.add_patch(earth)
        self.ax.axis('equal')
        self.ax.set_xlim(-10000, 10000)
        self.ax.set_ylim(-10000, 10000)

        # Integrar Matplotlib directamente en la ventana de CustomTkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_physics_engine(self):
        """Toma los inputs del cliente, ejecuta el RK4 en C++ y refresca el mapa"""
        if not self.libc:
            self.lbl_status.configure(text="❌ Error: DLL de C++ no cargada", text_color="red")
            return

        try:
            # 1. Recuperar y procesar datos de la interfaz
            altitude_km = float(self.txt_altitude.get())
            velocity_ms = float(self.txt_velocity.get())
            steps = int(self.txt_steps.get())

            # Conversión física: Radio de la tierra (6,371,000m) + altitud del cliente
            init_x = 0.0
            init_y = 6371000.0 + (altitude_km * 1000) 
            init_vx = velocity_ms
            init_vy = 0.0
            dt = 1.0  # Incremento de tiempo de 1 segundo

            # 2. Reservar memoria C-Types para el retorno de C++
            out_x = (ctypes.c_double * steps)()
            out_y = (ctypes.c_double * steps)()

            self.lbl_status.configure(text="🔄 Calculando en C++...", text_color="orange")
            self.update_idletasks()

            # 3. Llamar al motor de C++ ultra-rápido
            self.libc.simulate_orbit(
                ctypes.c_double(init_x), ctypes.c_double(init_y),
                ctypes.c_double(init_vx), ctypes.c_double(init_vy),
                ctypes.c_double(dt), ctypes.c_int(steps),
                out_x, out_y
            )

            # 4. Procesar coordenadas de metros a kilómetros
            x_coords = np.array(out_x) / 1000
            y_coords = np.array(out_y) / 1000

            # 5. Redibujar la gráfica sin cerrar la ventana
            self.ax.clear()
            self.ax.set_facecolor("#1e1e1e")
            self.ax.grid(True, color="#444444")
            self.ax.tick_params(colors='white')
            
            # Dibujar superficie terrestre y órbita simulada
            earth = plt.Circle((0, 0), 6371, color='#22a6b3', fill=True, alpha=0.5, label="Tierra")
            self.ax.add_patch(earth)
            self.ax.plot(x_coords, y_coords, color="#badc58", linewidth=2, label="Satelite")
            self.ax.plot(x_coords[-1], y_coords[-1], 'ro', label="Posición Final") # Dónde termina

            self.ax.set_title("Simulación en Tiempo Real (RK4 Engine)", color="white")
            self.ax.set_xlabel("Distancia X (km)", color="white")
            self.ax.set_ylabel("Distancia Y (km)", color="white")
            self.ax.axis('equal')
            self.ax.legend(facecolor="#2b2b2b", edgecolor="none", labelcolor="white")

            # Actualizar lienzo de Tkinter
            self.canvas.draw()
            self.lbl_status.configure(text="✅ Órbita Graficada Exitosamente", text_color="green")

        except ValueError:
            self.lbl_status.configure(text="❌ Error: Datos numéricos inválidos", text_color="red")
        except Exception as e:
            self.lbl_status.configure(text=f"❌ Error inesperado: {str(e)[:20]}", text_color="red")

if __name__ == "__main__":
    app = AstroSimApp()
    app.mainloop()