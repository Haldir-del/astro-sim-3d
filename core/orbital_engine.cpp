#include <iostream>
#include <cmath>
#include <vector>

// ---- Physical Constants for Earth-centered Orbit ----
const double G = 6.67430e-11; // Standard Gravitational Constant (m^3 kg^-1 s^-2)
const double M = 5.972e24;    // Mass of Earth (kg)

// Standard State Vector representation in 2D
struct State {
    double x, y, vx, vy;
};

// Computes derivatives for ordinary differential equations of motion
// Based on Newton's Law of Universal Gravitation
State derivatives(State s) {
    // r^3 calculation (distance from Earth center cubed)
    double r3 = std::pow(s.x * s.x + s.y * s.y, 1.5);
    State d;
    d.x = s.vx;  // Change in x is vx
    d.y = s.vy;  // Change in y is vy
    d.vx = -G * M * s.x / r3; // Acceleration x
    d.vy = -G * M * s.y / r3; // Acceleration y
    return d;
}

// ---- Public Interface for Python Integration (via DLL) ----
extern "C" {
    // Computes entire trajectory using high-speed numerical integration
    void simulate_orbit(double init_x, double init_y, double init_vx, double init_vy, 
                        double dt, int steps, double* out_x, double* out_y) {
        
        // Load initial state vector
        State s = {init_x, init_y, init_vx, init_vy};
        
        // Performance-critical numerical integration loop
        for (int i = 0; i < steps; ++i) {
            // Write coordinates back to Python memory space via pointers
            out_x[i] = s.x;
            out_y[i] = s.y;

            // ---- Runge-Kutta 4 (RK4) Method implementation ----
            // K1: Start point slope
            State k1 = derivatives(s);
            
            // K2: Midpoint slope 1
            State s2 = {s.x + 0.5 * dt * k1.x, s.y + 0.5 * dt * k1.y, s.vx + 0.5 * dt * k1.vx, s.vy + 0.5 * dt * k1.vy};
            State k2 = derivatives(s2);
            
            // K3: Midpoint slope 2
            State s3 = {s.x + 0.5 * dt * k2.x, s.y + 0.5 * dt * k2.y, s.vx + 0.5 * dt * k2.vx, s.vy + 0.5 * dt * k2.vy};
            State k3 = derivatives(s3);
            
            // K4: End point slope
            State s4 = {s.x + dt * k3.x, s.y + dt * k3.y, s.vx + dt * k3.vx, s.vy + dt * k3.vy};
            State k4 = derivatives(s4);

            // Compute final weighted average slope and update state
            s.x += (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
            s.y += (dt / 6.0) * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y);
            s.vx += (dt / 6.0) * (k1.vx + 2.0 * k2.vx + 2.0 * k3.vx + k4.vx);
            s.vy += (dt / 6.0) * (k1.vy + 2.0 * k2.vy + 2.0 * k3.vy + k4.vy);
        }
    }
}