# Unsteady-2D-Convection-Diffusion-Solver
Numerical implementation of a 2-D unsteady convection-diffusion solver using the Crank-Nicolson method and Successive Over-Relaxation (SOR) for laminar duct flow.

## Project Overview
This repository contains the mathematical formulation and numerical implementation of a solver for the 2-D unsteady convection-diffusion equation. The project models temperature distribution in a 2-D duct with a steady, laminar velocity field, investigating how thermal gradients develop under both steady and sinusoidal inlet conditions.

## Mathematical Formulation
The governing equation solved is the unsteady 2-D convection-diffusion equation for temperature:

$$\frac{\partial T}{\partial t} + u(y)\frac{\partial T}{\partial x} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$$

### Domain and Physics
* **Domain:** $0 \le x \le 10$ and $-1 \le y \le 1$.
* **Thermal Diffusivity ($\alpha$):** 0.01.
* **Velocity Field:** Fully developed laminar flow defined by $u(y) = u_{max}(1 - y^2)$, where $u_{max} = 1.5$.



## Numerical Implementation
The solver utilizes a Finite Difference Method (FDM) approach with the following specifications:

### Temporal Discretization
* **Crank-Nicolson (Trapezoidal) Method:** Employed for second-order temporal accuracy and improved stability compared to explicit methods. This transforms the PDE into a linear system of the form $a_p T_p = \Sigma a_{nbr}$.



### Linear System Solver
* **Successive Over-Relaxation (SOR):** Used to iteratively solve the resulting algebraic system.
* **Convergence:** The algorithm monitors residuals to ensure the energy balance is maintained within strict tolerances.

### Boundary Conditions
* **Inlet (x=0):** Dirichlet condition with a parabolic profile and optional sinusoidal time-variation: $T(0,y,t) = (1-y^2)^2 [1 + A \sin(2\pi f t)]$.
* **Bottom Wall (y=-1):** Dirichlet condition ($T=0$).
* **Top Wall (y=1):** Neumann condition (Adiabatic, $\partial T / \partial y = 0$).
* **Outlet (x=L):** Neumann condition ($\partial T / \partial x = 0$).

## Validation and Performance
The solver was validated through a global energy balance analysis. By comparing the rate of change of total heat content ($dQ/dt$) against the net heat flux (Inlet - Outlet - Wall Loss), the model achieved high physical plausibility.

* **Steady State Results:** Captured the expected thermal development where the hot inlet fluid is cooled by the bottom wall.
* **Unsteady Perturbations:** Successfully modeled the downstream smoothing of inlet temperature oscillations, demonstrating the diffusive nature of the system.

## Repository Contents
* **Joseph_Marsh_2D_Thermal_Solver_Technical_Analysis.pdf:** Detailed report covering the discretization derivation, energy balance equations, and graphical results of the simulation.
* **Unsteady_2D_Thermal_Analysis.py:** Fully documented code solving the thermal analysis, including animated plots. 
