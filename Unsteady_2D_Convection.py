import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


alpha = 0.01      # thermal diffusivity
umax  = 1.5       # max velocity

Lx = 10.0         # domain length in x
y_min, y_max = -1.0, 1.0

Nx = 25        # number of intervals in x 
Ny = 20           # number of intervals in y 

dx = Lx / Nx
dy = (y_max - y_min) / Ny
 
dt = 0.04        # time step
t_final = 20.0     # final time
save_every = 5
Nt = int(t_final / dt)

# Crank–Nicolson coefficients
rx = alpha * dt / dx**2
ry = alpha * dt / dy**2

# SOR parameters
omega = 1.3       # relaxation factor
tol = 1e-6        # SOR convergence tolerance
max_iter = 10000  # max SOR iterations per time step

x = np.linspace(0.0, Lx, Nx+1)           # i = 0..Nx
y = np.linspace(y_min, y_max, Ny+1)      # j = 0..Ny

# Velocity profile 
u = umax * (1.0 - y**2)

# c_j for each y-row
c = u * dt / (2.0 * dx)
A = 0.0
f = 1

def inlet_temperature(y_vals, t):
    return (1.0 - y_vals**2)**2 * (1.0 + A * np.sin(2.0*np.pi*f*t))

def apply_boundary_conditions(T, t):

    #Inlet x=0 (Dirichlet)
    T[0, :] = inlet_temperature(y, t)
    #Bottom y = -1 (Dirichlet T=0)
    T[:, 0] = 0.0
    #Top y = +1 (Neumann: dT/dy = 0)
    T[:, Ny] = T[:, Ny-1]
    #Outlet x = Lx (Neumann: dT/dx = 0)
    T[Nx, :] = T[Nx-1, :]
def sor_time_step(T_old, t_old, dt):
    t_new = t_old + dt

    # Start T_new as a copy of T_old
    T_new = T_old.copy()
    # Apply boundary conditions at t_new before SOR iterations
    apply_boundary_conditions(T_new, t_new)
    # SOR iteration
    for k in range(max_iter):
        max_change = 0.0

        apply_boundary_conditions(T_new, t_new)

        # Loop over interior nodes
        for i in range(1, Nx):
            for j in range(1, Ny):

                # coefficients at row j 
                aP = 1.0 + rx + ry
                bW = 0.5 * (rx + c[j])
                bE = 0.5 * (rx - c[j])
                bS = 0.5 * ry
                bN = 0.5 * ry

                # Right-hand side b_P 
                bP = ((1.0 - rx - ry) * T_old[i, j] +
                      bW * T_old[i-1, j] +
                      bE * T_old[i+1, j] +
                      bS * T_old[i, j-1] +
                      bN * T_old[i, j+1])
                # Gauss–Seidel new value using latest T_new neighbors
                T_W = T_new[i-1, j]   # already updated this iteration
                T_E = T_new[i+1, j]   # not yet updated in i, but old iter
                T_S = T_new[i, j-1]   # already updated in this iteration
                T_N = T_new[i, j+1]   # not yet updated in j, but old iter

                T_GS = (bP + bW*T_W + bE*T_E + bS*T_S + bN*T_N) / aP
                # SOR update
                T_new_val = (1.0 - omega) * T_new[i, j] + omega * T_GS
                # Track maximum change for convergence
                diff = abs(T_new_val - T_new[i, j])
                if diff > max_change:
                    max_change = diff

                T_new[i, j] = T_new_val

        # Check SOR convergence
        if max_change < tol:
            break
    else:
        #executes if for loop did not break
        print(f"SOR did not converge at t = {t_new:.3f}")

    apply_boundary_conditions(T_new, t_new)

    return T_new, t_new
T_old = np.zeros((Nx+1, Ny+1))  # T(x>0) = 0 initially

t = 0.0
apply_boundary_conditions(T_old, t)  # enforce inlet, walls, etc. at t=0

def global_heat(T):
    return np.sum(T) * dx * dy

def bottom_heat_loss(T):
    dTdy_bottom = (T[:, 1] - T[:, 0]) / dy   # shape: (Nx+1,)
    q_bottom_x = -alpha * dTdy_bottom        # flux at each x_i
    Q_bottom = np.sum(q_bottom_x) * dx
    return Q_bottom

def inlet_flux(T):
    return np.sum(u * T[0, :]) * dy

def outlet_flux(T):
    return np.sum(u * T[Nx, :]) * dy

# Arrays to store diagnostics
Q_global = np.zeros(Nt+1)
Q_bottom_arr = np.zeros(Nt+1)
Q_inlet_arr = np.zeros(Nt+1)
Q_outlet_arr = np.zeros(Nt+1)
time_arr = np.zeros(Nt+1)

# Initial condition
T_old = np.zeros((Nx+1, Ny+1))
t = 0.0
apply_boundary_conditions(T_old, t)

# Store initial diagnostics
Q_global[0]      = global_heat(T_old)
Q_bottom_arr[0]  = bottom_heat_loss(T_old)
Q_inlet_arr[0]   = inlet_flux(T_old)
Q_outlet_arr[0]  = outlet_flux(T_old)
time_arr[0]      = t

#Storing snapshots for temperature contours (every 2 seconds)
snapshots = []
snapshot_times = []

# Time integration
for n in range(1, Nt+1):
    T_new, t = sor_time_step(T_old, t, dt)

    # Diagnostics at time t
    Q_global[n]      = global_heat(T_new)
    Q_bottom_arr[n]  = bottom_heat_loss(T_new)
    Q_inlet_arr[n]   = inlet_flux(T_new)
    Q_outlet_arr[n]  = outlet_flux(T_new)
    time_arr[n]      = t
    #print(f"n={n:3d}, t={t:6.3f}, "
        #f"Q(t)={Q_global[n]: .6e}, "
        #f"Q_bottom(t)={Q_bottom_arr[n]: .6e}")
    # Prepare for next step
    # Save snapshots for contour plots / animation
    if n % save_every == 0 or n == Nt:
        snapshots.append(T_new.copy())
        snapshot_times.append(t)
    T_old = T_new

dQdt = (Q_global[1:] - Q_global[:-1]) / dt
rhs  = Q_inlet_arr[1:] - Q_bottom_arr[1:] - Q_outlet_arr[1:]
error = dQdt - rhs
print("Max |dQ/dt - RHS| over all steps:", np.max(np.abs(error)))

#global heat vs time
plt.figure()
plt.plot(time_arr, Q_global, label="Global heat Q(t)")
plt.xlabel("Time [s]")
plt.ylabel("Global heat content Q")
plt.title("Global heat content vs time")
plt.grid(True)
plt.legend()
plt.tight_layout()

#Bottom wall heat loss vs time
plt.figure()
plt.plot(time_arr, Q_bottom_arr, label="Bottom wall heat loss Q_bottom(t)", color="tab:red")
plt.xlabel("Time [s]")
plt.ylabel("Bottom wall heat loss Q_bottom")
plt.title("Bottom wall heat loss vs time")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time_arr[1:], dQdt, label="dQ/dt from Q(t)", linestyle="-")
plt.plot(time_arr[1:], rhs,  label="Q_inlet - Q_bottom - Q_outlet", linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Rate of change of global heat")
plt.title("Energy balance check")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()


#Contour Plot of Temperature
X, Y = np.meshgrid(x, y, indexing="ij")  # X[i,j] = x_i, Y[i,j] = y_j
#for k, T_snap in enumerate(snapshots):
#    plt.figure()
#    cs = plt.contourf(X, Y, T_snap, levels=10) 
#    plt.colorbar(cs, label="Temperature")
#    plt.xlabel("x")
#    plt.ylabel("y")
#    plt.title(f"Temperature contours at t = {snapshot_times[k]:.2f} s")
#    plt.tight_layout()
#    plt.show()

fig, ax = plt.subplots()
x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]
# Get global min/max across all snapshots so color scale stays fixed
T_all = np.array(snapshots)        # shape: (n_frames, Nx+1, Ny+1)
vmin = np.min(T_all)
vmax = np.max(T_all)

fig, ax = plt.subplots()

#Setting up the first frame, this took way too long to figure out :C
im = ax.imshow(
    snapshots[0].T,                # transpose because imshow uses (y,x)
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    aspect="auto",
    vmin=vmin,
    vmax=vmax
)

cbar = fig.colorbar(im, ax=ax, label="Temperature")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Temperature Contour Plot Over Time (A={A})")

def update(frame):
    im.set_data(snapshots[frame].T)
    ax.set_title(f"Temperature Contour Plot Over Time (A={A})")
    return [im]

anim = FuncAnimation(
    fig,
    update,
    frames=len(snapshots),
    interval=100,   # milliseconds between frames
    blit=True
)

plt.show()
