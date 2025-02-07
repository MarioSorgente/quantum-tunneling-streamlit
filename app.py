import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components
import matplotlib.patheffects as pe

# =============================================================================
# Custom CSS for a clean, minimalistic (Apple-like) design
# =============================================================================
st.markdown(
    """
    <style>
    /* Main container styling */
    .reportview-container .main .block-container {
        padding: 2rem 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f8f8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use Helvetica/Arial fonts for a modern feel
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 12,
    "axes.edgecolor": "#CCCCCC",
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# Physical Constants and Settings
# =============================================================================
hbar = 1.0545718e-34          # Reduced Planck constant (J·s)
m_e = 9.10938356e-31          # Electron mass (kg)
e_charge = 1.602176634e-19    # Elementary charge (C)
num_electrons = 5             # Number of electrons in the ensemble

# =============================================================================
# QuantumEnsemble Class: Contains Methods for Calculations & Animation
# =============================================================================
class QuantumEnsemble:
    def __init__(self):
        pass

    def transmission(self, E_eV, V0, L_nm):
        """
        Compute the transmission coefficient for a rectangular barrier.
        E_eV and V0 are in eV; L_nm is in nm.
        """
        E = E_eV * e_charge
        V0_j = V0 * e_charge
        L = L_nm * 1e-9
        if E < V0_j:
            kappa = np.sqrt(2 * m_e * (V0_j - E)) / hbar
            T = 1 / (1 + (V0**2 * np.sinh(kappa * L)**2) / (4 * E_eV * (V0 - E_eV + 1e-20)))
        else:
            k = np.sqrt(2 * m_e * (E - V0_j)) / hbar
            T = 1 / (1 + (V0**2 * np.sin(k * L)**2) / (4 * E_eV * (E_eV - V0 + 1e-20)))
        return np.clip(T, 0, 1)

    def create_transmission_plot(self, E, V0, L):
        """
        Generate a plot of transmission probability vs. electron energy.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        E_range = np.linspace(0.1, 10, 200)
        T_vals = [self.transmission(e, V0, L) for e in E_range]
        ax.plot(E_range, T_vals, 'b-', lw=2)
        ax.axvline(E, color='r', ls=':', lw=2, alpha=0.8)
        T_val = self.transmission(E, V0, L)
        ax.text(0.95, 0.9, f"T = {T_val:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=14,
                bbox=dict(facecolor='white', edgecolor='#CCCCCC', alpha=0.8))
        ax.set_title(f"Electron Energy = {E:.2f} eV  |  Barrier Height = {V0:.2f} eV  |  Barrier Width = {L:.2f} nm", fontsize=16)
        ax.set_xlabel("Energy (eV)", fontsize=14)
        ax.set_ylabel("Transmission Probability", fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)
        return fig

    def create_animation(self, E, V0, L):
        """
        Create a 2D animation of electrons approaching a barrier.
        Transmitted electrons (green) continue forward;
        reflected electrons (red) bounce back.
        """
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(8, 2))
        # Set x-axis from -L to 2L (nm)
        ax.set_xlim(-L, 2 * L)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Position (nm)", fontsize=12)
        ax.set_yticks([])

        # Draw the barrier with a subtle shadow effect
        barrier_shadow = plt.Rectangle((0.05, 0.22), L, 0.6, fc='gray', alpha=0.2, ec='none')
        ax.add_patch(barrier_shadow)
        barrier_rect = plt.Rectangle((0, 0.2), L, 0.6, fc='gray', alpha=0.4, ec='#CCCCCC', lw=1.5)
        ax.add_patch(barrier_rect)
        ax.text(L/2, 0.85, f"Barrier: {L:.2f} nm", ha="center", va="bottom", fontsize=12, color="black")

        # Calculate transmission probability and randomly decide outcomes
        T_val = self.transmission(E, V0, L)
        np.random.seed()  # fresh randomness each update
        outcomes = np.random.rand(num_electrons) < T_val
        # Colors: transmitted = green, reflected = red.
        colors = ['#4CAF50' if outcome else '#F44336' for outcome in outcomes]
        y_positions = np.linspace(0.3, 0.7, num_electrons)

        # Define trajectories (in nm) for each electron
        N1 = 25  # approach: from -L to 0
        N2 = 15  # crossing: from 0 to L
        N3 = 25  # final: transmitted (L to 2L) or reflected (0 to -L)
        x_trajectories = []
        for outcome in outcomes:
            approach = np.linspace(-L, 0, N1)
            barrier = np.linspace(0, L, N2)
            if outcome:
                final = np.linspace(L, 2 * L, N3)
            else:
                final = np.linspace(0, -L, N3)
            trajectory = np.concatenate([approach, barrier, final])
            x_trajectories.append(trajectory)

        max_frames = max(len(traj) for traj in x_trajectories)
        electrons = []
        for color in colors:
            point, = ax.plot([], [], 'o', markersize=12, color=color, alpha=0.9)
            # Add a shadow to the electron marker
            point.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
            electrons.append(point)

        def animate(frame):
            for i, (electron, traj) in enumerate(zip(electrons, x_trajectories)):
                x = traj[frame] if frame < len(traj) else traj[-1]
                electron.set_data([x], [y_positions[i]])
            return electrons

        anim = FuncAnimation(fig, animate, frames=max_frames, interval=40, blit=True)
        plt.close(fig)
        return anim.to_jshtml()

# =============================================================================
# Main Function: Set Up the Streamlit App Layout
# =============================================================================
def main():
    st.title("Electron Journey Simulator")
    st.markdown(
        """
        Welcome to the Electron Journey Simulator!
        
        **What’s happening?**  
        Electrons are sent toward a barrier. Some manage to tunnel through (green) while others are reflected (red).  
        
        **How to play:**  
        - Use the sliders in the sidebar to adjust the electron energy and the barrier properties.  
        - The top panel shows how likely the electrons are to pass through.  
        - The bottom panel animates the journey in a fun 2D view.
        """
    )

    # Sidebar: Organized simulation settings
    st.sidebar.markdown("## Simulation Settings")
    E = st.sidebar.slider("Electron Energy (eV)", 0.1, 10.0, 5.0, step=0.1)
    V0 = st.sidebar.slider("Barrier Height (eV)", 0.1, 10.0, 1.0, step=0.1)
    L = st.sidebar.slider("Barrier Width (nm)", 0.1, 5.0, 1.0, step=0.1)

    ensemble = QuantumEnsemble()

    # Top Panel: Transmission Probability Plot
    fig = ensemble.create_transmission_plot(E, V0, L)
    st.pyplot(fig)

    # Bottom Panel: 2D Animation of the Electron Journey
    animation_html = ensemble.create_animation(E, V0, L)
    components.html(animation_html, height=400)

if __name__ == '__main__':
    main()
