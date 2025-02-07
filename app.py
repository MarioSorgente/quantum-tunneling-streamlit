import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components

# Physical constants
hbar = 1.0545718e-34          # reduced Planck constant (J s)
m_e = 9.10938356e-31          # electron mass (kg)
e_charge = 1.602176634e-19    # elementary charge (C)
num_electrons = 5             # number of electrons in the ensemble

class QuantumEnsemble:
    def __init__(self):
        pass

    def transmission(self, E_eV, V0, L_nm):
        """
        Calculate the transmission coefficient for a rectangular barrier.
        E_eV and V0 are in eV; L_nm is in nm.
        """
        E = E_eV * e_charge
        V0_j = V0 * e_charge
        L = L_nm * 1e-9  # convert nm to meters
        if E < V0_j:
            kappa = np.sqrt(2 * m_e * (V0_j - E)) / hbar
            T = 1 / (1 + (V0**2 * np.sinh(kappa * L)**2) / (4 * E_eV * (V0 - E_eV + 1e-20)))
        else:
            k = np.sqrt(2 * m_e * (E - V0_j)) / hbar
            T = 1 / (1 + (V0**2 * np.sin(k * L)**2) / (4 * E_eV * (E_eV - V0 + 1e-20)))
        return np.clip(T, 0, 1)
    
    def create_transmission_plot(self, E, V0, L):
        """
        Create a Matplotlib figure of transmission probability versus energy.
        """
        fig, ax = plt.subplots(figsize=(8,5))
        E_range = np.linspace(0.1, 10, 200)
        T_vals = [self.transmission(e, V0, L) for e in E_range]
        ax.plot(E_range, T_vals, 'b-', lw=2)
        ax.axvline(E, color='r', ls=':', alpha=0.8)
        T_val = self.transmission(E, V0, L)
        ax.text(0.95, 0.9, f'T = {T_val:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.set_title(f'E = {E:.2f} eV | V₀ = {V0:.2f} eV | L = {L:.2f} nm')
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Transmission Probability")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)
        return fig

    def create_animation(self, E, V0, L):
        """
        Create an animation of the quantum ensemble.
        The x-axis is in nm. The barrier is drawn from x = 0 to x = L.
        Transmitted electrons (random outcome based on T) move from L to 2L (green),
        while reflected electrons return from 0 to -L (red).
        """
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(8, 2))
        # Set x-axis: from -L to 2L (nm)
        ax.set_xlim(-L, 2 * L)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Position (nm)")
        ax.set_yticks([])

        # Draw the potential barrier as a rectangle from 0 to L.
        barrier_rect = plt.Rectangle((0, 0.2), L, 0.6, fc='gray', alpha=0.4, ec='k', lw=1)
        ax.add_patch(barrier_rect)

        # Calculate transmission probability T.
        T_val = self.transmission(E, V0, L)
        # For reproducibility, use fresh randomness each update.
        np.random.seed()
        outcomes = np.random.rand(num_electrons) < T_val
        # Green (#4CAF50) for transmitted, red (#F44336) for reflected.
        colors = ['#4CAF50' if outcome else '#F44336' for outcome in outcomes]
        y_positions = np.linspace(0.3, 0.7, num_electrons)

        # Define trajectory segments (all distances in nm)
        N1 = 25  # approach: from -L to 0
        N2 = 15  # barrier crossing: from 0 to L
        N3 = 25  # final: transmitted electrons from L to 2L; reflected electrons from 0 to -L
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
        electrons = [ax.plot([], [], 'o', markersize=12, color=color, alpha=0.8)[0]
                     for color in colors]

        def animate(frame):
            for i, (electron, traj) in enumerate(zip(electrons, x_trajectories)):
                x = traj[frame] if frame < len(traj) else traj[-1]
                electron.set_data([x], [y_positions[i]])
            return electrons

        anim = FuncAnimation(fig, animate, frames=max_frames, interval=40, blit=True)
        plt.close(fig)
        # Return the animation as HTML using to_jshtml
        return anim.to_jshtml()

def main():
    st.title("Quantum Tunneling Simulation")
    st.markdown(
        "This simulation shows quantum tunneling through a rectangular barrier. "
        "Use the sidebar sliders to adjust the electron energy, barrier height, and width. "
        "The top panel shows the theoretical transmission probability vs. energy. "
        "The bottom panel animates a small ensemble of electrons, colored green if transmitted and red if reflected."
    )

    # Sidebar controls
    E = st.sidebar.slider("Electron Energy (eV)", 0.1, 10.0, 5.0, step=0.1)
    V0 = st.sidebar.slider("Barrier Height V₀ (eV)", 0.1, 10.0, 1.0, step=0.1)
    L = st.sidebar.slider("Barrier Width L (nm)", 0.1, 5.0, 1.0, step=0.1)

    ensemble = QuantumEnsemble()

    # Display the transmission probability plot.
    fig = ensemble.create_transmission_plot(E, V0, L)
    st.pyplot(fig)

    # Display the animation.
    animation_html = ensemble.create_animation(E, V0, L)
    st.components.v1.html(animation_html, height=400)

if __name__ == '__main__':
    main()
