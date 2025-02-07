import streamlit as st
import numpy as np
import plotly.graph_objects as go

# =============================================================================
# Custom CSS for a Clean, Minimalistic (Apple-Like) Look
# =============================================================================
st.markdown(
    """
    <style>
    /* Main container */
    .reportview-container .main .block-container {
        padding: 2rem 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f2f2f2;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Physical Constants and Simulation Settings
# =============================================================================
hbar = 1.0545718e-34          # Reduced Planck constant (J·s)
m_e = 9.10938356e-31          # Electron mass (kg)
e_charge = 1.602176634e-19    # Elementary charge (C)
num_electrons = 5             # Number of electrons in the ensemble

# =============================================================================
# Transmission Calculation Function
# =============================================================================
def transmission(E_eV, V0, L_nm):
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

# =============================================================================
# Trajectory Simulation in 3D
# =============================================================================
def simulate_trajectories(E, V0, L, n_frames=50, n1=20, n2=10, n3=20):
    """
    Simulate 3D trajectories for a small ensemble of electrons.
    
    The simulation is divided into three phases:
      - Phase 1 (frames 0 to n1): Brownian motion starting at x = -2*L.
      - Phase 2 (frames n1 to n1+n2): Drift toward the barrier (from x = -0.5*L to x = 0).
      - Phase 3 (frames n1+n2 to n_frames): After the barrier, electrons either
         tunnel (transmitted: from 0 to 2*L) or are reflected (from 0 to -L).
         
    Returns a list of trajectories (one per electron), a list of outcomes (True for transmitted),
    and the calculated transmission probability.
    """
    T_val = transmission(E, V0, L)
    outcomes = np.random.rand(num_electrons) < T_val
    trajectories = []
    for i in range(num_electrons):
        traj = np.zeros((n_frames, 3))  # columns: x, y, z (all in nm)
        # Phase 1: Brownian motion (from x = -2L to -0.5L)
        x_start = -2 * L
        x_a = -0.5 * L
        for t in range(n1):
            x = x_start + (x_a - x_start) * t / (n1 - 1) + np.random.normal(0, 0.1 * L)
            y = np.random.normal(0, 0.2)
            z = np.random.normal(0, 0.2)
            traj[t] = [x, y, z]
        # Phase 2: Approach barrier (from -0.5L to 0)
        for t in range(n1, n1 + n2):
            x = x_a + (0 - x_a) * (t - n1) / (n2 - 1) + np.random.normal(0, 0.05 * L)
            y = np.random.normal(0, 0.1)
            z = np.random.normal(0, 0.1)
            traj[t] = [x, y, z]
        # Phase 3: After barrier
        if outcomes[i]:
            # Transmitted: move from 0 to 2L
            for t in range(n1 + n2, n_frames):
                x = 0 + (2 * L - 0) * (t - (n1 + n2)) / (n3 - 1) + np.random.normal(0, 0.1 * L)
                y = np.random.normal(0, 0.1)
                z = np.random.normal(0, 0.1)
                traj[t] = [x, y, z]
        else:
            # Reflected: move from 0 to -L
            for t in range(n1 + n2, n_frames):
                x = 0 - (L - 0) * (t - (n1 + n2)) / (n3 - 1) + np.random.normal(0, 0.1 * L)
                y = np.random.normal(0, 0.1)
                z = np.random.normal(0, 0.1)
                traj[t] = [x, y, z]
        trajectories.append(traj)
    return trajectories, outcomes, T_val

# =============================================================================
# Create 3D Animation Figure with Plotly
# =============================================================================
def create_3d_figure(E, V0, L):
    """
    Create a 3D animated Plotly figure showing electron trajectories and the barrier.
    """
    n_frames = 50
    trajectories, outcomes, T_val = simulate_trajectories(E, V0, L, n_frames=n_frames)
    
    # Create one trace per electron (each trace is a single moving point)
    traces = []
    for i in range(num_electrons):
        color = '#4CAF50' if outcomes[i] else '#F44336'
        traj = trajectories[i]
        trace = go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode='markers',
            marker=dict(size=8, color=color, opacity=0.95),
            name=f'Electron {i+1}'
        )
        traces.append(trace)
    
    # Build animation frames: one frame per time step
    frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        for i in range(num_electrons):
            traj = trajectories[i]
            frame_data.append(dict(
                x=[traj[frame_idx, 0]],
                y=[traj[frame_idx, 1]],
                z=[traj[frame_idx, 2]]
            ))
        frames.append(dict(data=frame_data, name=str(frame_idx)))
    
    # Create a 3D barrier as a cuboid (using Mesh3d)
    # Barrier vertices (cuboid from x=0 to x=L, y from -1 to 1, z from -1 to 1)
    X = [0, 0, 0, 0, L, L, L, L]
    Y = [-1, -1, 1, 1, -1, -1, 1, 1]
    Z = [-1, 1, -1, 1, -1, 1, -1, 1]
    # Define the faces via indices (two triangles per face)
    i_faces = [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 6, 7]
    j_faces = [1, 2, 4, 2, 4, 3, 5, 6, 6, 7, 5, 5]
    k_faces = [2, 4, 5, 4, 5, 7, 6, 7, 7, 7, 5, 3]
    barrier_mesh = go.Mesh3d(
        x=X,
        y=Y,
        z=Z,
        color='gray',
        opacity=0.3,
        flatshading=True,
        i=i_faces,
        j=j_faces,
        k=k_faces,
        name='Barrier'
    )
    
    # Create base figure with the electron traces and barrier
    fig = go.Figure(
        data=traces + [barrier_mesh],
        layout=go.Layout(
            title=f"3D Electron Journey (E = {E:.2f} eV, V₀ = {V0:.2f} eV, L = {L:.2f} nm, T = {T_val:.3f})",
            scene=dict(
                xaxis=dict(title="X Position (nm)", backgroundcolor="white", gridcolor="#E5E5E5"),
                yaxis=dict(title="Y Position (nm)", backgroundcolor="white", gridcolor="#E5E5E5"),
                zaxis=dict(title="Z Position (nm)", backgroundcolor="white", gridcolor="#E5E5E5"),
                aspectmode='data'
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1,
                x=1.05,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 80, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}])]
            )]
        )
    )
    
    fig.frames = frames
    # (Optional) Add a slider to manually control the frame.
    sliders = [dict(
        steps=[dict(method="animate",
                    args=[[str(k)],
                          {"frame": {"duration": 80, "redraw": True},
                           "mode": "immediate",
                           "transition": {"duration": 0}}],
                    label=str(k))
               for k in range(n_frames)],
        active=0,
        transition={"duration": 0},
        x=0,  # slider starting position  
        y=0,  
        currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor='center'),
        len=1.0
    )]
    fig.update_layout(sliders=sliders)
    return fig

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.title("3D Electron Journey Simulator")
    st.markdown(
        """
        **Welcome!**  
        Watch as electrons—displayed in a modern 3D view with natural Brownian motion—approach a barrier.
        Some tunnel through (green) while others are reflected (red).
        
        **How to Play:**  
        Use the sliders in the sidebar to adjust the electron energy, barrier height, and barrier width.
        The barrier is shown as a translucent block, and the simulation updates in real time.
        """
    )
    
    # Sidebar: Simulation Settings (no overlap in parameters)
    st.sidebar.markdown("### Simulation Settings")
    E = st.sidebar.slider("Electron Energy (eV)", 0.1, 10.0, 5.0, step=0.1)
    V0 = st.sidebar.slider("Barrier Height (eV)", 0.1, 10.0, 1.0, step=0.1)
    L = st.sidebar.slider("Barrier Width (nm)", 0.1, 5.0, 1.0, step=0.1)
    
    # Generate and display the 3D animated figure
    fig = create_3d_figure(E, V0, L)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
