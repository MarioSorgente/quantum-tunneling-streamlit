import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuration
num_electrons = 5
n_frames = 45  # Total number of animation frames
electron_size = 8

# Physical constants
hbar = 1.0545718e-34      # Reduced Planck constant (J·s)
m_e = 9.10938356e-31      # Electron mass (kg)
e_charge = 1.602176634e-19  # Elementary charge (C)

# --- Caching for performance ---
@st.cache_data(show_spinner=False)
def calculate_T(E, V0, L):
    """Calculate transmission coefficient with numerical stability."""
    E_j = E * e_charge
    V0_j = V0 * e_charge
    L_m = L * 1e-9
    if E_j < V0_j:
        kappa = np.sqrt(2 * m_e * (V0_j - E_j + 1e-30)) / hbar
        T = 1 / (1 + (V0**2 * np.sinh(kappa * L_m)**2) / (4 * E * (V0 - E + 1e-30)))
    else:
        k = np.sqrt(2 * m_e * (E_j - V0_j + 1e-30)) / hbar
        T = 1 / (1 + (V0**2 * np.sin(k * L_m)**2) / (4 * E * (E - V0 + 1e-30)))
    return np.clip(T, 0, 1)

@st.cache_data(show_spinner=False, max_entries=3)
def simulate_trajectories(E, V0, L, _n_frames=n_frames):
    """
    Generate 3-phase 3D trajectories for electrons.
    
    Phase 1: Approach from x=-2 to 0.
    Phase 2: Crossing the barrier from x=0 to x=L.
    Phase 3: After the barrier:
       - If transmitted: from x=L to 4.
       - If reflected: from x=0 to -2.
    """
    T = calculate_T(E, V0, L)
    outcomes = np.random.rand(num_electrons) < T  # True means transmitted
    trajectories = []
    p1 = int(_n_frames * 1/3)  # e.g. 15 frames for approach
    p2 = int(_n_frames * 2/3)  # e.g. 30 frames for barrier crossing
    for outcome in outcomes:
        traj = np.zeros((_n_frames, 3))
        # Phase 1: Approach (x from -2 to 0)
        for t in range(p1):
            x = -2 + (0 - (-2)) * (t / (p1 - 1)) + np.random.normal(0, 0.05)
            y = np.random.normal(0, 0.05)
            z = np.random.normal(0, 0.05)
            traj[t] = [x, y, z]
        # Phase 2: Crossing the barrier (x from 0 to L)
        for t in range(p1, p2):
            x = 0 + (L - 0) * ((t - p1) / (p2 - p1)) * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.05)
            y = np.random.normal(0, 0.05)
            z = np.random.normal(0, 0.05)
            traj[t] = [x, y, z]
        # Phase 3: After the barrier
        for t in range(p2, _n_frames):
            if outcome:
                # Transmitted: x from L to 4
                x = L + (4 - L) * ((t - p2) / (_n_frames - p2 - 1)) + np.random.normal(0, 0.05)
            else:
                # Reflected: x from 0 to -2
                x = 0 + (-2 - 0) * ((t - p2) / (_n_frames - p2 - 1)) + np.random.normal(0, 0.05)
            y = np.random.normal(0, 0.05)
            z = np.random.normal(0, 0.05)
            traj[t] = [x, y, z]
        trajectories.append(traj)
    return np.array(trajectories), outcomes, T

def create_probability_plot(E, V0, L):
    """Create a bar chart showing T (green) and R (red) coefficients."""
    T_val = calculate_T(E, V0, L)
    R_val = 1 - T_val
    fig = go.Figure(data=[
        go.Bar(name="Transmission", x=["T"], y=[T_val], marker_color="#4CAF50"),
        go.Bar(name="Reflection", x=["R"], y=[R_val], marker_color="#F44336")
    ])
    fig.update_layout(
        title="Probability Coefficients",
        yaxis=dict(range=[0, 1], title="Probability"),
        barmode="group",
        plot_bgcolor="rgba(240,240,240,0.8)",
        paper_bgcolor="rgba(240,240,240,0.8)",
        font=dict(size=14)
    )
    return fig

def create_3d_figure(E, V0, L):
    """Create a 3D Plotly figure for the quantum simulation with fading trails."""
    trajectories, outcomes, T_val = simulate_trajectories(E, V0, L)
    colors = ['#4CAF50' if o else '#F44336' for o in outcomes]
    
    # --- Define the barrier trace ---
    barrier_height = max(0.5, V0 * 0.3)  # scales with V0 (minimum 0.5)
    barrier_width = 1  # fixed z dimension
    barrier_x = [0, 0, 0, 0, L, L, L, L]
    barrier_y = [-barrier_height, -barrier_height, barrier_height, barrier_height,
                 -barrier_height, -barrier_height, barrier_height, barrier_height]
    barrier_z = [-barrier_width, barrier_width, -barrier_width, barrier_width,
                 -barrier_width, barrier_width, -barrier_width, barrier_width]
    barrier_trace = go.Mesh3d(
        x=barrier_x,
        y=barrier_y,
        z=barrier_z,
        i=[0, 0, 0, 1, 2, 4, 4, 6, 6, 0, 2, 3],
        j=[1, 2, 4, 3, 3, 5, 6, 7, 5, 4, 6, 7],
        k=[2, 4, 5, 2, 7, 7, 5, 3, 7, 5, 4, 1],
        color="#8B4513",  # Brown color
        opacity=0.8,
        name=f'Barrier (L={L}nm, Height={barrier_height:.2f})'
    )
    
    # --- Prepare base traces for electrons: two traces per electron (trail and current position) ---
    base_traces = [barrier_trace]
    for i in range(num_electrons):
        # Trail trace (initially empty)
        trail_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color=colors[i], width=3),
            opacity=0.3,
            showlegend=False
        )
        # Current position trace (marker)
        current_trace = go.Scatter3d(
            x=[trajectories[i][0, 0]],
            y=[trajectories[i][0, 1]],
            z=[trajectories[i][0, 2]],
            mode='markers',
            marker=dict(size=electron_size, color=colors[i]),
            opacity=1.0,
            name=f'Electron {i+1}'
        )
        base_traces.extend([trail_trace, current_trace])
    fig = go.Figure(data=base_traces)
    
    # --- Build animation frames ---
    frames = []
    for k in range(n_frames):
        frame_data = [barrier_trace]  # always include the barrier trace
        for i in range(num_electrons):
            traj = trajectories[i]
            # Trail: points from start up to frame k (if k>0), with fixed low opacity.
            if k > 0:
                trail = go.Scatter3d(
                    x=traj[:k, 0],
                    y=traj[:k, 1],
                    z=traj[:k, 2],
                    mode='lines',
                    line=dict(color=colors[i], width=3),
                    opacity=0.3,
                    showlegend=False
                )
            else:
                trail = go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='lines',
                    line=dict(color=colors[i], width=3),
                    opacity=0.3,
                    showlegend=False
                )
            # Current: the last point of the trajectory at frame k, with full opacity.
            current = go.Scatter3d(
                x=[traj[k, 0]],
                y=[traj[k, 1]],
                z=[traj[k, 2]],
                mode='markers',
                marker=dict(size=electron_size, color=colors[i]),
                opacity=1.0,
                showlegend=False
            )
            frame_data.extend([trail, current])
        frames.append(go.Frame(data=frame_data, name=str(k)))
    fig.frames = frames
    
    # --- Update layout ---
    fig.update_layout(
        title=dict(
            text=f"Quantum Tunneling Simulation<br>E={E:.1f} eV, V₀={V0:.1f} eV, T={T_val:.3f}",
            x=0.05,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title="X (nm)", range=[-2, 4], backgroundcolor='rgba(240,240,240,0.8)'),
            yaxis=dict(title="Y (nm)", range=[-max(2, barrier_height*1.5), max(2, barrier_height*1.5)], backgroundcolor='rgba(240,240,240,0.8)'),
            zaxis=dict(title="Z (nm)", range=[-2, 2], backgroundcolor='rgba(240,240,240,0.8)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.9,
            x=1.1,  # Positioned to avoid overlap with legend.
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[dict(
                label="▶ Play",
                method="animate",
                args=[None, {"frame": {"duration": 70}, "fromcurrent": True}]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[str(k)],
                      {"frame": {"duration": 70, "redraw": True},
                       "mode": "immediate",
                       "transition": {"duration": 0}}],
                label=str(k)
            ) for k in range(n_frames)],
            active=0,
            transition={"duration": 0},
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor='center'),
            len=1.0
        )]
    )
    
    return fig

def main():
    st.set_page_config(page_title="3D Quantum Simulator", layout="wide")
    st.title("3D Quantum Tunneling Visualization")
    
    # --- Probability Plot ---
    st.subheader("Probability Coefficients")
    # Use sidebar parameters for probability plot as well.
    prob_E = st.sidebar.slider("Electron Energy (eV) [Prob.]", 0.1, 10.0, 5.0, 0.1, key="prob_E")
    prob_V0 = st.sidebar.slider("Barrier Height (eV) [Prob.]", 0.1, 10.0, 1.0, 0.1, key="prob_V0")
    prob_L = st.sidebar.slider("Barrier Width (nm) [Prob.]", 0.1, 5.0, 1.0, 0.1, key="prob_L")
    prob_fig = create_probability_plot(prob_E, prob_V0, prob_L)
    st.plotly_chart(prob_fig, use_container_width=True)
    
    # --- Simulation Parameters ---
    with st.sidebar:
        st.header("Simulation Parameters")
        E = st.slider("Electron Energy (eV)", 0.1, 10.0, 5.0, 0.1, key="E")
        V0 = st.slider("Barrier Height (eV)", 0.1, 10.0, 1.0, 0.1, key="V0")
        L = st.slider("Barrier Width (nm)", 0.1, 5.0, 1.0, 0.1, key="L")
    
    with st.spinner('Generating quantum simulation...'):
        fig = create_3d_figure(E, V0, L)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

if __name__ == "__main__":
    main()
