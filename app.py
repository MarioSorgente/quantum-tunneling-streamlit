import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --------------------------
# Page configuration & Styling
# --------------------------
st.set_page_config(page_title="3D Quantum Simulator", layout="wide")

# Custom CSS for an Apple-inspired aesthetic
st.markdown(
    """
    <style>
    /* Overall page styling */
    body {
        background-color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .reportview-container {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 10px;
    }
    /* Custom slider styling */
    div.stSlider > div {
        background: transparent;
    }
    div.stSlider input[type=range] {
        -webkit-appearance: none;
        width: 100%;
        height: 6px;
        border-radius: 5px;
        background: #ddd;
        outline: none;
    }
    div.stSlider input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #007aff;
        cursor: pointer;
        border: none;
    }
    /* Transmission coefficient card styling */
    .transmission-card {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 10px;
        font-size: 0.9em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Sidebar: Simulation Parameters
# --------------------------
st.sidebar.title("Simulation Parameters")
energy = st.sidebar.slider("Electron Energy (eV)", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
V0 = st.sidebar.slider("Barrier Height (eV)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
L = st.sidebar.slider("Barrier Width (nm)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# --------------------------
# Transmission Coefficient Calculation
# --------------------------
@st.cache_data
def compute_transmission(E, V0, L):
    """
    Computes the transmission coefficient T for a rectangular potential barrier.
    E: electron energy in eV
    V0: barrier height in eV
    L: barrier width in nm
    """
    # Physical constants
    m = 9.10938356e-31       # Electron mass in kg
    hbar = 1.0545718e-34     # Reduced Planck's constant in J*s
    eV_to_J = 1.602176634e-19  # 1 eV in Joules

    # Convert units: energy in Joules, barrier width in meters
    E_J = E * eV_to_J
    V0_J = V0 * eV_to_J
    L_m = L * 1e-9

    if E < V0:
        # Tunneling: exponential decay through the barrier
        kappa = np.sqrt(2 * m * (V0_J - E_J)) / hbar
        T = np.exp(-2 * kappa * L_m)
    else:
        # Over-the-barrier transmission:
        # Avoid division by zero if E and V0 are almost equal.
        if abs(E_J - V0_J) < 1e-12:
            T = 1.0
        else:
            k = np.sqrt(2 * m * (E_J - V0_J)) / hbar
            T = 1 / (1 + (V0_J**2 * (np.sin(k * L_m))**2 / (4 * E_J * (E_J - V0_J))))
    return np.clip(T, 0, 1)

T_value = compute_transmission(energy, V0, L)

# Display the transmission coefficient in a stylish card on the sidebar
st.sidebar.markdown(
    f"""
    <div class="transmission-card">
        <strong>Transmission Coefficient (T):</strong>
        <p>{T_value:.3f}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Transmission vs Energy Plot
# --------------------------
@st.cache_data
def compute_transmission_vs_energy(V0, L):
    energies = np.linspace(0.1, 10.0, 200)
    T_values = np.array([compute_transmission(E, V0, L) for E in energies])
    return energies, T_values

def transmission_vs_energy_plot(V0, L):
    energies, T_values = compute_transmission_vs_energy(V0, L)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=T_values,
            mode="lines",
            line=dict(color="green", width=3),
            name="T vs Energy",
        )
    )
    fig.update_layout(
        title="Transmission Coefficient vs Electron Energy",
        xaxis_title="Electron Energy (eV)",
        yaxis_title="Transmission Coefficient",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig

# --------------------------
# 3D Simulation of Quantum Tunneling
# --------------------------
@st.cache_data
def simulate_trajectories(energy, V0, L, n_electrons=5, frames_phase1=20, frames_phase2=20, frames_phase3=20):
    """
    Simulate electron trajectories in three phases:
      - Approach Phase: from x = -2 to 0
      - Barrier Crossing Phase: from x = 0 to L
      - After-Barrier Phase: transmitted electrons go from L to 4; reflected electrons go from 0 to -2.
    Returns a list of trajectory dictionaries (one per electron) and the total number of animation frames.
    """
    total_frames = frames_phase1 + frames_phase2 + frames_phase3
    trajectories = []
    # Set seed for reproducibility
    np.random.seed(42)
    T_sim = compute_transmission(energy, V0, L)

    for i in range(n_electrons):
        # Unique base offsets for y and z so trajectories do not overlap
        base_y = np.random.uniform(-0.5, 0.5)
        base_z = np.random.uniform(-0.5, 0.5)

        # Approach Phase: x from -2 to 0 with slight noise
        x1 = np.linspace(-2, 0, frames_phase1) + np.random.normal(0, 0.05, frames_phase1)
        y1 = base_y + np.random.normal(0, 0.02, frames_phase1)
        z1 = base_z + np.random.normal(0, 0.02, frames_phase1)

        # Barrier Crossing Phase: x from 0 to L with slight noise
        x2 = np.linspace(0, L, frames_phase2) + np.random.normal(0, 0.05, frames_phase2)
        y2 = base_y + np.random.normal(0, 0.02, frames_phase2)
        z2 = base_z + np.random.normal(0, 0.02, frames_phase2)

        # After-Barrier Phase: decide outcome based on transmission coefficient T
        if np.random.rand() < T_sim:
            # Transmitted: continue from L to 4
            x3 = np.linspace(L, 4, frames_phase3) + np.random.normal(0, 0.05, frames_phase3)
        else:
            # Reflected: return from 0 to -2
            x3 = np.linspace(0, -2, frames_phase3) + np.random.normal(0, 0.05, frames_phase3)
        y3 = base_y + np.random.normal(0, 0.02, frames_phase3)
        z3 = base_z + np.random.normal(0, 0.02, frames_phase3)

        # Concatenate phases
        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])
        z = np.concatenate([z1, z2, z3])

        trajectories.append({"x": x, "y": y, "z": z})
    return trajectories, total_frames

trajectories, total_frames = simulate_trajectories(energy, V0, L)

def build_3d_animation(trajectories, total_frames, V0, L, n_electrons=5):
    """
    Builds a Plotly 3D animation figure showing electron trajectories and the barrier.
    Each electron displays a fading trail (using RGBA colors for lower opacity) and a fully opaque current marker.
    The barrier is drawn as a brown Mesh3d trace.
    """
    # Barrier dimensions (using simulation units):
    # x: from 0 to L, y: height scales with V0 (min 0.5, scaling factor 0.3), z: fixed width = 1
    barrier_height = max(0.5, V0 * 0.3)
    x_barrier = [0, L, L, 0, 0, L, L, 0]
    y_barrier = [
        -barrier_height / 2,
        -barrier_height / 2,
        barrier_height / 2,
        barrier_height / 2,
        -barrier_height / 2,
        -barrier_height / 2,
        barrier_height / 2,
        barrier_height / 2,
    ]
    z_barrier = [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]
    # Define faces for the barrier (each face as two triangles)
    faces = [
        [0, 1, 2], [0, 2, 3],  # back face (z = -0.5)
        [4, 5, 6], [4, 6, 7],  # front face (z = 0.5)
        [0, 1, 5], [0, 5, 4],  # bottom face (y = -barrier_height/2)
        [2, 3, 7], [2, 7, 6],  # top face (y = barrier_height/2)
        [1, 2, 6], [1, 6, 5],  # right face (x = L)
        [0, 3, 7], [0, 7, 4],  # left face (x = 0)
    ]
    i_faces = [face[0] for face in faces]
    j_faces = [face[1] for face in faces]
    k_faces = [face[2] for face in faces]

    # Barrier trace (static)
    barrier_trace = go.Mesh3d(
        x=x_barrier,
        y=y_barrier,
        z=z_barrier,
        i=i_faces,
        j=j_faces,
        k=k_faces,
        color="saddlebrown",
        opacity=0.5,
        name="Barrier",
        showscale=False,
    )

    # Prepare initial traces for electrons:
    # For each electron, we create two traces: one for the trail (line) and one for the current position (marker)
    data = [barrier_trace]
    for traj in trajectories:
        # Trail trace (starts with the first point) using RGBA for faded blue color.
        data.append(
            go.Scatter3d(
                x=[traj["x"][0]],
                y=[traj["y"][0]],
                z=[traj["z"][0]],
                mode="lines",
                line=dict(color="rgba(0,0,255,0.3)", width=4),
                showlegend=False,
            )
        )
        # Current position trace (marker)
        data.append(
            go.Scatter3d(
                x=[traj["x"][0]],
                y=[traj["y"][0]],
                z=[traj["z"][0]],
                mode="markers",
                marker=dict(color="blue", size=6),
                showlegend=False,
            )
        )

    # Create animation frames
    frames = []
    for frame_idx in range(total_frames):
        frame_data = []
        # For each electron, update the trail and current marker traces.
        for traj in trajectories:
            # Trail: all positions up to the current frame index.
            x_trail = traj["x"][: frame_idx + 1]
            y_trail = traj["y"][: frame_idx + 1]
            z_trail = traj["z"][: frame_idx + 1]
            trail_trace = go.Scatter3d(
                x=x_trail,
                y=y_trail,
                z=z_trail,
                mode="lines",
                line=dict(color="rgba(0,0,255,0.3)", width=4),
                showlegend=False,
            )
            # Current position: the latest position.
            current_trace = go.Scatter3d(
                x=[traj["x"][frame_idx]],
                y=[traj["y"][frame_idx]],
                z=[traj["z"][frame_idx]],
                mode="markers",
                marker=dict(color="blue", size=6),
                showlegend=False,
            )
            frame_data.append(trail_trace)
            frame_data.append(current_trace)
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Build the complete figure with animation controls.
    fig = go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title="3D Quantum Tunneling Simulation",
            scene=dict(
                xaxis=dict(title="X", range=[-3, 5]),
                yaxis=dict(title="Y", range=[-2, 2]),
                zaxis=dict(title="Z", range=[-2, 2]),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        )
                    ],
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top",
                )
            ],
            sliders=[
                {
                    "currentvalue": {"prefix": "Frame: "},
                    "steps": [
                        {
                            "method": "animate",
                            "label": str(k),
                            "args": [
                                [str(k)],
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        }
                        for k in range(total_frames)
                    ],
                    "x": 0.1,
                    "y": -0.05,
                    "len": 0.8,
                }
            ],
        ),
    )
    return fig

fig_3d = build_3d_animation(trajectories, total_frames, V0, L)

# --------------------------
# Main Page: Display Plots and Animation
# --------------------------
st.title("3D Quantum Simulator")
st.markdown("### Transmission Coefficient vs Electron Energy")
st.plotly_chart(transmission_vs_energy_plot(V0, L), use_container_width=True)

st.markdown("### 3D Quantum Tunneling Simulation")
st.plotly_chart(fig_3d, use_container_width=True)
