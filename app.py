import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="3D Quantum Simulator", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f7;}
    .sidebar .sidebar-content {background-color: #ffffff; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .stSlider [data-baseweb="slider"] {padding: 0px 15px;}
    .stSlider div[data-baseweb="slider-track"] {background-color: #e0e0e0; height: 6px;}
    .stSlider div[data-baseweb="slider-track"] > div {background-color: #007AFF; height: 6px;}
    .stSlider div[data-baseweb="thumb"] {background-color: #007AFF; border: none; width: 16px; height: 16px;}
    .stSlider div[data-baseweb="thumb"]:focus {box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.2);}
    .card {background: white; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .card h3 {font-size: 1.1rem; margin: 0 0 8px 0; color: #1d1d1f;}
    .card p {font-size: 1.5rem; margin: 0; font-weight: bold; color: #007AFF;}
    </style>
""", unsafe_allow_html=True)

# Constants
m = 9.10938356e-31  # Electron mass (kg)
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
e_charge = 1.602176634e-19  # Elementary charge (C)
nm_to_m = 1e-9

@st.cache_data
def calculate_transmission(E_eV, V0_eV, L_nm):
    E = E_eV * e_charge
    V0 = V0_eV * e_charge
    L = L_nm * nm_to_m
    
    if E >= V0:
        k1 = np.sqrt(2*m*E)/hbar
        k2 = np.sqrt(2*m*(E - V0))/hbar
        T = 1 / (1 + ((k1**2 - k2**2)/(2*k1*k2))**2 * np.sin(k2*L)**2)
    else:
        kappa = np.sqrt(2*m*(V0 - E))/hbar
        T = 1 / (1 + (V0**2 * np.sinh(kappa*L)**2)/(4*E*(V0 - E)))
    
    return np.clip(T, 0.0, 1.0)

@st.cache_data
def simulate_trajectories(E, V0, L, T, num_electrons=5):
    trajectories = []
    for _ in range(num_electrons):
        transmitted = np.random.rand() < T
        x = np.linspace(-2, 4, 100)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        
        # Initial positions with offsets
        y_offset = np.random.uniform(-0.2, 0.2)
        z_offset = np.random.uniform(-0.2, 0.2)
        
        for i in range(len(x)):
            if x[i] < 0:
                y[i] = y_offset + 0.1 * np.random.randn()
                z[i] = z_offset + 0.1 * np.random.randn()
            elif 0 <= x[i] <= L:
                y[i] = y_offset + 0.1 * np.random.randn()
                z[i] = z_offset + 0.1 * np.random.randn()
                x[i] = x[i] * L  # Scale barrier width
            else:
                if transmitted:
                    y[i] = y_offset + 0.1 * np.random.randn()
                    z[i] = z_offset + 0.1 * np.random.randn()
                else:
                    x[i] = -x[i]  # Reflect
                    y[i] = y_offset + 0.1 * np.random.randn()
                    z[i] = z_offset + 0.1 * np.random.randn()
        
        trajectories.append((x, y, z))
    return trajectories

# Sidebar controls
with st.sidebar:
    st.markdown("## Parameters")
    E = st.slider("Electron Energy (eV)", 0.1, 10.0, 5.0, 0.1)
    V0 = st.slider("Barrier Height (eV)", 0.1, 10.0, 1.0, 0.1)
    L = st.slider("Barrier Width (nm)", 0.1, 5.0, 1.0, 0.1)
    
    T = calculate_transmission(E, V0, L)
    st.markdown(f'<div class="card"><h3>Transmission Coefficient</h3><p>{T:.3f}</p></div>', 
                unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    # T vs Energy plot
    energies = np.linspace(0.1, 10.0, 100)
    Ts = [calculate_transmission(e, V0, L) for e in energies]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energies, y=Ts, line=dict(color='#34C759', width=3)))
    fig.update_layout(
        title="Transmission Coefficient vs Energy",
        xaxis_title="Energy (eV)",
        yaxis_title="T",
        template="plotly_white",
        margin=dict(t=40, b=40),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 3D Visualization
    trajectories = simulate_trajectories(E, V0, L, T)
    
    fig = go.Figure()
    
    # Add barrier
    barrier_height = max(0.5, V0 * 0.3)
    barrier = go.Mesh3d(
        x=[0, 0, L, L, 0, 0, L, L],
        y=[-barrier_height, -barrier_height, -barrier_height, -barrier_height,
           barrier_height, barrier_height, barrier_height, barrier_height],
        z=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
        i=[7, 0, 0, 0, 4, 4, 6, 6],
        j=[3, 4, 1, 2, 5, 6, 5, 2],
        k=[0, 7, 2, 3, 6, 7, 1, 1],
        color='#A52A2A',
        opacity=0.3,
        flatshading=True
    )
    fig.add_trace(barrier)
    
    # Add electron trajectories with fading trails
    for idx, (x, y, z) in enumerate(trajectories):
        for i in range(1, len(x)):
            alpha = 0.1 + 0.9*(i/len(x))
            fig.add_trace(go.Scatter3d(
                x=x[:i], y=y[:i], z=z[:i],
                mode='lines',
                line=dict(
                    color=f'rgba(0, 122, 255, {alpha})',  # Using RGBA for opacity
                    width=3
                ),
                showlegend=False
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (nm)', range=[-2, 4]),
            yaxis=dict(title='Y', range=[-1, 1]),
            zaxis=dict(title='Z', range=[-1, 1]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None]),
                    ],
            x=0.1, y=0,
            pad=dict(t=40, r=10)
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)
