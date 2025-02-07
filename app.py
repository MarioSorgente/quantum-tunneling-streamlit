import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
num_electrons = 5
n_frames = 45  # Reduced for better performance
trail_length = 10
electron_size = 8

# Physical constants
hbar = 1.0545718e-34
m_e = 9.10938356e-31
e_charge = 1.602176634e-19

@st.cache_data(show_spinner=False)
def calculate_T(E, V0, L):
    """Calculate transmission coefficient with numerical stability"""
    E_j = E * e_charge
    V0_j = V0 * e_charge
    L_m = L * 1e-9
    
    if E_j < V0_j:
        kappa = np.sqrt(2*m_e*(V0_j - E_j + 1e-30))/hbar
        T = 1/(1 + (V0**2 * np.sinh(kappa*L_m)**2)/(4*E*(V0 - E + 1e-30)))
    else:
        k = np.sqrt(2*m_e*(E_j - V0_j + 1e-30))/hbar
        T = 1/(1 + (V0**2 * np.sin(k*L_m)**2)/(4*E*(E - V0 + 1e-30)))
    return np.clip(T, 0, 1)

@st.cache_data(show_spinner=False, max_entries=3)
def simulate_trajectories(E, V0, L, _n_frames=n_frames):
    """Generate optimized 3D trajectories with caching"""
    T = calculate_T(E, V0, L)
    outcomes = np.random.rand(num_electrons) < T
    
    trajectories = []
    for outcome in outcomes:
        x = np.linspace(-2, 4 if outcome else -2, _n_frames)
        y = np.cumsum(np.random.normal(0, 0.03, _n_frames))
        z = np.cumsum(np.random.normal(0, 0.03, _n_frames))
        
        # Add quantum tunneling delay effect
        barrier_mask = (x >= 0) & (x <= L)
        x[barrier_mask] = np.linspace(0, L, sum(barrier_mask)) * np.random.uniform(0.8, 1.2)
        
        trajectories.append(np.column_stack([x, y, z]))
    
    return np.array(trajectories), outcomes, T

def create_3d_figure(E, V0, L):
    """Create optimized 3D figure for Streamlit"""
    trajectories, outcomes, T_val = simulate_trajectories(E, V0, L)
    colors = ['#4CAF50' if o else '#F44336' for o in outcomes]

    fig = go.Figure()
    
    # Add barrier
    fig.add_trace(go.Mesh3d(
        x=[0,0,0,0,L,L,L,L],
        y=[-1,-1,1,1,-1,-1,1,1],
        z=[-1,1,-1,1,-1,1,-1,1],
        i=[0,0,0,1,2,4,4,6,6,0,2,3],
        j=[1,2,4,3,3,5,6,7,5,4,6,7],
        k=[2,4,5,2,7,7,5,3,7,5,4,1],
        color='#607D8B',
        opacity=0.2,
        name=f'Barrier (L={L}nm)'
    ))
    
    # Add electron traces
    for i in range(num_electrons):
        fig.add_trace(go.Scatter3d(
            x=trajectories[i,:,0],
            y=trajectories[i,:,1],
            z=trajectories[i,:,2],
            mode='lines+markers',
            marker=dict(size=electron_size, color=colors[i], opacity=0.9),
            line=dict(color=colors[i], width=3, opacity=0.2),
            name=f'Electron {i+1}'
        ))

    # Animation controls
    frames = [go.Frame(
        data=[go.Scatter3d(
            x=traj[:k+1,0],
            y=traj[:k+1,1],
            z=traj[:k+1,2],
        ) for traj in trajectories],
        name=str(k)
    ) for k in range(n_frames)]

    fig.frames = frames
    
    fig.update_layout(
        title=dict(
            text=f"Quantum Tunneling Simulation<br>E={E:.1f}eV, V0={V0}eV, T={T_val:.3f}",
            x=0.05,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title="X (nm)", range=[-2,4], backgroundcolor='rgba(240,240,240,0.8)'),
            yaxis=dict(title="Y (nm)", range=[-2,2], backgroundcolor='rgba(240,240,240,0.8)'),
            zaxis=dict(title="Z (nm)", range=[-2,2], backgroundcolor='rgba(240,240,240,0.8)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="▶ Play",
                method="animate",
                args=[None, {"frame": {"duration": 70}, "fromcurrent": True}]
            )]
        )]
    )
    
    return fig

def main():
    st.set_page_config(page_title="3D Quantum Simulator", layout="wide")
    st.title("3D Quantum Tunneling Visualization")
    
    with st.sidebar:
        st.header("Simulation Parameters")
        E = st.slider("Electron Energy (eV)", 0.1, 10.0, 5.0, 0.1)
        V0 = st.slider("Barrier Height (eV)", 0.1, 10.0, 1.0, 0.1)
        L = st.slider("Barrier Width (nm)", 0.1, 5.0, 1.0, 0.1)
    
    with st.spinner('Generating quantum simulation...'):
        fig = create_3d_figure(E, V0, L)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

if __name__ == "__main__":
    main()
