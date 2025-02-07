import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import base64

# Physical constants
hbar = 1.0545718e-34
m_e = 9.10938356e-31
e_charge = 1.602176634e-19
num_e = 5

def calculate_T(E, V0, L):
    """Calculate transmission coefficient"""
    E_j = E * e_charge
    V0_j = V0 * e_charge
    L_m = L * 1e-9
    
    if E_j < V0_j:
        kappa = np.sqrt(2*m_e*(V0_j - E_j))/hbar
        T = 1/(1 + (V0**2 * np.sinh(kappa*L_m)**2)/(4*E*(V0 - E)))
    else:
        k = np.sqrt(2*m_e*(E_j - V0_j))/hbar
        T = 1/(1 + (V0**2 * np.sin(k*L_m)**2)/(4*E*(E - V0)))
    return np.clip(T, 0, 1)

def create_animation(E, V0, L):
    """Create animation figure"""
    T = calculate_T(E, V0, L)
    outcomes = np.random.rand(num_e) < T
    colors = ['#4CAF50' if o else '#F44336' for o in outcomes]
    
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top plot
    E_range = np.linspace(0.1, 10, 200)
    T_vals = [calculate_T(e, V0, L) for e in E_range]
    ax_top.plot(E_range, T_vals, 'b-', lw=2)
    ax_top.axvline(E, color='r', ls=':', alpha=0.6)
    ax_top.set_title(f'E={E:.1f}eV, V0={V0}eV, L={L}nm\nT={T:.3f}, R={1-T:.3f}')
    ax_top.set_ylim(0, 1.1)
    ax_top.grid(True, alpha=0.3)
    
    # Bottom plot
    ax_bot.set_xlim(-1.5, 3.5)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis('off')
    ax_bot.add_patch(plt.Rectangle((1, 0.3), L*0.5, 0.4, fc='gray', alpha=0.5))
    
    # Animation
    y_pos = np.linspace(0.4, 0.6, num_e)
    electrons = [ax_bot.plot([], [], 'o', ms=12, color=c, alpha=0.8)[0] 
                for c in colors]
    
    x_trajs = []
    for outcome in outcomes:
        x_traj = np.concatenate([
            np.linspace(-1, 1, 15),
            np.linspace(1, 1, 10),
            np.linspace(1, 3 if outcome else -1, 15)
        ])
        x_trajs.append(x_traj)
    
    def animate(i):
        for j, e in enumerate(electrons):
            e.set_data([x_trajs[j][i]], [y_pos[j]])
        return electrons
    
    anim = FuncAnimation(fig, animate, frames=40, interval=50, blit=True)
    return anim

def main():
    st.set_page_config(page_title="Quantum Tunneling Simulator", layout="wide")
    st.title("1D Quantum Tunneling Simulator")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")
        E = st.slider("Electron Energy (eV)", 0.1, 10.0, 5.0)
        V0 = st.slider("Barrier Height (eV)", 0.1, 10.0, 1.0)
        L = st.slider("Barrier Width (nm)", 0.1, 5.0, 1.0)
    
    # Create animation
    anim = create_animation(E, V0, L)
    
    # Display animation
    st.write("### Quantum Simulation")
    with st.spinner("Generating animation..."):
        html = anim.to_jshtml()
        html = html.replace('height="288"', 'height="500"').replace('width="432"', 'width="800"')
        st.components.v1.html(html, height=600)

if __name__ == "__main__":
    main()
