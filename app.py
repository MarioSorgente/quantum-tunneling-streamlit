import numpy as np
import plotly.graph_objects as go

# Configuration
num_electrons = 5
n_frames = 60
barrier_opacity = 0.25
electron_size = 10
trail_length = 15  # Number of previous positions to show

def simulate_trajectories(E, V0, L, n_frames=50):
    """Generate 3D electron trajectories with tunneling probabilities"""
    T = calculate_T(E, V0, L)
    outcomes = np.random.rand(num_electrons) < T
    
    trajectories = []
    for outcome in outcomes:
        # Base trajectory through space (x, y, z)
        x = np.linspace(-2, 4 if outcome else -2, n_frames)
        
        # Add quantum "jitter" in y/z dimensions
        y = np.cumsum(np.random.normal(0, 0.05, n_frames))
        z = np.cumsum(np.random.normal(0, 0.05, n_frames))
        
        # Slow down near barrier (x=0 to x=L)
        mask = (x >= 0) & (x <= L)
        x[mask] = np.linspace(0, L, sum(mask))  # Linear through barrier
        
        trajectories.append(np.column_stack([x, y, z]))
    
    return np.array(trajectories), outcomes, T

def create_3d_figure(E, V0, L):
    """Create enhanced 3D visualization with proper physics and design"""
    trajectories, outcomes, T_val = simulate_trajectories(E, V0, L, n_frames)
    
    # Create figure with custom color scale
    fig = go.Figure()
    
    # Add barrier with improved mesh
    barrier_mesh = go.Mesh3d(
        x=[0, 0, 0, 0, L, L, L, L],
        y=[-1, -1, 1, 1, -1, -1, 1, 1],
        z=[-1, 1, -1, 1, -1, 1, -1, 1],
        i=[0, 0, 0, 1, 2, 4, 4, 6, 6, 0, 2, 3],
        j=[1, 2, 4, 3, 3, 5, 6, 7, 5, 4, 6, 7],
        k=[2, 4, 5, 2, 7, 7, 5, 3, 7, 5, 4, 1],
        color='#607D8B',
        opacity=barrier_opacity,
        flatshading=True,
        name=f'Barrier (V0={V0}eV)',
        showlegend=True
    )
    fig.add_trace(barrier_mesh)
    
    # Create electron traces with history trails
    colors = ['#4CAF50' if o else '#F44336' for o in outcomes]
    for i in range(num_electrons):
        fig.add_trace(go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='lines+markers',
            marker=dict(
                size=electron_size,
                color=colors[i],
                opacity=0.9,
                symbol='circle'
            ),
            line=dict(
                color=colors[i],
                width=4,
                opacity=0.3
            ),
            name=f'Electron {i+1}',
            hoverinfo='skip'
        ))
    
    # Create animation frames with smooth transitions
    frames = []
    for k in range(n_frames):
        frame_data = []
        for i in range(num_electrons):
            # Show trail of previous positions
            start_idx = max(0, k - trail_length)
            frame_data.append(
                go.Scatter3d(
                    x=trajectories[i, start_idx:k+1, 0],
                    y=trajectories[i, start_idx:k+1, 1],
                    z=trajectories[i, start_idx:k+1, 2],
                    mode='lines+markers'
                )
            )
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(k),
            traces=list(range(1, num_electrons + 1))
        ))
    
    # Animation controls and camera setup
    fig.update_layout(
        title=dict(
            text=f"<b>Quantum Tunneling Simulation</b><br>E={E:.1f}eV, V0={V0}eV, L={L}nm, T={T_val:.3f}",
            x=0.05,
            font=dict(size=20, color='#2c3e50')
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="X (nm)", font=dict(size=14)),
                backgroundcolor='rgba(245,245,245,0.8)',
                gridcolor='white'
            ),
            yaxis=dict(
                title=dict(text="Y (nm)", font=dict(size=14)),
                backgroundcolor='rgba(245,245,245,0.8)', 
                gridcolor='white'
            ),
            zaxis=dict(
                title=dict(text="Z (nm)", font=dict(size=14)),
                backgroundcolor='rgba(245,245,245,0.8)',
                gridcolor='white'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
            aspectratio=dict(x=2, y=1, z=1)
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=1,
            buttons=[dict(
                label="▶ Play",
                method="animate",
                args=[None, {
                    "frame": {"duration": 75, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 50}
                }]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[str(k)], {"frame": {"duration": 0}, "mode": "immediate"}],
                label=f"{k*100/(n_frames-1):.0f}%"
            ) for k in range(n_frames)],
            active=0,
            transition=dict(duration=0),
            x=0.1,
            y=0,
            len=0.9
        )],
        legend=dict(
            x=0.82,
            y=0.9,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc'
        )
    )
    
    fig.frames = frames
    
    # Add dynamic annotations
    annotations = [
        dict(
            x=0.5*L,
            y=0,
            z=0,
            text=f"Barrier Width: {L}nm",
            showarrow=False,
            font=dict(size=12, color='darkblue')
        ),
        dict(
            x=0.5*L,
            y=0,
            z=1.5,
            text=f"Transmission Probability: {T_val:.3f}",
            showarrow=False,
            font=dict(size=12, color='#2ecc71' if T_val > 0.5 else '#e74c3c')
        )
    ]
    fig.update_layout(annotations=annotations)
    
    return fig
