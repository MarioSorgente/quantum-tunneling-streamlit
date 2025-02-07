import plotly.graph_objects as go

def create_3d_figure(E, V0, L):
    """
    Create a 3D animated Plotly figure showing electron trajectories and the barrier.
    """
    n_frames = 50
    trajectories, outcomes, T_val = simulate_trajectories(E, V0, L, n_frames=n_frames)
    
    # Create one trace per electron (each trace is a moving point)
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
    
    # Build animation frames using go.Frame objects.
    frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        for i in range(num_electrons):
            traj = trajectories[i]
            scatter = go.Scatter3d(
                x=[traj[frame_idx, 0]],
                y=[traj[frame_idx, 1]],
                z=[traj[frame_idx, 2]],
                mode='markers',
                marker=dict(size=8, color=('#4CAF50' if outcomes[i] else '#F44336'), opacity=0.95)
            )
            frame_data.append(scatter)
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    # Create a 3D barrier as a cuboid using Mesh3d.
    # The barrier extends from x=0 to x=L (nm) and y,z from -1 to 1.
    X = [0, 0, 0, 0, L, L, L, L]
    Y = [-1, -1, 1, 1, -1, -1, 1, 1]
    Z = [-1, 1, -1, 1, -1, 1, -1, 1]
    # Faces are defined via indices
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
    
    # Build the base figure
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
                                           "fromcurrent": True,
                                           "transition": {"duration": 0}}])]
            )]
        )
    )
    
    # Set the frames for the figure using the list of go.Frame objects.
    fig.frames = frames
    
    # Add a slider to control the animation manually.
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
        x=0,
        y=0,
        currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor='center'),
        len=1.0
    )]
    fig.update_layout(sliders=sliders)
    return fig
