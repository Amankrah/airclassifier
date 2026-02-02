"""
Visualization module for cyclone air classifier.

Provides plotting utilities and rendering capabilities for
simulation results visualization.
"""

from .plotters import (
    plot_particle_positions_2d,
    plot_particle_positions_3d,
    plot_size_distribution,
    plot_velocity_field_slice,
    plot_tangential_velocity_profile,
    plot_simulation_summary,
    create_interactive_3d_plot,
)

from .renderer import (
    SimpleRenderer,
    PygletRenderer,
    visualize_results_interactive,
)

from .flow_viz import (
    # Single flow field visualization
    plot_velocity_magnitude_slice,
    plot_velocity_components_slice,
    plot_velocity_vectors,
    compute_streamlines,
    plot_streamlines_2d,
    plot_streamlines_3d,
    plot_radial_velocity_profile,
    plot_axial_velocity_profile,
    plot_vortex_structure,
    create_flow_animation,
    plot_pressure_gradient_estimate,
    plot_flow_summary,
    generate_seed_points_ring,
    generate_seed_points_grid,
    # System-level visualization
    visualize_air_system_flow,
    visualize_feed_system_flow,
    visualize_classification_results,
    visualize_complete_system_status,
    plot_particle_trajectories_3d,
    create_system_flow_summary,
)

from .geometry_viz import (
    # Main visualization classes
    GeometryVisualizer,
    VisualizationRequest,
    VisualizationType,
    RenderBackend,
    WarpMeshProcessor,
    PyVistaRenderer,
    MatplotlibRenderer,
    # Convenience functions
    visualize_geometry,
    quick_render,
    # Availability flags
    PYVISTA_AVAILABLE,
    WARP_AVAILABLE,
)

__all__ = [
    # Plotters
    "plot_particle_positions_2d",
    "plot_particle_positions_3d",
    "plot_size_distribution",
    "plot_velocity_field_slice",
    "plot_tangential_velocity_profile",
    "plot_simulation_summary",
    "create_interactive_3d_plot",
    # Renderers
    "SimpleRenderer",
    "PygletRenderer",
    "visualize_results_interactive",
    # Flow Visualization
    "plot_velocity_magnitude_slice",
    "plot_velocity_components_slice",
    "plot_velocity_vectors",
    "compute_streamlines",
    "plot_streamlines_2d",
    "plot_streamlines_3d",
    "plot_radial_velocity_profile",
    "plot_axial_velocity_profile",
    "plot_vortex_structure",
    "create_flow_animation",
    "plot_pressure_gradient_estimate",
    "plot_flow_summary",
    "generate_seed_points_ring",
    "generate_seed_points_grid",
    # System-level Visualization
    "visualize_air_system_flow",
    "visualize_feed_system_flow",
    "visualize_classification_results",
    "visualize_complete_system_status",
    "plot_particle_trajectories_3d",
    "create_system_flow_summary",
    # Geometry Visualization
    "GeometryVisualizer",
    "VisualizationRequest",
    "VisualizationType",
    "RenderBackend",
    "WarpMeshProcessor",
    "PyVistaRenderer",
    "MatplotlibRenderer",
    "visualize_geometry",
    "quick_render",
    "PYVISTA_AVAILABLE",
    "WARP_AVAILABLE",
]
