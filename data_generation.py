import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Arc, FancyArrowPatch


class TechnicalDrawingGenerator:
    def __init__(self, stl_file):
        """Initialize with STL file path"""
        self.mesh = trimesh.load_mesh(stl_file)
        self.mesh.apply_translation(-self.mesh.center_mass)  # Center the mesh

    def add_dimension_line(
        self,
        ax,
        point1,
        point2,
        offset=0.5,
        text_offset=0.1,
        dimension_text=None,
        arrow_style="<->",
        color="red",
    ):
        """Add a dimension line between two points"""
        x1, y1 = point1
        x2, y2 = point2

        # Calculate direction and perpendicular vectors
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return

        # Unit vector along the line
        ux, uy = dx / length, dy / length
        # Perpendicular unit vector
        px, py = -uy, ux

        # Offset points
        offset_point1 = (x1 + px * offset, y1 + py * offset)
        offset_point2 = (x2 + px * offset, y2 + py * offset)

        # Extension lines
        ax.plot(
            [x1, offset_point1[0]],
            [y1, offset_point1[1]],
            color=color,
            linewidth=0.8,
            linestyle="-",
        )
        ax.plot(
            [x2, offset_point2[0]],
            [y2, offset_point2[1]],
            color=color,
            linewidth=0.8,
            linestyle="-",
        )

        # Dimension line with arrows
        arrow = FancyArrowPatch(
            offset_point1,
            offset_point2,
            arrowstyle=arrow_style,
            color=color,
            linewidth=0.8,
            mutation_scale=15,
        )
        ax.add_patch(arrow)

        # Dimension text
        if dimension_text is None:
            dimension_text = f"{length:.1f}"

        mid_x = (offset_point1[0] + offset_point2[0]) / 2
        mid_y = (offset_point1[1] + offset_point2[1]) / 2 + text_offset

        ax.text(
            mid_x,
            mid_y,
            dimension_text,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    def add_radius_dimension(
        self, ax, center, radius_point, dimension_text=None, color="red"
    ):
        """Add a radius dimension line"""
        cx, cy = center
        rx, ry = radius_point

        radius = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)

        # Draw radius line
        ax.plot([cx, rx], [cy, ry], color=color, linewidth=0.8, linestyle="--")

        # Add R symbol and dimension
        if dimension_text is None:
            dimension_text = f"R{radius:.1f}"

        mid_x = (cx + rx) / 2
        mid_y = (cy + ry) / 2

        ax.text(
            mid_x,
            mid_y,
            dimension_text,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    def add_diameter_dimension(
        self, ax, center, radius, dimension_text=None, angle=45, color="red"
    ):
        """Add a diameter dimension line"""
        cx, cy = center

        # Calculate points on circle
        angle_rad = np.radians(angle)
        x1 = cx - radius * np.cos(angle_rad)
        y1 = cy - radius * np.sin(angle_rad)
        x2 = cx + radius * np.cos(angle_rad)
        y2 = cy + radius * np.sin(angle_rad)

        # Draw diameter line
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="<->",
            color=color,
            linewidth=0.8,
            mutation_scale=15,
        )
        ax.add_patch(arrow)

        # Add diameter symbol and dimension
        if dimension_text is None:
            dimension_text = f"⌀{2*radius:.1f}"

        ax.text(
            cx,
            cy + radius + 0.3,
            dimension_text,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    def get_orthographic_projection(self, view_direction):
        """Get 2D projection of the mesh from a specific view direction"""
        # Create transformation matrix for the view
        if view_direction == "front":
            # Front view (looking along -Z axis)
            transform = np.eye(4)
        elif view_direction == "top":
            # Top view (looking along -Y axis)
            transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        elif view_direction == "side":
            # Side view (looking along +X axis)
            transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        else:
            transform = np.eye(4)

        # Apply transformation
        mesh_copy = self.mesh.copy()
        mesh_copy.apply_transform(transform)

        # Get cross-section at z=0 plane
        try:
            section = mesh_copy.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
            if section is not None:
                slice_2D, _ = section.to_2D()
                return slice_2D
        except:
            pass

        # If sectioning fails, use projection of vertices
        vertices_2d = mesh_copy.vertices[:, :2]  # Take X,Y coordinates
        return vertices_2d

    def create_technical_drawing(self, output_prefix="technical_drawing"):
        """Create technical drawings with dimensions for all three views"""
        views = ["front", "top", "side"]
        view_titles = ["Front View", "Top View", "Side View"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Technical Drawing with Dimensions", fontsize=16, fontweight="bold"
        )

        # Flatten axes for easier indexing
        axes = axes.flatten()

        for i, (view, title) in enumerate(zip(views, view_titles)):
            ax = axes[i]

            # Get 2D projection
            projection = self.get_orthographic_projection(view)

            if hasattr(projection, "discrete") and projection.discrete:
                # Handle trimesh Path2D object
                for path in projection.discrete:
                    if len(path) > 0:
                        path_array = np.array(path)
                        if len(path_array.shape) == 2 and path_array.shape[1] >= 2:
                            ax.plot(
                                path_array[:, 0], path_array[:, 1], "k-", linewidth=1.5
                            )

                # Add dimensions based on the projection bounds
                bounds = projection.bounds
                if bounds is not None and len(bounds) == 4:
                    min_x, min_y, max_x, max_y = bounds

                    # Add overall width dimension
                    self.add_dimension_line(
                        ax,
                        (min_x, max_y + 0.5),
                        (max_x, max_y + 0.5),
                        offset=0,
                        text_offset=0.2,
                    )

                    # Add overall height dimension
                    self.add_dimension_line(
                        ax,
                        (max_x + 0.5, min_y),
                        (max_x + 0.5, max_y),
                        offset=0,
                        text_offset=0.2,
                    )

            else:
                # Handle array of vertices
                if len(projection) > 0:
                    if len(projection.shape) == 2 and projection.shape[1] >= 2:
                        # Plot convex hull of points
                        try:
                            from scipy.spatial import ConvexHull

                            hull = ConvexHull(projection[:, :2])
                            hull_points = projection[hull.vertices]
                            hull_points = np.vstack(
                                [hull_points, hull_points[0]]
                            )  # Close the loop
                            ax.plot(
                                hull_points[:, 0],
                                hull_points[:, 1],
                                "k-",
                                linewidth=1.5,
                            )

                            # Add dimensions
                            min_x, min_y = np.min(projection, axis=0)
                            max_x, max_y = np.max(projection, axis=0)

                            # Overall dimensions
                            self.add_dimension_line(
                                ax,
                                (min_x, max_y + 0.5),
                                (max_x, max_y + 0.5),
                                offset=0,
                                text_offset=0.2,
                            )
                            self.add_dimension_line(
                                ax,
                                (max_x + 0.5, min_y),
                                (max_x + 0.5, max_y),
                                offset=0,
                                text_offset=0.2,
                            )
                        except:
                            # Fallback: plot all points
                            ax.scatter(
                                projection[:, 0], projection[:, 1], s=1, c="black"
                            )

            ax.set_title(title, fontweight="bold")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        # Hide the fourth subplot
        axes[3].set_visible(False)

        # Add drawing information in the fourth subplot area
        info_text = f"""
        Drawing Information:
        • File: {self.mesh.metadata.get('file_name', 'Unknown')}
        • Vertices: {len(self.mesh.vertices)}
        • Faces: {len(self.mesh.faces)}
        • Volume: {self.mesh.volume:.2f}
        • Surface Area: {self.mesh.area:.2f}
        
        Dimensions shown in model units
        Generated automatically from STL file
        """

        axes[3].text(
            0.1,
            0.5,
            info_text,
            transform=axes[3].transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_complete.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Create individual detailed views
        self.create_detailed_views(output_prefix)

    def create_detailed_views(self, output_prefix):
        """Create detailed individual views with more comprehensive dimensions"""
        views = [("front", "Front View"), ("top", "Top View"), ("side", "Side View")]

        for view, title in views:
            fig, ax = plt.subplots(figsize=(12, 8))

            projection = self.get_orthographic_projection(view)

            # Plot the projection
            if hasattr(projection, "discrete") and projection.discrete:
                for path in projection.discrete:
                    if len(path) > 0:
                        path_array = np.array(path)
                        if len(path_array.shape) == 2 and path_array.shape[1] >= 2:
                            ax.plot(
                                path_array[:, 0], path_array[:, 1], "k-", linewidth=2
                            )

                bounds = projection.bounds
                if bounds is not None:
                    min_x, min_y, max_x, max_y = bounds
                    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

                    # Multiple dimension lines at different offsets
                    self.add_dimension_line(
                        ax,
                        (min_x, max_y + 1.0),
                        (max_x, max_y + 1.0),
                        offset=0,
                        text_offset=0.3,
                        dimension_text=f"{max_x - min_x:.2f}",
                    )

                    self.add_dimension_line(
                        ax,
                        (max_x + 1.0, min_y),
                        (max_x + 1.0, max_y),
                        offset=0,
                        text_offset=0.3,
                        dimension_text=f"{max_y - min_y:.2f}",
                    )

                    # Add centerlines
                    ax.axhline(
                        y=center_y,
                        color="blue",
                        linestyle="-.",
                        alpha=0.5,
                        linewidth=0.8,
                    )
                    ax.axvline(
                        x=center_x,
                        color="blue",
                        linestyle="-.",
                        alpha=0.5,
                        linewidth=0.8,
                    )

            ax.set_title(
                f"{title} - Detailed Dimensions", fontsize=14, fontweight="bold"
            )
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (units)")
            ax.set_ylabel("Y (units)")

            # Add border
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            plt.tight_layout()
            plt.savefig(
                f"{output_prefix}_{view}_detailed.png", dpi=300, bbox_inches="tight"
            )
            plt.show()


# Usage example
if __name__ == "__main__":
    # Initialize the technical drawing generator
    generator = TechnicalDrawingGenerator("ball-bearing.stl")

    # Create comprehensive technical drawings
    generator.create_technical_drawing("ball_bearing_technical")

    print("Technical drawings with dimensions have been generated!")
    print("Files created:")
    print("- ball_bearing_technical_complete.png (overview)")
    print("- ball_bearing_technical_front_detailed.png")
    print("- ball_bearing_technical_top_detailed.png")
    print("- ball_bearing_technical_side_detailed.png")
