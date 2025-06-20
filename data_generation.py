import matplotlib.pyplot as plt
import numpy as np
import trimesh


def load_and_center_mesh(file_path):
    """Load STL file and center its geometry."""
    mesh = trimesh.load_mesh(file_path)
    mesh.apply_translation(-mesh.center_mass)
    return mesh


def get_cross_section(mesh, origin, normal):
    """Get 2D cross-section of the mesh."""
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return None
    return section.to_2D()[0]  # Return only the Path2D


def plot_cross_section(ax, path2D, title, show_bbox=True):
    """Plot the 2D cross-section on a given axis."""
    for path in path2D.discrete:
        path = np.array(path)
        if path.shape[1] >= 2:
            ax.plot(path[:, 0], path[:, 1], "k-", linewidth=1)

    if show_bbox:
        min_bounds, max_bounds = path2D.bounds.reshape(2, 2)
        width, height = max_bounds - min_bounds
        rect = plt.Rectangle(
            min_bounds, width, height, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def create_cross_section_image(
    mesh, views, output_path="cross_sections.png", show_bbox=True
):
    """Create and save image of multiple cross-sections."""
    fig, axs = plt.subplots(1, len(views), figsize=(5 * len(views), 5))

    if len(views) == 1:
        axs = [axs]  # Ensure iterable

    for ax, (title, plane) in zip(axs, views.items()):
        path2D = get_cross_section(mesh, plane["origin"], plane["normal"])
        if path2D is not None:
            plot_cross_section(ax, path2D, title, show_bbox=show_bbox)
        else:
            ax.set_title(f"{title} (No section)")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return output_path


if __name__ == "__main__":
    STL_FILE = "ball-bearing.stl"
    OUTPUT_FILE = "cross_sections.png"

    # Define slicing views
    slicing_planes = {
        "Top View": {"origin": [0, 0, 0], "normal": [0, 0, 1]},
        "Front View": {"origin": [0, 0, 0], "normal": [0, 1, 0]},
        "Side View": {"origin": [0, 0, 0], "normal": [1, 0, 0]},
    }

    mesh = load_and_center_mesh(STL_FILE)
    result = create_cross_section_image(mesh, slicing_planes, OUTPUT_FILE)
    print(f"✅ Image saved to: {result}")
