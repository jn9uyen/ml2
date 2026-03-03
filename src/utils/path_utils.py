from pathlib import Path


def get_project_root() -> Path:
    """
    Detect the root directory by looking for common project markers.

    Searches upwards from the current file for a directory containing
    '.git', 'README.md', 'requirements.txt', or '.env'.

    Returns
    -------
    Path: The absolute path to the project root.

    Raises
    ------
    FileNotFoundError: If no project root marker is found before reaching the
    filesystem root.
    """
    project_markers = [".git", "README.md", "requirements.txt", ".env"]

    current_dir = Path(__file__).resolve().parent

    while current_dir != current_dir.parent:
        # Check if any of the markers exist in the current directory
        if any((current_dir / marker).exists() for marker in project_markers):
            return current_dir

        # Move up to the parent directory
        current_dir = current_dir.parent

    # If the loop completes, no project root was found
    raise FileNotFoundError("Could not determine project root. No markers found.")
