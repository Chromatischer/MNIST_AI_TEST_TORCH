import os

def generate_unique_filename(base_name, directory="."):
    """
    Generate a unique file name in the specified directory by appending a number if needed.

    Args:
        base_name (str): Desired base name for the file (e.g., "file.txt").
        directory (str): Directory to check for existing files. Defaults to the current directory.

    Returns:
        str: A unique file name.
    """
    # Extract the file name and extension
    name, ext = os.path.splitext(base_name)
    counter = 1
    unique_name = base_name

    # Check if the file already exists, and if so, append a number to the name
    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{name}_{counter}{ext}"
        counter += 1

    return os.path.join(directory, unique_name)

# Example usage:
file_name = generate_unique_filename("file.txt", "/path/to/directory")
print("Unique file name:", file_name)