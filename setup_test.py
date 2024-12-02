import os
import pathlib

# Create test images directory
test_dir = pathlib.Path(__file__).parent / "test_images"
test_dir.mkdir(exist_ok=True)

print(f"Created test directory at {test_dir}")
