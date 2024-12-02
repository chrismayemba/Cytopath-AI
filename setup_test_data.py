import os
import pathlib
from PIL import Image, ImageDraw
import numpy as np

def create_test_data_structure():
    """Create test data directory structure"""
    base_dir = pathlib.Path(__file__).parent / "test_data"
    categories = [
        "NILM",
        "LSIL",
        "HSIL",
        "SCC",  # Squamous Cell Carcinoma
        "OTHER"
    ]
    
    # Create base directory
    base_dir.mkdir(exist_ok=True)
    
    # Create category directories
    for category in categories:
        category_dir = base_dir / category
        category_dir.mkdir(exist_ok=True)
        
    return base_dir

def create_synthetic_cell_image(category: str, size=(224, 224)):
    """Create synthetic cell image based on category"""
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Base parameters
    num_cells = np.random.randint(15, 25)
    
    # Category-specific parameters
    if category == "NILM":
        cell_colors = [(150, 150, 150), (160, 160, 160)]
        nucleus_colors = [(100, 100, 100)]
        nucleus_sizes = [0.3, 0.4]
    elif category == "LSIL":
        cell_colors = [(170, 140, 140), (180, 150, 150)]
        nucleus_colors = [(120, 80, 80)]
        nucleus_sizes = [0.4, 0.5]
    elif category == "HSIL":
        cell_colors = [(190, 130, 130), (200, 140, 140)]
        nucleus_colors = [(140, 60, 60)]
        nucleus_sizes = [0.5, 0.6]
    elif category == "SCC":
        cell_colors = [(210, 120, 120), (220, 130, 130)]
        nucleus_colors = [(160, 40, 40)]
        nucleus_sizes = [0.6, 0.7]
    else:  # OTHER
        cell_colors = [(160, 160, 140), (170, 170, 150)]
        nucleus_colors = [(110, 110, 90)]
        nucleus_sizes = [0.35, 0.45]
    
    # Draw cells
    for _ in range(num_cells):
        # Random position
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        
        # Random size
        radius = np.random.randint(15, 35)
        
        # Draw cell
        cell_color = cell_colors[np.random.randint(0, len(cell_colors))]
        draw.ellipse(
            [(x-radius, y-radius), (x+radius, y+radius)],
            fill=cell_color,
            outline=(128, 128, 128)
        )
        
        # Draw nucleus
        nucleus_color = nucleus_colors[np.random.randint(0, len(nucleus_colors))]
        nucleus_size = nucleus_sizes[np.random.randint(0, len(nucleus_sizes))]
        nucleus_radius = int(radius * nucleus_size)
        draw.ellipse(
            [(x-nucleus_radius, y-nucleus_radius),
             (x+nucleus_radius, y+nucleus_radius)],
            fill=nucleus_color,
            outline=(64, 64, 64)
        )
    
    return image

def generate_test_images(num_images_per_category=10):
    """Generate test images for each category"""
    base_dir = create_test_data_structure()
    categories = ["NILM", "LSIL", "HSIL", "SCC", "OTHER"]
    
    for category in categories:
        category_dir = base_dir / category
        print(f"Generating {num_images_per_category} images for {category}...")
        
        for i in range(num_images_per_category):
            image = create_synthetic_cell_image(category)
            image_path = category_dir / f"{category.lower()}_{i+1}.jpg"
            image.save(image_path, quality=95)
            print(f"  Created {image_path}")

if __name__ == "__main__":
    generate_test_images()
