import numpy as np
from PIL import Image, ImageDraw
import pathlib

def create_mock_cell_image(size=(224, 224)):
    """Create a mock cervical cell image for testing"""
    # Create a white background
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw some cell-like structures
    for _ in range(20):
        # Random position
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        
        # Random size
        radius = np.random.randint(10, 30)
        
        # Random color (cell-like)
        color = (
            np.random.randint(100, 200),  # R
            np.random.randint(100, 200),  # G
            np.random.randint(100, 200)   # B
        )
        
        # Draw cell
        draw.ellipse(
            [(x-radius, y-radius), (x+radius, y+radius)],
            fill=color,
            outline=(128, 128, 128)  # gray
        )
        
        # Draw nucleus
        nucleus_radius = radius // 3
        draw.ellipse(
            [(x-nucleus_radius, y-nucleus_radius),
             (x+nucleus_radius, y+nucleus_radius)],
            fill=(128, 0, 128),     # purple
            outline=(64, 64, 64)     # dark gray
        )
    
    return image

def main():
    # Create test image
    image = create_mock_cell_image()
    
    # Save image
    output_path = pathlib.Path(__file__).parent / "test_images" / "sample.jpg"
    image.save(output_path, quality=95)
    print(f"Created test image at {output_path}")

if __name__ == "__main__":
    main()
