import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Set the width and height of the screen
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Grayscale Map")

# Define your 2D array of brightness values (e.g., a NumPy array)
# Example array with random values:
matrix_width, matrix_height = 600, 600

# Create a Pygame surface with the same dimensions as the matrix
surface = pygame.Surface((matrix_width, matrix_height))

# Initialize Pygame clock
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    brightness_values = np.random.rand(matrix_height, matrix_width)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((255, 255, 255))

    # Use NumPy to create an array representing the entire image
    pixel_array = (brightness_values * 255).astype(np.uint8)
    pixel_array = np.repeat(pixel_array[:, :, np.newaxis], 3, axis=2)  # Repeat values for RGB channels

    # Check and correct the dimensions if necessary
    if pixel_array.shape != (matrix_height, matrix_width, 3):
        pixel_array = pixel_array[:matrix_height, :matrix_width, :]

    # Use pygame.surfarray to transfer pixel data to the surface
    print(pixel_array.shape);
    print(surface.get_size());
    pygame.surfarray.blit_array(surface, pixel_array)

    # Draw the surface onto the screen
    screen.blit(surface, (0, 0))

    # Update the display
    pygame.display.flip()

    # Print the framerate
    framerate = clock.get_fps()
    print(f"Framerate: {framerate:.2f} fps")

    # Cap the frame rate to 60 frames per second
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
