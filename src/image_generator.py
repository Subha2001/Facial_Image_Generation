from tensorflow.keras.models import load_model  # Import the load_model function
from tensorflow.keras.layers import Conv2DTranspose  # Import the Conv2DTranspose layer
from PIL import Image  # Import the PIL Image module for image manipulation
import numpy as np  # Import NumPy for numerical operations

# Define a custom layer that inherits from Conv2DTranspose.
# Modify or expand this class with any custom functionality you require.
class CustomConv2DTranspose(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        super(CustomConv2DTranspose, self).__init__(*args, **kwargs)  # Initialize the parent class
    # You can override methods here if needed.

def load_generator_model():
    """
    Loads the generator model using the custom object dictionary to properly deserialize
    layers like CustomConv2DTranspose.
    """
    # Adjust the path to your model as needed.
    model = load_model("model/generator_model_16000.h5",  # Load the saved generator model
                        custom_objects={'Conv2DTranspose': CustomConv2DTranspose})  # Pass the custom layer to load_model
    return model  # Return the loaded model

# Load the generator model once from the cache.
generator = load_generator_model()  # Load the generator model
latent_dim = 64  # Adjust if your generator uses a different latent dimension.

def generate_image(model, latent_dim):
    """
    Generates a new image using the provided generator model and latent space dimension.
    Assumes the model outputs images in the range [-1, 1].
    """
    # Generate a single random latent vector.
    latent_vector = np.random.normal(0, 1, (1, latent_dim))  # Create a random latent vector

    # Predict a new image from the generator.
    generated = model.predict(latent_vector)  # Generate an image using the generator

    # Post-process: scale from [-1, 1] to [0, 1] and remove the batch dimension.
    generated = (generated + 1) / 2.0  # Scale the pixel values to [0, 1]
    generated = generated[0]  # Remove the batch dimension
    generated = np.clip(generated, 0, 1)  # Clip the values to the range [0, 1]

    # Convert to uint8 for display.
    generated = (generated * 255).astype(np.uint8)  # Convert the pixel values to uint8

    # Convert to a PIL Image. Use grayscale mode "L" if single channel, else "RGB".
    if generated.ndim == 2 or generated.shape[-1] == 1:  # Check if the image is grayscale
        if generated.ndim == 3: # If the image has a single channel with a channel dimension
            generated = np.squeeze(generated, axis=-1) # remove that channel dimension
        final_img = Image.fromarray(generated, mode="L")  # Create a grayscale PIL Image
    else:
        final_img = Image.fromarray(generated, mode="RGB")  # Create an RGB PIL Image

    return final_img  # Return the generated PIL Image