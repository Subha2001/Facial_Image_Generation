import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime  # Imported to log the current time

# Set page configuration
st.set_page_config(page_title="GAN Image Generator", layout="centered")

# Inject CSS to center content horizontally only
st.markdown(
    """
    <style>
    /* Center main container horizontally */
    .center-block {
        max-width: 800px; /* Adjust width as needed */
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    /* Optionally, center the expander content */
    .expander-content {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# Wrap the entire main content in a centered div.
st.markdown('<div class="center-block">', unsafe_allow_html=True)

# Title and introductory text (centered horizontally)
st.title("😀 Human Face Thinker")
st.markdown("Welcome! Click the below button to think a new human face")

# Use an expander for the "About This App" section
with st.expander("About This App"):
    st.markdown(
        """
        <div class="expander-content">
        This application leverages a pre-trained GAN model that thinks human faces.
        </div>
        """,
        unsafe_allow_html=True
    )

# Cache heavy resources (load the generator model)
@st.cache_resource
def load_generator_model():
    # Adjust the path to your model as needed.
    model = load_model("data/model_00200.h5")
    return model

# Load the generator model once from the cache.
generator = load_generator_model()
latent_dim = 64  # Adjust if your generator uses a different latent dimension.

def generate_image(model, latent_dim):
    """
    Generates a new image using the provided generator model and latent space dimension.
    Assumes the model outputs images in the range [-1, 1].
    """
    # Generate a single random latent vector.
    latent_vector = np.random.normal(0, 1, (1, latent_dim))
    
    # Predict a new image from the generator.
    generated = model.predict(latent_vector)
    
    # Post-process: scale from [-1, 1] to [0, 1] and remove the batch dimension.
    generated = (generated + 1) / 2.0
    generated = generated[0]
    generated = np.clip(generated, 0, 1)
    
    # Convert to uint8 for display.
    generated = (generated * 255).astype(np.uint8)
    
    # Convert to a PIL Image.
    if generated.ndim == 2 or generated.shape[-1] == 1:
        if generated.ndim == 3:
            generated = np.squeeze(generated, axis=-1)
        final_img = Image.fromarray(generated, mode="L")
    else:
        final_img = Image.fromarray(generated, mode="RGB")
    
    return final_img

# Place the Generate Image button at the top (centered horizontally by the .center-block wrapper)
if st.button("Generate Image"):
    with st.spinner("Thinking Human Face 🤔🤔🤔🤔🤔"):
        img = generate_image(generator, latent_dim)
        st.image(img, caption="Generated Image", use_container_width=True)
    # Log the event to your terminal with the current time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Face generated at {timestamp}")
    st.success("Image generated successfully!")

# Close the centered div.
st.markdown('</div>', unsafe_allow_html=True)