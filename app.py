import streamlit as st
from datetime import datetime
# Import the generate_image function along with the generator model and latent_dim.
from src.image_generator import generate_image, generator, latent_dim

# (Optional) Set Streamlit page configuration.
st.set_page_config(page_title="GAN Image Generator", layout="centered")

# Inject CSS to center content horizontally.
st.markdown(
    """
    <style>
    /* Center main container horizontally */
    .center-block {
        max-width: 800px;
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

# Wrap the content in a centered div.
st.markdown('<div class="center-block">', unsafe_allow_html=True)

# Title and introductory text.
st.title("😀 Human Face Imaginator")
st.markdown("Welcome! Click the below button to imagine a new human face")

# Use an expander to display additional information.
with st.expander("About This App"):
    st.markdown(
        """
        <div class="expander-content">
        This application leverages a pre-trained GAN model that imagine human faces.
        </div>
        """,
        unsafe_allow_html=True
    )

# Button to generate image.
if st.button("Generate Image"):
    with st.spinner("Imagining New Human Face 🤔🤔🤔🤔🤔 [Please Wait..... It can take some time]"):
        img = generate_image(generator, latent_dim)
        st.image(img, caption="Generated Image", use_container_width=True, channels="RGB")
    # Log the event with a timestamp.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Face generated at {timestamp}")
    st.success("Face imagined successfully!")

# Close the centered div.
st.markdown('</div>', unsafe_allow_html=True)