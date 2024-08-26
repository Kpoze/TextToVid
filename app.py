import streamlit as st
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from base64 import b64encode
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import deep_translator
from langdetect import detect


pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Fonction pour traduire un texte en anglais
def translate_to_en(text=""):
    langue          = detect(text)
    translated_text = deep_translator.GoogleTranslator(source=langue, target='en').translate(text)
    return {'langue': langue, 'textTraduit': translated_text}

# Fonction utilitaire pour générer le lien de téléchargement
def get_binary_file_downloader_html(bin_file, file_label='File', file_mime='video/mp4'):
    data = b64encode(bin_file).decode()
    href = f'<a href="data:{file_mime};base64,{data}" download="output_video.mp4">{file_label}</a>'
    return href

# Titre de l'application
st.title("Générateur de vidéo")

# Champ de saisie pour le prompt
prompt                 = st.text_input("Entrez votre texte ici:")
video_duration_seconds = st.text_input("Entrez la durée de la vidéo")

# Bouton pour soumettre le prompt
if st.button("Générer"):
    if prompt and video_duration_seconds:
        video_duration_seconds = int(video_duration_seconds)
        tranlated_prompt       = translate_to_en(prompt)
        #st.write(f"Texte traduit: {tranlated_prompt['textTraduit']}")
        num_frames             = video_duration_seconds * 10
        video_frames           = pipe(tranlated_prompt['textTraduit'], negative_prompt="low quality", num_inference_steps=25, num_frames=num_frames).frames[0]
        video_path             = export_to_video(video_frames)
        # Générer le lien de téléchargement
        with open(video_path, "rb") as f:
            video_bytes = f.read()
         # Afficher le bouton de téléchargement dans Streamlit
        st.download_button(label="Télécharger la vidéo", data=video_bytes, file_name="video_generated.mp4", mime="video/mp4")

    else:
        st.write('veuillez remplir les champs')