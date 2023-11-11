# Paquetes incorporados de Python
from pathlib import Path  # Para manejar rutas de archivos y directorios
import PIL  # Para trabajar con imágenes

# Paquetes externos
import streamlit as st  # Para construir aplicaciones web interactivas

# Módulos locales
import settings  # Configuraciones personalizadas para el proyecto
import helper  # Funciones auxiliares definidas localmente

# Configuración del diseño de la página
st.set_page_config(
    page_title="Detector de objetos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal de la página
st.title("Detector de objetos")

# Barra lateral
st.sidebar.header("Configuración del Modelo de Aprendizaje Automático")

# Opciones del modelo
model_type = st.sidebar.radio(
    "Seleccionar Tarea", ['Detección', 'Segmentación'])

# Barra deslizante para ajustar la confianza del modelo
confidence = float(st.sidebar.slider(
    "Seleccionar Confianza del Modelo", 25, 100, 40)) / 100

# Selección de Detección o Segmentación
if model_type == 'Detección':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentación':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Carga del modelo de aprendizaje automático preentrenado
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"No se pudo cargar el modelo. Verifica la ruta especificada: {model_path}")
    st.error(ex)

st.sidebar.header("Configuración de Imagen/Video")
source_radio = st.sidebar.radio(
    "Seleccionar Fuente", settings.SOURCES_LIST)

source_img = None
# Si se selecciona una imagen
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Elige una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Imagen por defecto",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Imagen cargada",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error al abrir la imagen.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Imagen Detectada',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detectar Objetos'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Imagen Detectada',
                         use_column_width=True)
                try:
                    with st.expander("Resultados de Detección"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("¡Aún no se ha cargado ninguna imagen!")

# ... (resto del código) ...
# ... (código anterior) ...

elif source_radio == settings.VIDEO:
    # Reproducir video almacenado
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    # Reproducir video desde la cámara web
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    # Reproducir transmisión de video RTSP
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    # Reproducir video de YouTube
    helper.play_youtube_video(confidence, model)

else:
    st.error("¡Por favor, selecciona un tipo de fuente válido!")

# Fin del código

