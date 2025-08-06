import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pickle
import torch
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN

# Set the title of the application
st.set_page_config(page_title="Facial Verification and Anti-Spoofing", layout="wide")

# --- Header and Branding ---
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo_path = 'logo.png'
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.warning("Branding logo not found. Place 'logo.png' in the app directory.")
    except Exception as e:
        st.warning(f"Error loading logo: {e}")

with col2:
    st.title("Facial Verification and Anti-Spoofing System")

st.markdown("---")
st.write("Upload an image or use your webcam to perform facial verification and anti-spoofing.")

# --- Load Models and Encoder ---
@st.cache_resource
def load_models():
    loaded_model = None
    encoder = None
    embedder = None
    detector = None
    midas = None
    transform = None
    try:
        with open('svm_model_160x160.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        embedder = FaceNet()
        detector = MTCNN()
        if torch.cuda.is_available():
            try:
                midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
                midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                transform = midas_transforms.dpt_transform
                st.success("MiDaS model loaded successfully.", icon="✅")
            except Exception as midas_e:
                st.warning(f"Could not load MiDaS model: {midas_e}. Anti-spoofing will be disabled.", icon="⚠️")
                midas = None
                transform = None
        else:
            st.info("MiDaS model loading skipped (no CUDA available). Anti-spoofing will be disabled.", icon="ℹ️")
            midas = None
            transform = None
        st.success("Core models and components loaded successfully.", icon="✅")
        return loaded_model, encoder, embedder, detector, midas, transform
    except FileNotFoundError as e:
        st.error(f"Error loading model or encoder file: {e}. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory as app.py", icon="❌")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}", icon="❌")
        return None, None, None, None, None, None

loaded_model, encoder, embedder, detector, midas, midas_transform = load_models()

if loaded_model is not None and encoder is not None and embedder is not None and detector is not None:
    if midas is not None and midas_transform is not None:
        @st.cache_data
        def estimate_depth(img, _midas_model, _transform):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = _transform(img_rgb)
            if len(transformed.shape) == 3:
                input_batch = transformed.unsqueeze(0)
            else:
                input_batch = transformed
            with torch.no_grad():
                prediction = _midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            return prediction.cpu().numpy()

    @st.cache_data
    def get_embedding(face_img, _embedder_model):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = _embedder_model.embeddings(face_img)
        return yhat[0]

    input_method = st.radio("Choose input method:", ("Upload Image", "Take Photo with Webcam"))

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
            st.subheader("Processing Results:")
            col_det, col_anti = st.columns(2)
            with col_det:
                st.write("Performing Face Detection...")
                results = detector.detect_faces(image_rgb)
                image_with_box = image_rgb.copy()
                if len(results) == 0:
                    st.warning("No faces detected in the image.", icon="⚠️")
                else:
                    st.success(f"{len(results)} face(s) detected.", icon="✅")
                    x, y, w, h = results[0]['box']
                    y1, y2, x1, x2 = max(0, y), min(image_rgb.shape[0], y + h), max(0, x), min(image_rgb.shape[1], x + w)
                    box_color = (255, 0, 0)
                    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), box_color, 5)
                    st.image(image_with_box, caption="Detected Face", use_column_width=True)
                    face_rgb = image_rgb[y1:y2, x1:x2]
            with col_anti:
                is_spoof = False
                if midas is not None and midas_transform is not None:
                    st.write("Performing Anti-Spoofing Check...")
                    try:
                        depth_map = estimate_depth(image, midas, midas_transform)
                        face_depth_region = depth_map[max(0, y):min(depth_map.shape[0], y+h), max(0, x):min(depth_map.shape[1], x+w)]
                        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_U8C1)
                        depth_map_color = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2RGB)
                        st.image(depth_map_color, caption="Estimated Depth Map", use_column_width=True)
                        if face_depth_region.size > 0:
                            depth_variation = np.std(face_depth_region)
                            st.write(f"Depth standard deviation in face region: {depth_variation:.2f}")
                            spoofing_threshold = 1.0
                            if depth_variation < spoofing_threshold:
                                is_spoof = True
                                st.error("⚠️ Spoof attempt detected (flat image). Verification aborted.")
                            else:
                                st.success("✅ Real face detected. Proceeding with verification...")
                        else:
                            st.warning("Could not analyze depth in face region.", icon="⚠️")
                    except Exception as e:
                        st.error(f"An error occurred during anti-spoofing: {e}", icon="❌")
                else:
                    st.info("Anti-spoofing is disabled because the MiDaS model could not be loaded.", icon="ℹ️")
                if not is_spoof and len(results) > 0 and face_rgb.size > 0:
                    st.write("Performing Face Verification...")
                    target_size = (160, 160)
                    face_resized = cv2.resize(face_rgb, target_size)
                    test_im_embedding = get_embedding(face_resized, embedder)
                    test_im_embedding = np.expand_dims(test_im_embedding, axis=0)
                    prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                    predicted_class_index = np.argmax(prediction_proba)
                    prediction_confidence = prediction_proba[predicted_class_index]
                    st.subheader("Verification Outcome:")
                    confidence_threshold = 0.75
                    if 0 <= predicted_class_index < len(encoder.classes_):
                        predicted_class = encoder.inverse_transform([predicted_class_index])[0]
                        print(f"Verification Confidence for {predicted_class}: {prediction_confidence:.2f}")  # Print to terminal
                        if prediction_confidence >= confidence_threshold:
                            st.success(f"Predicted Identity: **{predicted_class}**", icon="✅")
                            st.write(f"Confidence: **{prediction_confidence:.2f}**")
                            box_color = (0, 255, 0)
                        else:
                            st.warning("Face not recognized", icon="⚠️")
                            st.write(f"Confidence: **{prediction_confidence:.2f}** (below {confidence_threshold:.0%})")
                            box_color = (0, 0, 255)
                    else:
                        print(f"Verification Confidence (Out of Bounds): {prediction_confidence:.2f}")  # Print to terminal
                        st.warning(f"Predicted class index ({predicted_class_index}) is out of bounds for the encoder.", icon="⚠️")
                        st.write("Face not recognized (prediction out of bounds)")
                        st.write(f"Confidence: **{prediction_confidence:.2f}**")
                        box_color = (0, 0, 255)
                    image_for_redraw = cv2.imdecode(file_bytes, 1)
                    image_for_redraw_rgb = cv2.cvtColor(image_for_redraw, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(image_for_redraw_rgb, (x1, y1), (x2, y2), box_color, 5)
                    st.image(image_for_redraw_rgb, caption="Processed Face", use_column_width=True)
                elif len(results) > 0 and face_rgb.size > 0:
                    st.warning("Face not verified due to spoofing detection or anti-spoofing error.", icon="⚠️")
                    st.image(image_with_box, caption="Detected Face (Spoof/Error)", use_column_width=True)
        else:
            st.warning("No faces detected in the image.", icon="⚠️")
    elif input_method == "Take Photo with Webcam":
        st.subheader("Take Photo with Webcam")
        captured_image = st.camera_input("Click 'Take Photo' to capture an image from your webcam.")
        if captured_image is not None:
            file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.subheader("Processing Results:")
            col_det, col_anti = st.columns(2)
            with col_det:
                st.write("Performing Face Detection...")
                results = detector.detect_faces(image_rgb)
                image_with_box = image_rgb.copy()
                if len(results) == 0:
                    st.warning("No faces detected in the captured image.", icon="⚠️")
                else:
                    st.success(f"{len(results)} face(s) detected.", icon="✅")
                    x, y, w, h = results[0]['box']
                    y1, y2, x1, x2 = max(0, y), min(image_rgb.shape[0], y + h), max(0, x), min(image_rgb.shape[1], x + w)
                    box_color = (255, 0, 0)
                    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), box_color, 5)
                    st.image(image_with_box, caption="Detected Face", use_column_width=True)
                    face_rgb = image_rgb[y1:y2, x1:x2]
            with col_anti:
                is_spoof = False
                if midas is not None and midas_transform is not None:
                    st.write("Performing Anti-Spoofing Check...")
                    try:
                        depth_map = estimate_depth(image, midas, midas_transform)
                        face_depth_region = depth_map[max(0, y):min(depth_map.shape[0], y+h), max(0, x):min(depth_map.shape[1], x+w)]
                        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_U8C1)
                        depth_map_color = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2RGB)
                        st.image(depth_map_color, caption="Estimated Depth Map", use_column_width=True)
                        if face_depth_region.size > 0:
                            depth_variation = np.std(face_depth_region)
                            st.write(f"Depth standard deviation in face region: {depth_variation:.2f}")
                            spoofing_threshold = 1.0
                            if depth_variation < spoofing_threshold:
                                is_spoof = True
                                st.error("⚠️ Spoof attempt detected (flat image). Verification aborted.")
                            else:
                                st.success("✅ Real face detected. Proceeding with verification...")
                        else:
                            st.warning("Could not analyze depth in face region.", icon="⚠️")
                    except Exception as e:
                        st.error(f"An error occurred during anti-spoofing: {e}", icon="❌")
                else:
                    st.info("Anti-spoofing is disabled because the MiDaS model could not be loaded.", icon="ℹ️")
                if not is_spoof and len(results) > 0 and face_rgb.size > 0:
                    st.write("Performing Face Verification...")
                    target_size = (160, 160)
                    face_resized = cv2.resize(face_rgb, target_size)
                    test_im_embedding = get_embedding(face_resized, embedder)
                    test_im_embedding = np.expand_dims(test_im_embedding, axis=0)
                    prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                    predicted_class_index = np.argmax(prediction_proba)
                    prediction_confidence = prediction_proba[predicted_class_index]
                    st.subheader("Verification Outcome:")
                    confidence_threshold = 0.75
                    if 0 <= predicted_class_index < len(encoder.classes_):
                        predicted_class = encoder.inverse_transform([predicted_class_index])[0]
                        print(f"Verification Confidence for {predicted_class}: {prediction_confidence:.2f}")  # Print to terminal
                        if prediction_confidence >= confidence_threshold:
                            st.success(f"Predicted Identity: **{predicted_class}**", icon="✅")
                            st.write(f"Confidence: **{prediction_confidence:.2f}**")
                            box_color = (0, 255, 0)
                        else:
                            st.warning("Face not recognized", icon="⚠️")
                            st.write(f"Confidence: **{prediction_confidence:.2f}** (below {confidence_threshold:.0%})")
                            box_color = (0, 0, 255)
                    else:
                        print(f"Verification Confidence (Out of Bounds): {prediction_confidence:.2f}")  # Print to terminal
                        st.warning(f"Predicted class index ({predicted_class_index}) is out of bounds for the encoder.", icon="⚠️")
                        st.write("Face not recognized (prediction out of bounds)")
                        st.write(f"Confidence: **{prediction_confidence:.2f}**")
                        box_color = (0, 0, 255)
                    image_for_redraw = cv2.imdecode(file_bytes, 1)
                    image_for_redraw_rgb = cv2.cvtColor(image_for_redraw, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(image_for_redraw_rgb, (x1, y1), (x2, y2), box_color, 5)
                    st.image(image_for_redraw_rgb, caption="Processed Face", use_column_width=True)
                elif len(results) > 0 and face_rgb.size > 0:
                    st.warning("Face not verified due to spoofing detection or anti-spoofing error.", icon="⚠️")
                    st.image(image_with_box, caption="Detected Face (Spoof/Error)", use_column_width=True)
        else:
            st.warning("No faces detected in the captured image.", icon="⚠️")
else:
    st.warning("Model loading failed. Please check the error messages above and ensure model files are in the correct location.", icon="❌")

st.sidebar.subheader("How to run the app:")
st.sidebar.write("1. Save the code above as `app.py`.")
st.sidebar.write("2. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory.")
st.sidebar.write("3. Open a terminal in that directory.")
st.sidebar.write("4. Run the command: `streamlit run app.py`")
st.sidebar.write("5. Access the app in your web browser.")