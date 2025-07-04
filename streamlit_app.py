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
# import streamlit.runtime # Removed as it's not available in non-Streamlit environments

# Set the title of the application
st.set_page_config(page_title="Facial Verification and Anti-Spoofing", layout="wide")

# --- Header and Branding ---
col1, col2 = st.columns([1, 4])
with col1:
    # Add a placeholder for your branding logo
    # Replace 'path/to/your/logo.png' with the actual path to your logo file
    try:
        # Assuming the logo is in the same directory or an 'images' subdirectory
        logo_path = 'logo.png' # Replace with your logo file name
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.warning("Branding logo not found. Place 'logo.png' in the app directory.")
    except Exception as e:
        st.warning(f"Error loading logo: {e}")

with col2:
    st.title("Facial Verification and Anti-Spoofing System")

st.markdown("---") # Add a horizontal rule for visual separation

st.write("Upload an image or use your webcam to perform facial verification and anti-spoofing.")

# --- Load Models and Encoder (Cache these to avoid reloading on each interaction) ---
@st.cache_resource
def load_models():
    # Load the saved SVM model
    # Ensure these paths are correct relative to where you run the Streamlit app
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
        # Initialize the FaceNet embedder
        embedder = FaceNet()
        # Initialize the MTCNN detector
        detector = MTCNN()

        # Load MiDaS model
        # Only load MiDaS if torch is available (assuming Streamlit runtime is present when CUDA is)
        if torch.cuda.is_available():
             try:
                midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # Use "MiDaS_small" for faster but lower quality
                midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                transform = midas_transforms.dpt_transform
                st.success("MiDaS model loaded successfully.", icon="✅")
             except Exception as midas_e:
                 st.warning(f"Could not load MiDaS model: {midas_e}. Anti-spoofing will be disabled.", icon="⚠️")
                 midas = None # Ensure midas is None if loading fails
                 transform = None # Ensure transform is None if loading fails
        else:
            st.info("MiDaS model loading skipped (no CUDA available). Anti-spoofing will be disabled.", icon="ℹ️")
            midas = None
            transform = None


        st.success("Core models and components loaded successfully.", icon="✅")
        return loaded_model, encoder, embedder, detector, midas, transform

    except FileNotFoundError as e:
        st.error(f"Error loading model or encoder file: {e}. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory as app.py", icon="❌")
        return None, None, None, None, None, None # Return None for all in case of error
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}", icon="❌")
        return None, None, None, None, None, None # Return None for all in case of error


loaded_model, encoder, embedder, detector, midas, midas_transform = load_models()

# Check if models were loaded successfully before proceeding
if loaded_model is not None and encoder is not None and embedder is not None and detector is not None: # Midas can be None if loading failed

    # --- Depth Estimation Function ---
    # This function is only needed if MiDaS was loaded successfully
    if midas is not None and midas_transform is not None:
        @st.cache_data
        def estimate_depth(img, _midas_model, _transform):
            """Estimate depth from RGB image using MiDaS."""
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = _transform(img_rgb)

            if len(transformed.shape) == 3:
                input_batch = transformed.unsqueeze(0)
            else:
                input_batch = input_batch.unsqueeze(0) # Ensure input is a batch

            with torch.no_grad():
                prediction = _midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            return prediction.cpu().numpy()

    # --- Face Embedding Function ---
    @st.cache_data
    def get_embedding(face_img, _embedder_model):
        face_img = face_img.astype('float32') # 3D(160x160x3)
        face_img = np.expand_dims(face_img, axis=0)
        yhat= _embedder_model.embeddings(face_img)
        return yhat[0] # 512D image (1x1x512)


    # --- Input Method Selection ---
    input_method = st.radio("Choose input method:", ("Upload Image", "Take Photo with Webcam"))

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Convert the image to RGB for processing and displaying
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the uploaded image
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

            st.subheader("Processing Results:")

            # Use columns for better layout of detection and anti-spoofing
            col_det, col_anti = st.columns(2)

            with col_det:
                # --- Face Detection and Bounding Box ---
                st.write("Performing Face Detection...")
                results = detector.detect_faces(image_rgb)

                if len(results) == 0:
                    st.warning("No faces detected in the image.", icon="⚠️")
                else:
                    st.success(f"{len(results)} face(s) detected.", icon="✅")
                    # Assuming you want to process the first detected face
                    x, y, w, h = results[0]['box']

                    # Ensure bounding box coordinates are within image bounds
                    y1, y2, x1, x2 = max(0, y), min(image_rgb.shape[0], y + h), max(0, x), min(image_rgb.shape[1], x + w)

                    # Draw bounding box on the image copy for display
                    image_with_box = image_rgb.copy()
                    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (255, 0, 0), 5) # Red box

                    # Display the image with bounding box
                    st.image(image_with_box, caption="Detected Face", use_column_width=True)

                    # Extract the face
                    face_rgb = image_rgb[y1:y2, x1:x2]

            with col_anti:
                # --- Anti-Spoofing ---
                if midas is not None and midas_transform is not None:
                    st.write("Performing Anti-Spoofing Check...")
                    try:
                        # Use original BGR image for MiDaS
                        depth_map = estimate_depth(image, midas, midas_transform)
                        # Corrected slicing for depth map
                        face_depth_region = depth_map[max(0, y):min(depth_map.shape[0], y+h), max(0, x):min(depth_map.shape[1], x+w)]

                        # Normalize depth map for visualization (optional, but often helpful)
                        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_U8C1)
                        depth_map_color = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2RGB)
                        st.image(depth_map_color, caption="Estimated Depth Map", use_column_width=True)


                        if face_depth_region.size > 0:
                            depth_variation = np.std(face_depth_region)

                            st.write(f"Depth standard deviation in face region: {depth_variation:.2f}")

                            # Anti-spoofing threshold (fine-tune as needed)
                            # This threshold should be determined based on experimentation with real and spoof data
                            # A value around 1.0 was used in previous analysis, but might need adjustment
                            spoofing_threshold = 1.0 # Example threshold

                            if depth_variation < spoofing_threshold:
                                st.error("⚠️ Spoof attempt detected (flat image). Verification aborted.")
                            else:
                                st.success("✅ Real face detected. Proceeding with verification...")

                                # --- Face Verification ---
                                st.write("Performing Face Verification...")

                                # Resize the face to the target size (160x160)
                                target_size = (160, 160)
                                face_resized = cv2.resize(face_rgb, target_size)

                                # Get embedding and predict
                                test_im_embedding = get_embedding(face_resized, embedder)
                                test_im_embedding = np.expand_dims(test_im_embedding, axis=0) # Model expects a batch

                                prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                                predicted_class_index = np.argmax(prediction_proba)
                                prediction_confidence = prediction_proba[predicted_class_index]

                                st.subheader("Verification Outcome:")
                                confidence_threshold = 0.75 # 75%

                                # Check if the predicted index is within the bounds of the encoder's classes
                                if 0 <= predicted_class_index < len(encoder.classes_):
                                    predicted_class = encoder.inverse_transform([predicted_class_index])[0]

                                    if prediction_confidence >= confidence_threshold:
                                        st.success(f"Predicted Identity: **{predicted_class}**", icon="✅")
                                        st.write(f"Confidence: **{prediction_confidence:.2f}**")
                                    else:
                                        st.warning("Face not recognized", icon="⚠️")
                                        st.write(f"Confidence: **{prediction_confidence:.2f}** (below {confidence_threshold:.0%})")
                                else:
                                    st.warning(f"Predicted class index ({predicted_class_index}) is out of bounds for the encoder.", icon="⚠️")
                                    st.write("Face not recognized (prediction out of bounds)")
                                    st.write(f"Confidence: **{prediction_confidence:.2f}**")


                        else:
                            st.warning("Could not analyze depth in face region.", icon="⚠️") # Handle case where face_depth_region is empty


                    except Exception as e:
                         st.error(f"An error occurred during anti-spoofing or verification: {e}", icon="❌")
                else:
                     st.info("Anti-spoofing is disabled because the MiDaS model could not be loaded.", icon="ℹ️")
                     # Proceed with verification if anti-spoofing is skipped
                     st.write("Performing Face Verification...")

                     # Resize the face to the target size (160x160)
                     target_size = (160, 160)
                     face_resized = cv2.resize(face_rgb, target_size)

                     # Get embedding and predict
                     test_im_embedding = get_embedding(face_resized, embedder)
                     test_im_embedding = np.expand_dims(test_im_embedding, axis=0) # Model expects a batch

                     prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                     predicted_class_index = np.argmax(prediction_proba)
                     prediction_confidence = prediction_proba[predicted_class_index]

                     st.subheader("Verification Outcome:")
                     confidence_threshold = 0.75 # 75%

                     # Check if the predicted index is within the bounds of the encoder's classes
                     if 0 <= predicted_class_index < len(encoder.classes_):
                         predicted_class = encoder.inverse_transform([predicted_class_index])[0]

                         if prediction_confidence >= confidence_threshold:
                             st.success(f"Predicted Identity: **{predicted_class}**", icon="✅")
                             st.write(f"Confidence: **{prediction_confidence:.2f}**")
                         else:
                             st.warning("Face not recognized", icon="⚠️")
                             st.write(f"Confidence: **{prediction_confidence:.2f}** (below {confidence_threshold:.0%})")
                     else:
                         st.warning(f"Predicted class index ({predicted_class_index}) is out of bounds for the encoder.", icon="⚠️")
                         st.write("Face not recognized (prediction out of bounds)")
                         st.write(f"Confidence: **{prediction_confidence:.2f}**")


        else:
            st.warning("No faces detected in the image.", icon="⚠️")

    elif input_method == "Take Photo with Webcam":
        st.subheader("Take Photo with Webcam")
        captured_image = st.camera_input("Click 'Take Photo' to capture an image from your webcam.")

        if captured_image is not None:
            # Read the captured image
            file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Convert the image to RGB for processing and displaying
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # No need to display the captured image again, camera_input does it

            st.subheader("Processing Results:")

            # Use columns for better layout of detection and anti-spoofing
            col_det, col_anti = st.columns(2)

            with col_det:
                # --- Face Detection and Bounding Box ---
                st.write("Performing Face Detection...")
                results = detector.detect_faces(image_rgb)

                if len(results) == 0:
                    st.warning("No faces detected in the captured image.", icon="⚠️")
                else:
                    st.success(f"{len(results)} face(s) detected.", icon="✅")
                    # Assuming you want to process the first detected face
                    x, y, w, h = results[0]['box']

                    # Ensure bounding box coordinates are within image bounds
                    y1, y2, x1, x2 = max(0, y), min(image_rgb.shape[0], y + h), max(0, x), min(image_rgb.shape[1], x + w)


                    # Draw bounding box on the image copy for display
                    image_with_box = image_rgb.copy()
                    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (255, 0, 0), 5) # Red box

                    # Display the image with bounding box
                    st.image(image_with_box, caption="Detected Face", use_column_width=True)

                    # Extract the face
                    face_rgb = image_rgb[y1:y2, x1:x2]

            with col_anti:
                # --- Anti-Spoofing ---
                if midas is not None and midas_transform is not None:
                    st.write("Performing Anti-Spoofing Check...")
                    try:
                        # Use original BGR image for MiDaS
                        depth_map = estimate_depth(image, midas, midas_transform)
                        # Corrected slicing for depth map
                        face_depth_region = depth_map[max(0, y):min(depth_map.shape[0], y+h), max(0, x):min(depth_map.shape[1], x+w)]

                        # Normalize depth map for visualization (optional, but often helpful)
                        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_U8C1)
                        depth_map_color = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2RGB)
                        st.image(depth_map_color, caption="Estimated Depth Map", use_column_width=True)


                        if face_depth_region.size > 0:
                            depth_variation = np.std(face_depth_region)

                            st.write(f"Depth standard deviation in face region: {depth_variation:.2f}")

                            # Anti-spoofing threshold (fine-tune as needed)
                            spoofing_threshold = 1.0 # Example threshold

                            if depth_variation < spoofing_threshold:
                                st.error("⚠️ Spoof attempt detected (flat image). Verification aborted.")
                            else:
                                st.success("✅ Real face detected. Proceeding with verification...")

                                # --- Face Verification ---
                                st.write("Performing Face Verification...")

                                # Resize the face to the target size (160x160)
                                target_size = (160, 160)
                                face_resized = cv2.resize(face_rgb, target_size)

                                # Get embedding and predict
                                test_im_embedding = get_embedding(face_resized, embedder)
                                test_im_embedding = np.expand_dims(test_im_embedding, axis=0) # Model expects a batch

                                prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                                predicted_class_index = np.argmax(prediction_proba)
                                prediction_confidence = prediction_proba[predicted_class_index]

                                st.subheader("Verification Outcome:")
                                confidence_threshold = 0.75 # 75%

                                # Check if the predicted index is within the bounds of the encoder's classes
                                if 0 <= predicted_class_index < len(encoder.classes_):
                                    predicted_class = encoder.inverse_transform([predicted_class_index])[0]

                                    if prediction_confidence >= confidence_threshold:
                                        st.success(f"Predicted Identity: **{predicted_class}**", icon="✅")
                                        st.write(f"Confidence: **{prediction_confidence:.2f}**")
                                    else:
                                        st.warning("Face not recognized", icon="⚠️")
                                        st.write(f"Confidence: **{prediction_confidence:.2f}** (below {confidence_threshold:.0%})")
                                else:
                                     st.warning(f"Predicted class index ({predicted_class_index}) is out of bounds for the encoder.", icon="⚠️")
                                     st.write("Face not recognized (prediction out of bounds)")
                                     st.write(f"Confidence: **{prediction_confidence:.2f}**")


                    except Exception as e:
                         st.error(f"An error occurred during anti-spoofing or verification: {e}", icon="❌")
                else:
                     st.info("Anti-spoofing is disabled because the MiDaS model could not be loaded.", icon="ℹ️")
                     # Proceed with verification if anti-spoofing is skipped
                     st.write("Performing Face Verification...")

                     # Resize the face to the target size (160x160)
                     target_size = (160, 160)
                     face_resized = cv2.resize(face_rgb, target_size)

                     # Get embedding and predict
                     test_im_embedding = get_embedding(face_resized, embedder)
                     test_im_embedding = np.expand_dims(test_im_embedding, axis=0) # Model expects a batch

                     prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                     predicted_class_index = np.argmax(prediction_proba)
                     prediction_confidence = prediction_proba[predicted_class_index]

                     st.subheader("Verification Outcome:")
                     confidence_threshold = 0.75 # 75%

                     # Check if the predicted index is within the bounds of the encoder's classes
                     if 0 <= predicted_class_index < len(encoder.classes_):
                         predicted_class = encoder.inverse_transform([predicted_class_index])[0]

                         if prediction_confidence >= confidence_threshold:
                             st.success(f"Predicted Identity: **{predicted_class}**", icon="✅")
                             st.write(f"Confidence: **{prediction_confidence:.2f}**")
                         else:
                             st.warning("Face not recognized", icon="⚠️")
                             st.write(f"Confidence: **{prediction_confidence:.2f}** (below {confidence_threshold:.0%})")
                     else:
                         st.warning(f"Predicted class index ({predicted_class_index}) is out of bounds for the encoder.", icon="⚠️")
                         st.write("Face not recognized (prediction out of bounds)")
                         st.write(f"Confidence: **{prediction_confidence:.2f}**")

        else:
            st.warning("No faces detected in the captured image.", icon="⚠️")


else:
    st.warning("Model loading failed. Please check the error messages above and ensure model files are in the correct location.", icon="❌")

# --- Instructions for running the app ---
st.sidebar.subheader("How to run the app:")
st.sidebar.write("1. Save the code above as `app.py`.")
st.sidebar.write("2. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory.")
st.sidebar.write("3. Open a terminal in that directory.")
st.sidebar.write("4. Run the command: `streamlit run app.py`")
st.sidebar.write("5. Access the app in your web browser.")