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
import streamlit.runtime

# Set the title of the application
st.title("Facial Verification and Anti-Spoofing System")

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
        # Only load MiDaS if torch is available and not running in Colab directly without a runtime
        # This check helps avoid errors when running in environments without full torch/cuda setup
        if torch.cuda.is_available() and streamlit.runtime.exists():
             try:
                midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # Use "MiDaS_small" for faster but lower quality
                midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                transform = midas_transforms.dpt_transform
             except Exception as midas_e:
                 st.warning(f"Could not load MiDaS model: {midas_e}. Anti-spoofing will be disabled.")
                 midas = None # Ensure midas is None if loading fails
                 transform = None # Ensure transform is None if loading fails
        else:
            st.info("MiDaS model loading skipped (either no CUDA or not running in Streamlit runtime). Anti-spoofing will be disabled.")
            midas = None
            transform = None


        st.success("Models and components loaded successfully.")
        return loaded_model, encoder, embedder, detector, midas, transform

    except FileNotFoundError as e:
        st.error(f"Error loading model or encoder file: {e}. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory as app.py")
        return None, None, None, None, None, None # Return None for all in case of error
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
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

    # --- Face Embedding Function ---
    @st.cache_data
    def get_embedding(face_img, _embedder_model):
        face_img = face_img.astype('float32') # 3D(160x160x3)
        face_img = np.expand_dims(face_img, axis=0)
        yhat= _embedder_model.embeddings(face_img)
        return yhat[0] # 512D image (1x1x512)


    # --- Streamlit UI ---
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

        # --- Face Detection and Bounding Box ---
        st.write("Performing Face Detection...")
        results = detector.detect_faces(image_rgb)

        if len(results) == 0:
            st.warning("No faces detected in the image.")
        else:
            st.success(f"{len(results)} face(s) detected.")
            # Assuming you want to process the first detected face
            x, y, w, h = results[0]['box']

            # Draw bounding box on the image copy for display
            image_with_box = image_rgb.copy()
            cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (255, 0, 0), 5) # Red box

            # Display the image with bounding box
            st.image(image_with_box, caption="Detected Face", use_column_width=True)

            # Extract the face
            face_rgb = image_rgb[y:y+h, x:x+w]

            # --- Anti-Spoofing ---
            if midas is not None and midas_transform is not None:
                st.write("Performing Anti-Spoofing Check...")
                try:
                    depth_map = estimate_depth(image, midas, midas_transform) # Use original BGR image for MiDaS
                    face_depth_region = depth_map[y:y+h, x:y+h+w, x:x+w] # Corrected slicing
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
                        predicted_class = encoder.inverse_transform([predicted_class_index])[0]
                        prediction_confidence = prediction_proba[predicted_class_index]


                        st.subheader("Verification Outcome:")
                        st.write(f"Predicted Identity: **{predicted_class}**")
                        st.write(f"Confidence: **{prediction_confidence:.2f}**")

                        # Optional: Display the extracted face (optional)
                        # st.image(face_resized, caption="Extracted Face (160x160)", use_column_width=True)


                except Exception as e:
                     st.error(f"An error occurred during anti-spoofing or verification: {e}")
            else:
                 st.info("Anti-spoofing is disabled because the MiDaS model could not be loaded.")
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
                 predicted_class = encoder.inverse_transform([predicted_class_index])[0]
                 prediction_confidence = prediction_proba[predicted_class_index]

                 st.subheader("Verification Outcome:")
                 st.write(f"Predicted Identity: **{predicted_class}**")
                 st.write(f"Confidence: **{prediction_confidence:.2f}**")


    else:
        st.write("Please upload an image to begin.")

else:
    st.warning("Model loading failed. Please check the error messages above and ensure model files are in the correct location.")

# --- Instructions for running the app ---
st.sidebar.subheader("How to run the app:")
st.sidebar.write("1. Save the code above as `app.py`.")
st.sidebar.write("2. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory.")
st.sidebar.write("3. Open a terminal in that directory.")
st.sidebar.write("4. Run the command: `streamlit run app.py`")
st.sidebar.write("5. Access the app in your web browser.")