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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from av import VideoFrame

# Set the title of the application
st.title("Facial Verification and Anti-Spoofing System")

st.write("Upload an image, use your webcam, or use a video file to perform facial verification and anti-spoofing.")

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
        # Load the saved SVM model and label encoder
        # Assuming these files are in the same directory as your app.py
        model_path = 'svm_model_160x160.pkl'
        encoder_path = 'label_encoder.pkl'

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None, None, None, None
        if not os.path.exists(encoder_path):
             st.error(f"Encoder file not found: {encoder_path}")
             return None, None, None, None, None, None

        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        # Initialize the FaceNet embedder
        embedder = FaceNet()
        # Initialize the MTCNN detector
        detector = MTCNN()

        # Load MiDaS model
        # Only load MiDaS if torch is available and not running in Colab directly without a runtime
        # This check helps avoid errors when running in environments without full torch/cuda setup
        # and also allows for a more robust loading within the Streamlit lifecycle
        if torch.cuda.is_available() and hasattr(streamlit, 'runtime') and streamlit.runtime.exists():
             try:
                midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # Use "MiDaS_small" for faster but lower quality
                midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                transform = midas_transforms.dpt_transform
                st.success("MiDaS model loaded successfully.")
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
        # Removed @st.cache_data as it causes issues with torch tensors
        def estimate_depth(img, _midas_model, _transform):
            """Estimate depth from RGB image using MiDaS."""
            # MiDaS expects RGB input
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = _transform(img_rgb)

            # Ensure input is a batch
            if len(transformed.shape) == 3:
                input_batch = transformed.unsqueeze(0)
            else:
                input_batch = transformed

            with torch.no_grad():
                prediction = _midas_model(input_batch)

                # Resize prediction to original image size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            return prediction.cpu().numpy()

    # --- Face Embedding Function ---
    # Removed @st.cache_data as it can cause issues with model predictions
    def get_embedding(face_img, _embedder_model):
        face_img = face_img.astype('float32') # 3D(160x160x3)
        face_img = np.expand_dims(face_img, axis=0)
        # 4D (Nonex160x160x3)
        yhat= _embedder_model.embeddings(face_img)
        return yhat[0] # 512D image (1x1x512)

    # --- Video Transformer for Webcam Feed ---
    class VideoProcessor(VideoTransformerBase):
        def __init__(self, detector, embedder, loaded_model, encoder, midas=None, midas_transform=None):
            self.detector = detector
            self.embedder = embedder
            self.loaded_model = loaded_model
            self.encoder = encoder
            self.midas = midas
            self.midas_transform = midas_transform
            self.target_size = (160, 160)
            self.spoofing_threshold = 1.0 # Fine-tune this threshold
            self.frame_count = 0
            self.process_interval = 5 # Process every 5th frame for optimization

        def recv(self, frame: VideoFrame) -> VideoFrame:
            img = frame.to_ndarray(format="bgr24") # Get frame as numpy array (BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for MTCNN

            self.frame_count += 1

            # Only process every process_interval frames for optimization
            if self.frame_count % self.process_interval != 0:
                 # Return the original frame if not processing
                 return VideoFrame.from_ndarray(img, format="bgr24")


            # --- Face Detection ---
            results = self.detector.detect_faces(img_rgb)

            if len(results) > 0:
                # Assuming you want to process the first detected face
                x, y, w, h = results[0]['box']

                # Ensure bounding box coordinates are within image bounds
                y1, y2, x1, x2 = max(0, y), min(img_rgb.shape[0], y + h), max(0, x), min(img_rgb.shape[1], x + w)

                # Draw bounding box on the frame
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red box

                # Extract the face using corrected coordinates
                face_rgb = img_rgb[y1:y2, x1:x2]

                # --- Anti-Spoofing ---
                is_spoof = False
                if self.midas is not None and self.midas_transform is not None:
                    try:
                        # Use original BGR image for MiDaS
                        depth_map = estimate_depth(img, self.midas, self.midas_transform)
                        # Corrected slicing for depth map
                        face_depth_region = depth_map[y1:y2, x1:x2]

                        # Check if the face depth region is valid before calculating std dev
                        if face_depth_region.size > 0:
                            depth_variation = np.std(face_depth_region)

                            if depth_variation < self.spoofing_threshold:
                                is_spoof = True
                    except Exception as e:
                         # st.warning(f"Error during anti-spoofing in frame: {e}") # Suppress frequent warnings
                         pass # Continue without anti-spoofing if there's an error

                if is_spoof:
                    # Display spoofing alert on the frame
                    cv2.putText(img_rgb, "SPOOF DETECTED", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Red text for spoof
                else:
                    # --- Face Verification ---
                    # Resize the face to the target size (160x160)
                    # Ensure face_rgb is not empty or invalid before resizing
                    if face_rgb.size > 0:
                        face_resized = cv2.resize(face_rgb, self.target_size)

                        # Get embedding and predict
                        test_im_embedding = get_embedding(face_resized, self.embedder)
                        test_im_embedding = np.expand_dims(test_im_embedding, axis=0) # Model expects a batch

                        prediction_proba = self.loaded_model.predict_proba(test_im_embedding)[0]
                        # Find the index of the class with the highest probability
                        predicted_class_index = np.argmax(prediction_proba)
                        # Get the class label using the encoder
                        predicted_class = self.encoder.classes_[predicted_class_index]
                        prediction_confidence = prediction_proba[predicted_class_index]

                        # Display prediction on the frame
                        text = f"{predicted_class}: {prediction_confidence:.2f}"
                        cv2.putText(img_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Green text for identity
                    else:
                        cv2.putText(img_rgb, "Invalid Face Region", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) # Yellow text


            # Convert back to BGR for returning as VideoFrame
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            return VideoFrame.from_ndarray(img_bgr, format="bgr24")


    # --- Input Method Selection ---
    input_method = st.radio("Choose input method:", ("Upload Image", "Webcam"))

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
                        # Use original BGR image for MiDaS
                        depth_map = estimate_depth(image, midas, midas_transform)
                        # Corrected slicing for depth map
                        face_depth_region = depth_map[max(0, y):min(depth_map.shape[0], y+h), max(0, x):min(depth_map.shape[1], x+w)]
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
                            # Find the index of the class with the highest probability
                            predicted_class_index = np.argmax(prediction_proba)
                            # Get the class label using the encoder
                            predicted_class = encoder.classes_[predicted_class_index]
                            prediction_confidence = prediction_proba[predicted_class_index]

                            st.subheader("Verification Outcome:")
                            st.write(f"Predicted Identity: **{predicted_class}**")
                            st.write(f"Confidence: **{prediction_confidence:.2f}**")

                    except Exception as e:
                         st.error(f"An error occurred during anti-spoofing or verification: {e}")
                else:
                     st.info("Anti-spoofing is disabled because the MiDaS model could not be loaded.")
                     # Proceed with verification if anti-spoofing is skipped
                     st.write("Performing Face Verification...")

                     # Resize the face to the target size (160,160)
                     target_size = (160, 160)
                     face_resized = cv2.resize(face_rgb, target_size)

                     # Get embedding and predict
                     test_im_embedding = get_embedding(face_resized, embedder)
                     test_im_embedding = np.expand_dims(test_im_embedding, axis=0) # Model expects a batch

                     prediction_proba = loaded_model.predict_proba(test_im_embedding)[0]
                     predicted_class_index = np.argmax(prediction_proba)
                     predicted_class = encoder.classes_[predicted_class_index]
                     prediction_confidence = prediction_proba[predicted_class_index]

                     st.subheader("Verification Outcome:")
                     st.write(f"Predicted Identity: **{predicted_class}**")
                     st.write(f"Confidence: **{prediction_confidence:.2f}**")


        else:
            st.write("Please upload an image to begin.")

    elif input_method == "Webcam":
        st.subheader("Webcam Feed")

        # Initialize session state for webcam running if it doesn't exist
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False

        # Create buttons to start and stop the webcam
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Webcam")
        with col2:
            stop_button = st.button("Stop Webcam")

        # Logic to toggle webcam state based on button clicks
        if start_button:
            st.session_state.webcam_running = True
        if stop_button:
            st.session_state.webcam_running = False

        # Display the webrtc_streamer only if webcam_running is True
        if st.session_state.webcam_running:
            st.write("Webcam is streaming and processing frames...")
            # Use the VideoProcessor class to process frames
            webrtc_ctx = webrtc_streamer(
                key="webcam-stream",
                video_processor_factory=lambda: VideoProcessor(detector, embedder, loaded_model, encoder, midas, midas_transform),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        else:
            st.write("Click 'Start Webcam' to begin the stream.")


else:
    st.warning("Model loading failed. Please check the error messages above and ensure model files are in the correct location.")

# --- Instructions for running the app ---
st.sidebar.subheader("How to run the app:")
st.sidebar.write("1. Save the code above as `app.py`.")
st.sidebar.write("2. Make sure 'svm_model_160x160.pkl' and 'label_encoder.pkl' are in the same directory.")
st.sidebar.write("3. Open a terminal in that directory.")
st.sidebar.write("4. Run the command: `streamlit run app.py`")
st.sidebar.write("5. Access the app in your web browser.")