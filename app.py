import streamlit as st
import cv2
import torch
import numpy as np
import time
from setup_models import get_face_detector, get_face_feature2d, get_face_feature3d, get_embedding3d
from userProfileDB import UserProfileDB
from torch.nn.functional import cosine_similarity
import pygame

# Initialize Pygame mixer
pygame.mixer.init()
def play_sound(status):
    if status == "success":
        sound = pygame.mixer.Sound("sounds/success.wav")
    elif status == "failed":
        sound = pygame.mixer.Sound("sounds/failed.wav")
    elif status == "error":
        sound = pygame.mixer.Sound("sounds/error.wav")
    sound.play()
    pygame.time.delay(5000)

class FaceRecognitionSystem:
    def __init__(self, device, similarity_threshold=0.7):
        self.device = device
        self.face_detector = get_face_detector(device=device)
        self.face2d = get_face_feature2d(device=device)
        self.face3d = get_face_feature3d()
        self.user_profile_db = UserProfileDB()
        self.similarity_threshold = similarity_threshold

    def process_frame_for_inference(self, video_frame):
        """Processes a frame for inference mode."""
        face, box, prob = self.detect_face(video_frame)
        if face is not None:
            self.draw_bounding_box(video_frame, box, prob)
            if prob >= 0.99:
                embedding2d, embedding3d = self.extract_embeddings(face)
                return video_frame, embedding2d, embedding3d
        return video_frame, None, None

    def process_frame_for_registration(self, video_frame):
        """Processes a frame for registration mode."""
        face, box, prob = self.detect_face(video_frame)
        if face is not None:
            self.draw_bounding_box(video_frame, box, prob)
            if prob >= 0.999:
                embedding2d, embedding3d = self.extract_embeddings(face)
                return video_frame, embedding2d, embedding3d
        return video_frame, None, None

    def detect_face(self, video_frame):
        """Detects a face in the video frame."""
        face, box, prob = self.face_detector(video_frame, return_prob=True)
        if box is not None:
            box = box[0].astype(int)
            face = face.unsqueeze(0).to(self.device)
        return face, box, prob

    def draw_bounding_box(self, video_frame, box, prob):
        """Draws the bounding box and probability on the frame."""
        cv2.rectangle(video_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(video_frame, f"{prob:.5f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def extract_embeddings(self, face):
        """Extracts 2D and 3D embeddings from the face."""
        embedding2d = self.face2d(face).detach()
        embedding3d = get_embedding3d(self.face3d, face.cpu().numpy(), self.device)
        return embedding2d, embedding3d

    def register_mode(self, user_data, embedding2d, embedding3d):
        """Registers a new user with their embeddings."""
        try:
            self.user_profile_db.add_user(
                name=user_data["name"],
                dob=user_data["dob"],
                phone_number=user_data["phone_number"],
                face_embedding2d=embedding2d,
                face_embedding3d=embedding3d
            )
            return True
        except Exception as e:
            return False

    def inference_mode(self, embedding2d, embedding3d):
        """Performs inference to recognize a user."""
        embedding_infer = torch.cat((embedding2d, embedding3d), dim=1)

        user_embeddings = self.user_profile_db.get_embeddings_dataframe()
        if user_embeddings is None or user_embeddings.empty:
            st.warning("Database is empty")
            return None, None

        embeddings2d_db = torch.stack([e.squeeze(0) for e in user_embeddings["embedding2d"]])
        embeddings3d_db = torch.stack([e.squeeze(0) for e in user_embeddings["embedding3d"]])
        embeddings_db = torch.cat((embeddings2d_db, embeddings3d_db), dim=1).to(self.device)

        similarity_scores = cosine_similarity(embedding_infer, embeddings_db)

        best_match_index = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_match_index].item()

        if best_score > self.similarity_threshold:
            user_id = int(user_embeddings.iloc[best_match_index]["user_id"])

            try:
                user_info = self.user_profile_db.get_user_info(user_id)
                if user_info:
                    st.success(f"User found: {user_info}, Similarity: {best_score:.4f}")
                    return user_info, "success"
                else:
                    st.error(f"User with ID {user_id} not found in the database.")
                    return None, "failed"
            except Exception as e:
                st.error(f"Error retrieving user info: {e}")
                return None, "failed"
        else:
            st.warning("User not found")
            return None, None

# Streamlit App
def main():
    # Initialize the FaceRecognitionSystem
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # Use CPU for inference
    
    st.set_page_config(page_title="Face Recognition System", page_icon=":sunglasses:")
    st.title("Face Recognition System")
    mode = st.sidebar.selectbox("Mode", ["Not Selected", "Inference", "Register"])

    if mode == "Register":
        frs = FaceRecognitionSystem(device)
        st.sidebar.header("User Registration")
        user_name = st.sidebar.text_input("Name")
        
        # Use st.date_input for date of birth
        user_dob = st.sidebar.date_input("Date of Birth", format="YYYY-MM-DD")
        
        user_phone = st.sidebar.text_input("Phone Number")

        if st.sidebar.button("Register User"):
            if not all([user_name, user_dob, user_phone]):
                st.sidebar.error("Please fill in all fields.")
            else:
                # Convert date to string format
                user_dob_str = user_dob.strftime("%Y-%m-%d")
                
                user_data = {
                    "name": user_name,
                    "dob": user_dob_str,
                    "phone_number": user_phone
                }
                # Start capturing video for registration
                cap = cv2.VideoCapture(0)
                stframe = st.empty()

                # Countdown for registration
                countdown_time = 3  # seconds
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame")
                        break
                    
                    elapsed_time = time.time() - start_time
                    remaining_time = countdown_time - int(elapsed_time)

                    if remaining_time > 0:
                        # Display countdown on the frame
                        cv2.putText(frame, str(remaining_time), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        stframe.image(frame, channels="BGR")
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue  # Continue to the next iteration without processing
                    
                    frame, embedding2d, embedding3d = frs.process_frame_for_registration(frame)

                    if embedding2d is not None and embedding3d is not None:
                        success = frs.register_mode(user_data, embedding2d, embedding3d)
                        if success:
                            play_sound("success")
                        else:
                            play_sound("failed")
                        cap.release()
                        break
                    if frame is not None:
                        stframe.image(frame, channels="BGR")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()

    elif mode == "Inference":
        frs = FaceRecognitionSystem(device)
        st.sidebar.header("Inference")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        # Add a button to stop the inference
        stop_button = st.sidebar.button("Stop Inference")

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            frame, embedding2d, embedding3d = frs.process_frame_for_inference(frame)
            if embedding2d is not None and embedding3d is not None:
                _, status = frs.inference_mode(embedding2d, embedding3d)
                if status is not None:
                    play_sound(status)
            if frame is not None:
                stframe.image(frame, channels="BGR")

            # Check if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

if __name__ == "__main__":
    main()