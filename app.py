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
    """Plays a sound based on the status."""
    sounds = {
        "success": "sounds/success.wav",
        "failed": "sounds/failed.wav",
        "error": "sounds/error.wav"
    }
    sound_file = sounds.get(status)
    if sound_file:
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
        pygame.time.delay(3000)

class FaceRecognitionSystem:
    def __init__(self, device, similarity_threshold=0.7):
        self.device = device
        self.face_detector = get_face_detector(device=device)
        self.face2d = get_face_feature2d(device=device)
        self.face3d = get_face_feature3d()
        self.user_profile_db = UserProfileDB()
        self.similarity_threshold = similarity_threshold

    def detect_face(self, video_frame):
        """Detects a face in the video frame."""
        face, box, prob = self.face_detector(video_frame, return_prob=True)
        if box is not None:
            box = box[0].astype(int)
            face = face.unsqueeze(0).to(self.device)
        return face, box, prob

    def draw_bounding_box(self, video_frame, box, prob):
        """Draws a bounding box and probability on the frame."""
        cv2.rectangle(video_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(video_frame, f"{prob:.5f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def extract_embeddings(self, face):
        """Extracts 2D and 3D embeddings from the face."""
        embedding2d = self.face2d(face).detach()
        embedding3d = get_embedding3d(self.face3d, face.cpu().numpy(), self.device)
        return embedding2d, embedding3d

    def register_user(self, user_data, embedding2d, embedding3d):
        """Registers a new user with their embeddings."""
        try:
            success = self.user_profile_db.add_user(
                name=user_data["name"],
                dob=user_data["dob"],
                phone_number=user_data["phone_number"],
                face_embedding2d=embedding2d,
                face_embedding3d=embedding3d
            )
            if success is not None:
                return "success"
            else:
                return "error"
        except Exception as e:
            return "error"

    def recognize_user(self, embedding2d, embedding3d):
        """Performs inference to recognize a user."""
        embedding_infer = torch.cat((embedding2d, embedding3d), dim=1)
        user_embeddings = self.user_profile_db.get_embeddings_dataframe()
        if user_embeddings is None or user_embeddings.empty:
            st.warning("Database is empty")
            return None, None

        embeddings2d_db = torch.stack([e.squeeze(0).to(self.device) for e in user_embeddings["embedding2d"]])
        embeddings3d_db = torch.stack([e.squeeze(0).to(self.device) for e in user_embeddings["embedding3d"]])
        embeddings_db = torch.cat((embeddings2d_db, embeddings3d_db), dim=1)

        similarity_scores = cosine_similarity(embedding_infer, embeddings_db)
        best_match_index = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_match_index].item()

        if best_score > self.similarity_threshold:
            user_id = int(user_embeddings.iloc[best_match_index]["user_id"])
            user_info = self.user_profile_db.get_user_info(user_id)
            return user_info, "success"
        return None, "failed"

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

def main():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # Use CPU for inference
    st.set_page_config(page_title="Face Recognition System", page_icon=":sunglasses:")
    st.title("Face Recognition System")
    mode = st.sidebar.selectbox("Mode", ["Not Selected", "Inference", "Register"])

    if mode == "Not Selected":
        st.info("Please select a mode from the sidebar.")
        return

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if mode == "Register":
        frs = FaceRecognitionSystem(device)
        user_name = st.sidebar.text_input("Name")
        user_dob = st.sidebar.date_input("Date of Birth", format="YYYY-MM-DD")
        user_phone = st.sidebar.text_input("Phone Number")

        if st.sidebar.button("Register User"):
            if not all([user_name, user_dob, user_phone]):
                st.sidebar.error("Please fill in all fields.")
                return

            user_data = {
                "name": user_name,
                "dob": user_dob.strftime("%Y-%m-%d"),
                "phone_number": user_phone
            }
            countdown_time = 3
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame")
                    break

                elapsed_time = time.time() - start_time
                frame, embedding2d, embedding3d = frs.process_frame_for_registration(frame)
                if elapsed_time < countdown_time:
                    cv2.putText(frame, str(countdown_time - int(elapsed_time)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    if embedding2d is not None and embedding3d is not None:
                        if frs.register_user(user_data, embedding2d, embedding3d) == "success":
                            play_sound("success")
                        elif frs.register_user(user_data, embedding2d, embedding3d) == "error":
                            play_sound("error")
                        break

                stframe.image(frame, channels="BGR")

    elif mode == "Inference":
        frs = FaceRecognitionSystem(device)
        frame_counter = 0
        inference_wait = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            frame, embedding2d, embedding3d = frs.process_frame_for_inference(frame)
            if embedding2d is not None and embedding3d is not None:
                if frame_counter >= inference_wait:
                    user_info, status = frs.recognize_user(embedding2d, embedding3d)
                    print(user_info)
                    play_sound(status)
                    frame_counter = 0
                else:
                    frame_counter += 1

            stframe.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
