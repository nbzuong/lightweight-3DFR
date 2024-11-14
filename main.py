import cv2
import torch
import os
from models import mtcnn, inception_resnet_v1


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Load the MTCNN model for face detection
mtcnn_state_dict_path = os.path.join(os.path.dirname(__file__), 'weights/mtcnn/')
mtcnn = mtcnn.MTCNN(selection_method='probability', device=device, state_dict_path=mtcnn_state_dict_path)

# Load the Inception Resnet V1 model for face feature extraction
weight_paths = {
    "casia-webface": os.path.join("weights", "casia-webface", "20180408-102900-casia-webface.pt"),
    "vggface2": os.path.join("weights", "vggface2", "20180402-114759-vggface2.pt")
}
pretrained_model = "vggface2"
iresnet = inception_resnet_v1.InceptionResnetV1(pretrained=pretrained_model, 
                                                            state_dict_path=weight_paths[pretrained_model],
                                                            device=device).eval()



video_capture = cv2.VideoCapture(0)
face_count = 0
max_faces_to_save = 5

while True:

    result, video_frame = video_capture.read()
    if result is False:
        break

    face, box, prob = mtcnn(video_frame, return_prob=True)
    if box is not None:
        box = box[0].astype(int)
        # Draw the bounding box and probability
        cv2.rectangle(video_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(video_frame, f"{prob:.5f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
        if prob >= 0.99:
            face = face.unsqueeze(0).to(device)
            embedding = iresnet(face).detach().cpu()

    cv2.imshow("Face Recognition Project", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()