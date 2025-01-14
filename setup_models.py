import torch
import os
from models.mtcnn import mtcnn
from models.facenet import facenet as fn
import onnxruntime as ort

def get_face_detector(weights_path=None, device = None):
    '''
    Returns the MTCNN model for face detection
    
    Args:
    - weights_path: str, path to the weights directory, default is None for automatic path
    - device: torch.device, device to run the model on
    '''
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "weights/mtcnn/")
    face_detector = mtcnn.MTCNN(image_size=120, margin=20, 
                                select_largest=True, post_process=False, 
                                device=device, state_dict_path=weights_path).eval()
    return face_detector

def get_face_feature2d(weights_path=None, pretrained_model="vggface2", device=None):
    '''
    Returns the Facenet model for 2d face feature extraction
    
    Args:
    - weights_path: str, path to the weights directory, default is None for automatic path
    - pretrained_model: str, name of the pretrained model, either "casia-webface" or "vggface2"
    - device: torch.device, device to run the model on
    '''
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "weights/facenet")
    
    if pretrained_model == "casia-webface":
        facenet_weight_paths = os.path.join(weights_path, "20180408-102900-casia-webface.pt")
    elif pretrained_model == "vggface2":
        facenet_weight_paths = os.path.join(weights_path, "20180402-114759-vggface2.pt")
    else:
        raise ValueError("Invalid pretrained model name for facenet. Choose either 'casia-webface' or 'vggface2'")
    
    facenet = fn.InceptionResnetV1(pretrained=pretrained_model, 
                                    state_dict_path=facenet_weight_paths,
                                    device=device).eval()
    return facenet

def get_face_feature3d(weights_path=None, model="mb1_120x120"):
    '''
    Returns the 3ddfa_v2 model for 3d face embedding \\
    The 3ddfa_v2 model is taken from https://github.com/cleardusk/3DDFA_V2 \\
    MIT License
    
    Args:
    - weights_path: str, path to the model file, default is None for automatic path
    - model : str, name of the model, either "mb1_120x120" or "mb05_120x120"\\
        mb1_120x120: 120x120 input size, 3.27M Params, 183.5M Macs\\
        mb05_120x120: 120x120 input size, 0.85M Params, 49.5M Macs
    '''
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "weights/3ddfa_v2")
    
    if model == "mb1_120x120":
        model_path = os.path.join(weights_path, "mb1_120x120.onnx")
    elif model == "mb05_120x120":
        model_path = os.path.join(weights_path, "mb05_120x120.onnx")
    else:
        raise ValueError("Invalid model name for 3ddfa_v2. Choose either 'mb1_120x120' or 'mb05_120x120'")
    
    tddfa = ort.InferenceSession(model_path)
    
    return tddfa
    
def get_embedding3d(onnx_model, img, device): # img with batch dimension added
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    embedding3d = onnx_model.run([output_name], {input_name: img})
    embedding3d = torch.tensor(embedding3d[0])
    return embedding3d.to(device)

def calculate_score(embedding_infer, embedding_db):
    '''
    Calculate the cosine similarity score between two embeddings
    
    Args:
    - embedding: torch.Tensor, 2d+3d embedding of the input face
    - embedding_db: torch.Tensor, 2d+3d embedding of the face in the database
    '''
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding_infer, embedding_db)
    score = cosine_similarity.item()
    return score