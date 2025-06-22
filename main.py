# Required imports
import torch
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from facenet_pytorch import MTCNN, InceptionResnetV1

# Check GPU version -> https://docs.pytorch.org/get-started/locally/
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Downloads FaceNet and MTCNN
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True,
    device=device
) # Parametres gived with docs

resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval() # Parametres gived with docs

# Pre-proccesing data with MTCNN
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Detection face
        img_cropped = mtcnn(img)
        if img_cropped is None:
            print(f"Face wasn't found in {image_path}")
            return None
            
        # if a face is detected -> placing image on GPU (If supported version)
        img_cropped = img_cropped.to(device)
        
        # Give embedding in an image
        with torch.no_grad():
            img_embedding = resnet(img_cropped.unsqueeze(0))
            
        return img_embedding.squeeze().cpu().numpy()  # Always return CPU tensor
    
    except Exception as e:
        print(f"Error in processing {image_path}: {str(e)}")
        return None

# Pre-loads dataset of face employee's
dataset_dir = "testdataset"
features_list = []
labels = []
filenames = []

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)

    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        features = extract_features(img_path)

        if features is not None:
            features_list.append(features)
            labels.append(class_name)
            filenames.append(img_name)

        else:
            print(f"Skiped: {img_path}")

# Check available data
if len(features_list) == 0:
    raise ValueError("There are no correct images of faces in the dataset")
X = np.array(features_list)
# DEBUG -> print(f"Dimension data: {X.shape} (512-embeddings)")

# Fit KNN (Cosine similarity)
n_neighbors = 3 # Count results (Default: 3)
knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
knn.fit(X)

# Example promt 
query_image = "testimage2.jpg"
query_features = extract_features(query_image)
if query_features is None:
    raise ValueError("Error when processing requested image")

# KNN (Find Nearst Neighbour)
distances, indices = knn.kneighbors([query_features])

print("\nNearst image's:")
for i in range(len(indices[0])):
    idx = indices[0][i]
    if idx < len(filenames):
        print(f"Filename: {dataset_dir}/{labels[idx]}/{filenames[idx]}, Class: {labels[idx]}, Distance: {distances[0][i]:.4f}")
    else:
        print(f"Index {idx} not included in the range")