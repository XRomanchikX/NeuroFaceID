import json
from typing import NoReturn, List
import sys
import logging
import torch
from sklearn.neighbors import NearestNeighbors
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import numpy as np

type Features = None
type TrainedKNN = None
type FeaturesList = None
type Labels = None
type Filenames = None

class NeuroFaceID:
    def __init__(self, logger: int = 0) -> NoReturn:
        """
        Инциализирование `NeuroFaceID` для работы с ним.

        :type logger: bool
        :param logger: Логирование: Включение - 1 || Выключение - 0 (default) 
        """
        if logger == 1:
            logger = logging.INFO
        else:
            logger = logging.NOTSET

        logging.Logger("NeuroFaceID", logger)

        try:
            self.init_config_file()
            logging.info("Config was init!")

            self.dataset_dir = self.CONFIG['DATASET_DIR']
            self.n_neighbors = self.CONFIG['N_NEIGHBORS']
            self.device = self.CONFIG['DEVICE'] if self.CONFIG['DEVICE'] == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Device: {self.device}, DatasetDir: {self.dataset_dir}, NumNeighborns: {self.n_neighbors}")

            self.init_models()
            logging.info("Models was init!")

            self.features_list, self.labels, self.filenames = self.init_dataset()
            logging.info("Dataset was init!")

            self.knn = self.train_knn()
            logging.info("KNN was trained!")
            
        except:
            raise RuntimeError("Error code: 1\nPlease, send about this error author")
        
        logging.info("NeuroFaceID - ready to work!")

    def init_config_file(self) -> NoReturn:
        """
        Инциализация кофигурационного файла
        """
        try:
            with open("config.json", "r") as f:
                CONFIG = json.load(f)
            self.CONFIG = CONFIG

            logging.info("Config file was loaded")

        except FileNotFoundError:
            logging.info("No such config file: `config.json`")
            sys.exit(1)

    def init_models(self) -> NoReturn:
        """
        Инциализация моделей (MTCNN && InceptionResnetV1)
        """
        try:
            self.MTCNN = MTCNN(
                image_size=160, 
                margin=0, 
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], 
                factor=0.709, 
                post_process=True,
                device=self.device
            )
            self.resnet = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()
            logging.info("Models (MTCNN && InceptionResnetV1) was loaded!")
        except Exception as e:
            logging.info(f"Failed initialize models. Error: {e}")
            sys.exit(1)

    def extract_features(self, image) -> Features:
        try:
            img = Image.open(image).convert('RGB')
            
            # Detection face
            img_cropped = self.MTCNN(img)
            if img_cropped is None:
                logging.info(f"Face wasn't found in {image}")
                return None
                
            # if a face is detected -> placing image on GPU (If supported version)
            img_cropped = img_cropped.to(self.device)
            
            # Give embedding in an image
            with torch.no_grad():
                img_embedding = self.resnet(img_cropped.unsqueeze(0))
                
            return img_embedding.squeeze().cpu().numpy()  # Always return CPU tensor
        
        except Exception as e:
            logging.info(f"Error in processing {image}: {str(e)}")
            return None
    
    def init_dataset(self) -> List[FeaturesList, Labels, Filenames]:
        """
        Инциализация датасета
        """
        features_list, labels, filenames = [], [], []

        for class_name in os.listdir(self.dataset_dir):
            class_dir = os.path.join(self.dataset_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                features = self.extract_features(img_path)

                if features is not None:
                    features_list.append(features)
                    labels.append(class_name)
                    filenames.append(img_name)

                else:
                    logging.info(f"Skiped: {img_path}")

        if len(features_list) == 0:
            raise ValueError("There are no correct images of faces in the dataset")
        else:
            return features_list, labels, filenames
    
    def train_knn(self) -> TrainedKNN:
        X = np.array(self.features_list)

        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
        knn.fit(X)

        return knn

    def prompt(self, image: str):
        """
        :type image: str

        Необходимо указать прямой путь к файлу (к примеру: `testimage2.jpg`)
        """
        image_features = self.extract_features(image)

        if image_features is None:
            raise ValueError("Error when processing requested image")
        
        dist, indices = self.knn.kneighbors([image_features])

        print("\nNearst image's:")
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.filenames):
                print(f"Filename: {self.dataset_dir}/{self.labels[idx]}/{self.filenames[idx]}, Class: {self.labels[idx]}, Distance: {dist[0][i]:.4f}")
            else:
                print(f"Index {idx} not included in the range")