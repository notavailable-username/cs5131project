import numpy as np
from models.few_shot_models.siamese import SiameseModel
from models.few_shot_models.prototypical import PrototypicalModel
from models.few_shot_models.matching import MatchingModel
from models.few_shot_models.maml import MAMLModel
from models.few_shot_models.relation_network import RelationNetworkModel
from utils.ensemble import ensemble_vote

class FewShotEnsemble:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        # Instantiate five different few-shot models.
        self.models = [
            SiameseModel(),
            PrototypicalModel(),
            MatchingModel(),
            MAMLModel(),
            RelationNetworkModel()
        ]

    def predict(self, image_patch):
        """
        Run each model's prediction on the image patch and aggregate
        the results using ensemble voting.
        """
        predictions = [model.predict(image_patch) for model in self.models]
        final_class, final_conf = ensemble_vote(predictions)
        if final_conf < self.confidence_threshold:
            final_class = "uncertain"
        return final_class, final_conf

if __name__ == "__main__":
    import cv2
    dummy_image = cv2.imread("dummy.jpg")
    ensemble = FewShotEnsemble()
    cls, conf = ensemble.predict(dummy_image)
    print("Predicted:", cls, "Confidence:", conf)
