import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from typing import List, Tuple, Dict, Any, Union, Optional

from melodysim.model.siamese_net import SiameseNet

# IMPORTANT: same songs are labelled with 0 in training (means low distance)
class LightningSiameseNet(L.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        train_classifier_gap = config["train_classifier_gap"]
        embedding_dim = config["siamese_emb_dim"]

        self.siamese_net = SiameseNet(embedding_dim=embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        self.criterion_triplet = nn.TripletMarginLoss(margin=2, p=2)
        self.criterion_classification = nn.BCEWithLogitsLoss()
        self.train_classifier_gap = train_classifier_gap

        if not hasattr(self, "audio_processor"):
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
        if not hasattr(self, "audio_model"):
            self.audio_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(self.device)

    def forward_siamese_net(self, anchors, positives, negatives):
        triplet_embeddings = torch.stack([
                self.siamese_net(anchors),
                self.siamese_net(positives),
                self.siamese_net(negatives)
            ], dim=1
        )
        return triplet_embeddings

    def forward_classifier(self, triplet_embeddings):
        diff_same = torch.abs(triplet_embeddings[:,0] - triplet_embeddings[:,1])
        diff_diff = torch.abs(triplet_embeddings[:,0] - triplet_embeddings[:,2])
        logit_same = self.classifier(diff_same).squeeze()
        logit_diff = self.classifier(diff_diff).squeeze()
        return logit_same, logit_diff

    @torch.no_grad()
    def _inference_step(self, sample1: torch.Tensor, sample2: torch.Tensor):
        B = sample1.shape[0]
        assert sample2.shape[0] == B

        out_embs1 = self.siamese_net(sample1)
        out_embs2 = self.siamese_net(sample2)
        diff = torch.abs(out_embs1 - out_embs2)
        logit = self.classifier(diff).squeeze()
        scores = torch.sigmoid(logit)
        return scores

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.siamese_net.parameters()) + list(self.classifier.parameters()),
            lr=1e-3
        )
        return optimizer
