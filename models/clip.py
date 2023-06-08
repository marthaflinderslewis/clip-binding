import clip
import torch


class CLIPModel(torch.nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.clip_model = clip_model
        self.device = device

    def forward(self, images, texts):
        return self.clip_model(images, texts)

    def encode_image(self, images):
        return self.clip_model.encode_image(images)

    def compute_text_representations(self, texts):
        tokenized_text = [
            clip.tokenize(["a photo of" + t for t in _texts]) for _texts in texts
        ]

        tokenized_text = torch.stack(tokenized_text, dim=0)
        bsz, num_captions, padding_length = tokenized_text.shape
        batch_texts = tokenized_text.view([-1, padding_length]).to(self.device)
        text_features = self.clip_model.encode_text(batch_texts)

        return text_features

    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        text_features = self.compute_text_representations(texts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view([bsz, num_captions, -1])

        batch_images = batch_images.to(self.device)
        batch_feat = self.encode_image(batch_images)

        batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)
        batch_feat = batch_feat.unsqueeze(2)

        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            text_features, batch_feat
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image


def get_clip_model(config, device):
    # load the model
    clip_model, preprocess = clip.load(config.clip_model, device=device)
    model = CLIPModel(clip_model, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    return model, optimizer
