import torch
import torch.nn as nn


class MatrixModel(nn.Module):
    def __init__(self, clip_model, adjtoi, nountoi, config, device):
        super().__init__()
        self.clip_model = clip_model
        self.adjtoi = adjtoi
        self.nountoi = nountoi
        self.device = device

        adjv_num = max(list(adjtoi.values())) + 1  # labels start from 0
        noun_num = max(list(nountoi.values())) + 1
        self.emb_dim = config.emb_dim
        self.a = nn.Embedding(adjv_num, self.emb_dim * self.emb_dim)
        self.n = nn.Embedding(noun_num, self.emb_dim)

    def forward(self, batch_img, texts):
        batch_img = batch_img.to(self.device)
        batch_feat = self.clip_model.encode_image(batch_img)
        batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)

        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])
        pairs = [pair.split() for pairs in texts for pair in pairs]
        pairs = [(self.adjtoi[a], self.nountoi[n]) for a, n in pairs]
        pairs = torch.tensor(pairs).to(self.device)
        img_labels = pairs.view([bsz, num_captions, -1])

        # Look up the embeddings for the positive and negative examples.
        # shape: (batch size, nbr contexts, emb dim)
        tgt_a_emb = self.a(img_labels[:, :, 0])
        batch_size, n_ctx, _ = tgt_a_emb.shape
        tgt_n_emb = self.n(img_labels[:, :, 1])

        # compose a and n
        adjs = tgt_a_emb.view((batch_size, n_ctx, self.emb_dim, self.emb_dim))
        nouns = tgt_n_emb.view(batch_size, n_ctx, self.emb_dim, 1)

        tgt_embs = (adjs.matmul(nouns)).squeeze(3)

        n_batch, n_ctx, emb_dim = tgt_embs.shape

        # View this as a 3-dimensional tensor, with
        # shape (batch size, 1, embedding dimension)
        batch_feat = batch_feat.view(n_batch, emb_dim, 1)

        # Compute the dot products between target word embeddings and context
        # embeddings. We express this as a batch matrix multiplication (bmm).
        # shape: (batch size, 1, nbr contexts)
        dots = tgt_embs.type(batch_feat.dtype).bmm(batch_feat)
        # View this result as a 2-dimensional tensor.
        # shape: (batch size, nbr contexts)
        dots = dots.view(n_batch, n_ctx)

        return dots


class MatrixRelObjModel(nn.Module):
    def __init__(self, clip_model, reltoi, nountoi, config, device):
        super().__init__()
        self.clip_model = clip_model
        self.reltoi = reltoi
        self.nountoi = nountoi
        self.device = device

        reln_num = max(list(reltoi.values())) + 1  # labels start from 0
        noun_num = max(list(nountoi.values())) + 1
        emb_dim = config.emb_dim
        self.r = nn.Embedding(reln_num, emb_dim * emb_dim)
        self.n = nn.Embedding(noun_num, emb_dim)

    def forward(self, batch_img, texts):
        batch_img = batch_img.to(self.device)
        batch_feat = self.clip_model.encode_image(batch_img)
        batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)

        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])
        pairs = [pair.split() for pairs in texts for pair in pairs]
        pairs = [
            (self.nountoi[subj], self.reltoi[rel], self.nountoi[obj])
            for subj, rel, obj in pairs
        ]
        pairs = torch.tensor(pairs).to(self.device)
        img_labels = pairs.view([bsz, num_captions, -1])

        # Look up the embeddings for the positive and negative examples.
        # shape: (batch size, nbr contexts, emb dim)
        tgt_s_emb = self.n(img_labels[:, :, 0])
        tgt_r_emb = self.r(img_labels[:, :, 1])
        tgt_o_emb = self.n(img_labels[:, :, 2])

        n_batch, n_ctx, emb_dim = tgt_s_emb.shape

        # compose s, r, o
        rels = tgt_r_emb.view((n_batch, n_ctx, emb_dim, emb_dim))
        objs = tgt_o_emb.view(n_batch, n_ctx, emb_dim, 1)

        vps = rels.matmul(objs)
        vps = vps.squeeze(3)
        tgt_embs = tgt_s_emb * vps

        # tgt_embs = (adjs.matmul(nouns)).squeeze(3)
        n_batch, n_ctx, emb_dim = tgt_embs.shape

        # View this as a 3-dimensional tensor, with
        # shape (batch size, 1, embedding dimension)
        batch_feat = batch_feat.view(n_batch, emb_dim, 1)

        # Transpose the tensor for matrix multiplication
        # shape: (batch size, emb dim, nbr contexts)
        # tgt_embs=tgt_embs.transpose(1, 2)

        # Compute the dot products between target word embeddings and context
        # embeddings. We express this as a batch matrix multiplication (bmm).
        # shape: (batch size, 1, nbr contexts)
        dots = tgt_embs.type(batch_feat.dtype).bmm(batch_feat)
        # View this result as a 2-dimensional tensor.
        # shape: (batch size, nbr contexts)
        dots = dots.view(n_batch, n_ctx)

        return dots
