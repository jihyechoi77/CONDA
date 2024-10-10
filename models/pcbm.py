import torch
import torch.nn as nn


class PosthocLinearCBM(nn.Module):
    def __init__(self, concept_bank, backbone_name, idx_to_class=None, n_classes=5,
                 dim_emb=512, top_k=10,
                 alpha=0.5, lam=0.1):
        """
        PosthocCBM Linear Layer. 
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            backbone_name (str): Name of the backbone, e.g. clip:RN50.
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
            unsupervised (bool, optional): Whether to use unsupervised concept learning. False if using static concept bank
            n_concepts (int, optional): Number of concepts; parameters for unsupervised concept learning. Defaults to 100.
            dim_emb (int, optional): Dimensionality of the embedding; parameters for unsupervised concept learning. Defaults to 512.
            top_k (int, optional): Number of top-k concept scores to use for regularization; parameters for unsupervised concept learning. Defaults to 10.
            alpha (float, optional): Elastic net regularization parameter. Defaults to 0.5.
            lam (float, optional): Weight of the regularization term. Defaults to 0.1.
        """
        super(PosthocLinearCBM, self).__init__()
        # Get the concept information from the bank
        self.backbone_name = backbone_name
        self.d_embedding = dim_emb

        self.cavs_ = concept_bank.vectors
        self.intercepts_ = concept_bank.intercepts
        self.n_concepts = concept_bank.vectors.shape[0]
        self.cavs = nn.Linear(self.d_embedding, self.n_concepts)
        self.cavs.weight = nn.Parameter(concept_bank.vectors, requires_grad=False)
        self.cavs.bias = nn.Parameter(concept_bank.intercepts.squeeze(), requires_grad=False)

        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()

        self.n_classes = n_classes

        # Will be used to plot classifier weights nicely
        self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

        self.top_k = top_k
        self.alpha = alpha
        self.lam = lam


    def compute_dist(self, emb):
        # self.norms = torch.norm(self.cavs_, p=2, dim=1, keepdim=True).detach()
        # Computing the geometric margin to the decision boundary specified by CAV.
        # margins_ = (torch.matmul(self.cavs_, emb.T) + self.intercepts_) / (self.norms)
        # return margins_.T
        margins = (self.cavs(emb)) / (torch.norm(self.cavs.weight, p=2, dim=1, keepdim=True).T)
        return margins


    def forward(self, emb, return_concept=False):
        x = self.compute_dist(emb)  # concept scores
        out = self.classifier(x)
        if return_concept:
            return out, x
        return out


    def loss(self, out, target):
        cls_loss = nn.CrossEntropyLoss()(out, target)
        # elastic net penalty
        # loss = cls_loss
        l1_penalty = self.alpha
        l2_penalty = 1 - l1_penalty

        loss = cls_loss
        for param in self.classifier.parameters():
            loss += self.lam * (l1_penalty * torch.sum(torch.abs(param)) + l2_penalty * torch.sum(param ** 2))

        return loss

    def forward_projs(self, projs):
        return self.classifier(projs)
    
    def trainable_params(self):
        return list(self.classifier.parameters())
    
    def classifier_weights(self):
        return self.classifier.weight
    
    def set_weights(self, weights, bias):
        self.classifier.weight.data = torch.tensor(weights).to(self.classifier.weight.device)
        self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
        return 1

    def analyze_classifier(self, k=5, print_lows=False):
        weights = self.classifier.weight.clone().detach()
        output = []

        # if len(self.idx_to_class) == 2:
        #     weights = [weights.squeeze(), weights.squeeze()]
        
        for idx, cls in self.idx_to_class.items():
            cls_weights = weights[idx]
            topk_vals, topk_indices = torch.topk(cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.names[j] for j in topk_indices]
            analysis_str = [f"Class : {cls}"]
            for j, c in enumerate(topk_concepts):
                # analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
                analysis_str.append(f"\t {topk_indices[j]} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.names[j] for j in topk_indices]
                analysis_str = [f"Class : {cls}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(output)
        return analysis

    def initialize_before_adapt(self, pcbm):
        """Initialize the weights in concept projection and linear probing layers
        based on the weights of reference PCBM (before adaptation)
        """
        # Copy the weights from pcbm to pcbm_adapt for the shared part
        with torch.no_grad():
            if not self.unsupervised:
                # Copy the existing concept activation vectors (CAVs)
                self.cavs[:pcbm.n_concepts] = pcbm.cavs
            else:
                # Copy the weights for the unsupervised CAVs
                self.cavs.weight[:pcbm.n_concepts] = pcbm.cavs.weight
                self.cavs.bias[:pcbm.n_concepts] = pcbm.cavs.bias

        # Randomly initialize the new part of the weights
        nn.init.xavier_normal_(self.cavs.weight[pcbm.n_concepts:])
        nn.init.zeros_(self.cavs.bias[pcbm.n_concepts:])

        # Freeze the original part of the cavs weights
        for param in self.cavs.parameters():
            param.requires_grad = False

        # Unfreeze the new part (k) of the cavs weights
        for name, param in self.cavs.named_parameters():
            if "weight" in name:
                param[pcbm.n_concepts:].requires_grad = True
            if "bias" in name:
                param[pcbm.n_concepts:].requires_grad = True

        # Copy the classifier weights and randomly initialize the new part
        with torch.no_grad():
            self.classifier.weight[:, :pcbm.n_classes] = pcbm.classifier.weight
            self.classifier.bias = pcbm.classifier.bias

        # Randomly initialize the remaining part (k, L) of the classifier weights
        nn.init.xavier_normal_(self.classifier.weight[:, pcbm.n_classes:])


class PosthocHybridCBM(nn.Module):
    def __init__(self, bottleneck: PosthocLinearCBM, l2_penalty=0.001):
        """
        PosthocCBM Hybrid Layer. 
        Takes an embedding as the input, outputs class-level predictions.
        Uses both the embedding and the concept predictions.
        Args:
            bottleneck (PosthocLinearCBM): [description]
            l2_penalty (float, optional): L2 regularization parameter. Defaults to 0.001.
        """
        super(PosthocHybridCBM, self).__init__()
        # Get the concept information from the bank
        self.bottleneck = bottleneck
        # A single linear layer will be used as the classifier
        # if not self.bottleneck.unsupervised:
        #     self.d_embedding = self.bottleneck.cavs.weight.shape[1]
        # else:
        #     self.d_embedding = self.bottleneck.cavs.in_features
        
        self.n_classes = self.bottleneck.n_classes
        self.l2_penalty = l2_penalty
        self.residual_classifier = nn.Linear(self.bottleneck.d_embedding, self.n_classes)

    def forward(self, emb, return_dist=False):
        x = self.bottleneck.compute_dist(emb)
        out = self.bottleneck.classifier(x) + self.residual_classifier(emb)
        if return_dist:
            return out, x
        return out

    def loss(self, out, target, _):
        cls_loss = nn.CrossEntropyLoss()(out, target)
        # elastic net penalty
        loss = cls_loss + self.l2_penalty * (self.residual_classifier.weight ** 2).mean()
        return cls_loss, loss

    def trainable_params(self):
        return self.residual_classifier.parameters()
    
    def classifier_weights(self):
        return self.residual_classifier.weight

    def analyze_classifier(self):
        return self.bottleneck.analyze_classifier()

    def compute_dist(self, emb):
        return self.bottleneck.compute_dist(emb)

