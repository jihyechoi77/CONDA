
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.jit
import torch.optim as optim


from adaptation.utils import compute_mahalanobis, sample_estimator

class Adaptation(nn.Module):
    """adapts a model by various loss options during testing.

    Once wrapped, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, steps=1, episodic=False,
                 reg_l2norm = 3,
                 opt='adam', lr=0.01,
                 num_res_concepts=5,
                 jointly=False):
        super().__init__()
        self.model = model

        if not jointly:
            classifier_params_to_update = self.model.classifier.parameters()
        else:
            classifier_params_to_update = list(self.model.classifier.parameters()) + list(self.residual_bottleneck.parameters()) + list(self.residual_classifier.parameters())
        if opt == 'sgd':
            self.optimizer_cavs = torch.optim.SGD(self.model.cavs.parameters(), lr=lr)
            self.optimizer_classifier = torch.optim.SGD(classifier_params_to_update, lr=lr)
        elif opt == 'adam':
            self.optimizer_cavs = torch.optim.Adam(self.model.cavs.parameters(), lr=lr)
            self.optimizer_classifier = torch.optim.Adam(classifier_params_to_update, lr=lr)
        else:
            raise NotImplementedError

        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.train_stats = None

        # Note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer_cavs)

        # regularization coefficients
        self.reg_l2norm = reg_l2norm

        self.jointly = jointly # whether to jointly adapt the classifier and the residual concept bottleneck; if false, sequentially

        # for RCB
        self.num_res_concepts = num_res_concepts
        self.residual_bottleneck = nn.Linear(self.model.d_embedding, num_res_concepts).cuda()
        self.residual_classifier = nn.Linear(num_res_concepts, self.model.n_classes).cuda()
        self.residual_optimizer = torch.optim.SGD(
            list(self.residual_bottleneck.parameters()) + list(self.residual_classifier.parameters()),
            lr=lr
        )


    def forward(self, x, y, adapt_cavs, adapt_classifier, adapt_rcb, return_concept=True):
        """
        :param x: feature embedding
        :param statistics: dictionary of {"mean": mean, "precision": inverse of covariance}, each of which is the list of per-class statistics
        :return:
            outputs: logits
            concepts: concept scores
        """
        if self.episodic:
            self.reset()


        concepts_res = None
        if adapt_cavs:
            # adapt the concept bottleneck
            for param in self.model.cavs.parameters():
                param.requires_grad = True
            for param in self.model.classifier.parameters():
                param.requires_grad = False

            # for _ in range(10):
            for _ in range(self.steps):
                # self.model.source_stats = sample_estimator(self.model.projs_train, self.model.train_lbls, self.model.num_classes)
                outputs, concepts = forward_and_adapt(x, y, self.model, self.optimizer_cavs,
                                                      method='mahalanobis',
                                                      # method='crossentropy',
                                                      reg_frob=True,
                                                      param_orig=self.model_state)

        if self.jointly:
            for param in self.model.cavs.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.residual_bottleneck.parameters():
                param.requires_grad = True
            for param in self.residual_classifier.parameters():
                param.requires_grad = True

            outputs, concepts, concepts_res = forward_and_adapt_jointly(x, y, self.model,
                                                                      self.residual_bottleneck, self.residual_classifier,
                                                                      self.optimizer_classifier,
                                                                      method='crossentropy',
                                                                        reg_sparsity=True)

        else:
            if adapt_classifier:
                # adjust the weights in the classifier
                for param in self.model.cavs.parameters():
                    param.requires_grad = False
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

                for _ in range(self.steps):
                    outputs, concepts = forward_and_adapt(x, y, self.model, self.optimizer_classifier,
                                                          method='crossentropy',
                                                          param_orig=self.model_state,
                                                          reg_sparsity=True)

            if adapt_rcb:
                for param in self.model.cavs.parameters():
                    param.requires_grad = False
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
                for param in self.residual_bottleneck.parameters():
                    param.requires_grad = True
                for param in self.residual_classifier.parameters():
                    param.requires_grad = True

                for _ in range(self.steps):
                    outputs, concepts, concepts_res = forward_and_adapt_residual(x, y,
                                                                       self.model,
                                                                       self.residual_bottleneck, self.residual_classifier,
                                                                       self.residual_optimizer)
                # with torch.no_grad():
                    # _, concepts = self.model(x, return_concept=True)


        if return_concept:
            return outputs, concepts, concepts_res
        else:
            return outputs


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# @torch.jit.script
def cross_entropy(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(yhat, y.squeeze(1))

def frobenius(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    x1 = ...  # (d, m) tensor with original values
    x2 = ...  # (d, m) tensor with updated values
    """
    return torch.linalg.matrix_norm(x1-x2, ord='fro')

def elasticnet_regularization(w: torch.Tensor, alpha=0.99) -> torch.Tensor:
    """
    calculate the elastic net regularization term for model weights
    :param w: weights in a layer
    :return: loss value for elastic net regularization
    """
    return alpha * torch.sum(torch.abs(w)) + (1-alpha) * torch.sum(w ** 2)


def cos_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    x1 = ...  # (d, m) tensor with original values
    x2 = ...  # (d, m) tensor with updated values
    """
    x1 = x1 / torch.linalg.vector_norm(x1, dim=1, ord=2, keepdim=True)
    x2 = x2 / torch.linalg.vector_norm(x2, dim=1, ord=2, keepdim=True)

    if not x1.shape[0] == 1: # when num_concept == 1 for Residual concept bottleneck
        # Calculate pairwise cosine similarity within x1
        cos_sim_within_x1 = F.cosine_similarity(x1.unsqueeze(1), x1.unsqueeze(0), dim=2)
        # For within x1, exclude the self-similarity along the diagonal by filling it with zeros
        mask = torch.eye(x1.size(0), device=x1.device).bool()
        cos_sim_within_x1.masked_fill_(mask, 0)
        loss_within_x1 = 1 - cos_sim_within_x1.sum() / (x1.size(0) * (x1.size(0) - 1))  # Exclude diagonal

    else:
        loss_within_x1 = 0.

    # Calculate pairwise cosine similarity between x1 and x2
    cos_sim_between_x1_x2 = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=2)

    # Since we want dissimilarity for the loss, we can subtract the cosine similarities from 1
    # and sum or average them as needed for the loss. Note that cosine similarity ranges from -1 to 1,
    # with 1 meaning completely similar, and -1 meaning completely dissimilar.


    # Sum or average the cosine similarities to form the loss components
    loss_between_x1_x2 = 1 - cos_sim_between_x1_x2.mean()

    # Combine the losses
    total_loss = loss_within_x1 + loss_between_x1_x2

    return total_loss


def coherency(scores, top_k=10):
    # coefficients for concept learing regularization terms
    top_k_coeff = 1 / top_k

    # Get the top 10 values along the last dimension (encouraged to be large)
    k = min(top_k, scores.shape[0])
    top_k_values = torch.topk(scores, k=k, dim=0, largest=True,
                              sorted=True).values  # dim=(top-k, n_concepts)
    # Calculate the mean of the top-k values
    mean_top_k_values = torch.mean(top_k_values)
    return top_k_coeff * mean_top_k_values

def mahalanobis(x: torch.Tensor, y: torch.Tensor, statistics: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
    """Increasing inter-class separation and decreasing intra-class variation
    x: feature embeddings
    y: predicted labels
    statistics: per-class mean and precision
    """

    # compute intra-class variation
    means = torch.stack([statistics["mean"][_y] for _y in y])
    precisions = torch.stack([statistics["precision"][_y] for _y in y])
    intra = compute_mahalanobis(x, means, precisions)
    # del means, precisions
    # torch.cuda.empty_cache()


    # compute inter-class variation
    inter = []
    num_classes = len(statistics["mean"])
    batch_size = x.shape[0]
    for c in range(num_classes):
        means = statistics["mean"][c].unsqueeze(0).repeat(batch_size,1)
        precisions = statistics["precision"][c].unsqueeze(0).repeat(batch_size,1, 1)
        inter.append(compute_mahalanobis(x, means, precisions))

    # inter = inter.sum()
    inter = torch.stack(inter).sum(0) - intra
    inter /= (num_classes-1)

    return (3*torch.log(intra) - torch.log(inter))



@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, y, model, optimizer, method,
                      param_orig=None,
                      reg_sparsity=False, reg_frob=False):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """

    optimizer.zero_grad()


    if method == 'mahalanobis':
        outputs, concepts = model(x, return_concept=True)
        means = [torch.from_numpy(np_array).cuda() for np_array in model.source_stats['mean']]
        precisions = [torch.from_numpy(np_array).cuda() for np_array in model.source_stats['precision']]
        statistics_ch = {'mean': means, 'precision': precisions}
        loss = mahalanobis(concepts, y, statistics_ch).mean(0)

        # means_embs = [torch.from_numpy(np_array).cuda() for np_array in model.source_stats_embs['mean']]
        # precisions_embs = [torch.from_numpy(np_array).cuda() for np_array in model.source_stats_embs['precision']]
        # statistics_embs_ch = {'mean': means_embs, 'precision': precisions_embs}
        # loss_embs = mahalanobis(x, y, statistics_embs_ch).mean(0)


    elif method == 'crossentropy':
        outputs, concepts = model(x, return_concept=True)
        loss = cross_entropy(y, outputs).mean(0)


    if reg_frob:
        # ensure that CAVs don't deviate too much
        loss_frob = frobenius(model.cavs.weight, param_orig['cavs.weight'])
        coeff_frob = 0.1 # torch.abs(loss_mahal) / (loss_frob + 1e-6)
        loss += coeff_frob * loss_frob

    if reg_sparsity:
        # elastic-net penalty for sparsity
        coeff_elastic = 1
        for param in model.classifier.parameters():
            loss +=  coeff_elastic * (coeff_elastic / model.n_classes / model.n_concepts) * elasticnet_regularization(param)

    # entropy minimization
    # loss = 0.5*softmax_entropy(outputs).mean(0)




    # model.classifier.weight.grad = None
    # model.classifier.bias.grad = None

    loss.backward()
    optimizer.step()

    # print(f'main loss: {loss}')


    return outputs, concepts


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_residual(x, y,
                               model,
                               res_bottleneck, res_classifier,
                               optimizer,
                               l2_penalty=0.01):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()

    concepts_res = res_bottleneck(x)
    # normalize
    concepts_res = concepts_res / torch.linalg.vector_norm(concepts_res, dim=1, ord=2, keepdim=True)
    outputs_res = res_classifier(concepts_res)

    with torch.no_grad():
        outputs, concepts = model(x, return_concept=True)

    loss = cross_entropy(y, outputs+outputs_res).mean(0)
    # print(cross_entropy(outputs, y).mean(0), loss)

    loss -= coherency(concepts_res, top_k=10)

    coeff_sim = 1.
    loss += coeff_sim*cos_similarity(res_bottleneck.weight, model.cavs.weight)
    # print(loss)
    loss.backward()
    optimizer.step()

    return outputs+outputs_res, concepts, concepts_res


@torch.enable_grad()
def forward_and_adapt_jointly(x, y,
                            model, res_bottleneck, res_classifier,
                            optimizer, method,
                            ):
    """Forward and adapt model on batch of data.
    Jointly update the classifier and the residual concept bottleneck
    """

    optimizer.zero_grad()

    # Main Classifier
    if method == 'crossentropy':
        outputs, concepts = model(x, return_concept=True)
        loss = cross_entropy(y, outputs).mean(0)

        # elastic-net penalty for sparsity
        coeff_elastic = 2.0
        for param in model.cavs.parameters():
            loss += coeff_elastic * (
                        coeff_elastic / model.n_classes / model.n_concepts) * elasticnet_regularization(param)

    # Residual Concept Bottleneck
    concepts_res = res_bottleneck(x)
    # normalize
    concepts_res = concepts_res / torch.linalg.vector_norm(concepts_res, dim=1, ord=2, keepdim=True)
    outputs_res = res_classifier(concepts_res)

    loss += cross_entropy(y, outputs + outputs_res).mean(0)
    # print(cross_entropy(outputs, y).mean(0), loss)

    loss -= coherency(concepts_res, top_k=10)

    coeff_sim = 0.1
    loss += coeff_sim * cos_similarity(res_bottleneck.weight, model.cavs.weight)

    loss.backward()
    optimizer.step()

    return outputs + outputs_res, concepts, concepts_res


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    # optimizer.load_state_dict(optimizer_state)
    optimizer[0].load_state_dict(optimizer_state[0])
