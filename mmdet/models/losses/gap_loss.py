import torch


def flatten_onehot_labels_m(labels, num_classes, m):
    bs = labels.shape[0]
    onehot_labels = torch.zeros([bs, num_classes], device=labels.device)
    pos_labels = labels == m
    onehot_labels[pos_labels, labels[pos_labels]] = 1
    return onehot_labels.reshape(-1)


def flatten_onehot_labels(labels, num_classes):
    bs = labels.shape[0]
    onehot_labels = torch.zeros([bs, num_classes], device=labels.device)
    pos_labels = labels < num_classes
    onehot_labels[pos_labels, labels[pos_labels]] = 1
    return onehot_labels.reshape(-1)


class GAPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_samples, num_classes=1203, gamma=1.0, lam=1.0, delta_RS=0.50, eps=1e-10):
        targets = flatten_onehot_labels(labels, num_classes)
        gap_grads = torch.zeros(logits.shape).cuda()

        total_ranking_error = 0
        classes = torch.unique(labels)

        w = (1 / num_samples) ** gamma
        w = w / (torch.sum(w) + 1e-9) * num_classes

        if num_classes in classes:
            classes = classes[:-1]
        for m in classes:
            cls_targets = flatten_onehot_labels_m(labels, num_classes, m)
            pos_labels = (cls_targets > 0.)
            pos_logits = logits[pos_labels]
            pos_targets = cls_targets[pos_labels]
            pos_num = len(pos_logits)

            # Do not use neg with scores less than minimum pos logit
            # since changing its score does not have an effect on precision
            threshold_logit = torch.min(pos_logits) - delta_RS
            # Excluding positive samples of other categories from negative set of m
            relevant_neg_labels = ((targets == 0) & (logits >= threshold_logit))

            relevant_neg_logits = logits[relevant_neg_labels]
            relevant_neg_grad = torch.zeros(len(relevant_neg_logits)).cuda()
            ranking_error = torch.zeros(pos_num).cuda()
            prec = torch.zeros(pos_num).cuda()
            pos_grad = torch.zeros(pos_num).cuda()

            order = torch.argsort(pos_logits)
            for ii in order:
                pos_relations = pos_logits - pos_logits[ii]
                neg_relations = relevant_neg_logits - pos_logits[ii]

                if delta_RS > 0:
                    pos_relations = torch.clamp(pos_relations / (2 * delta_RS) + 0.5, min=0, max=1)
                    neg_relations = torch.clamp(neg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
                else:
                    pos_relations = (pos_relations >= 0).float()
                    neg_relations = (neg_relations >= 0).float()

                # R(ii) among P(m)
                rank_pos = torch.sum(pos_relations)
                # R(ii) among N
                FP_num = torch.sum(neg_relations)
                # R(ii)
                rank = rank_pos + FP_num

                ranking_error[ii] = FP_num / rank
                prec[ii] = rank_pos / rank

                if FP_num > eps:
                    # Gradient for ii in P(m)
                    pos_grad[ii] -= (1 - prec[ii])
                    # Gradient for N
                    relevant_neg_grad += (neg_relations / rank)

            total_ranking_error += ranking_error.mean() / len(classes)

            gap_grads[pos_labels] += (pos_grad * w[m] / (pos_num * len(classes))) * lam
            gap_grads[relevant_neg_labels] += (relevant_neg_grad * w[m] / (pos_num * len(classes))) * lam

        ctx.save_for_backward(gap_grads)

        return total_ranking_error

    @staticmethod
    def backward(ctx, out_grad1):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None, None, None, None