import torch.nn as nn
import torch.nn.functional as F
import torch


class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.linear1 = nn.Linear(in_features=2048, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out = F.relu(self.linear2(out))
        out = F.softmax(self.linear3(out), dim=1)
        return out

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = 1 - nn.functional.cosine_similarity(x_reprets[i], y_reprets[j], dim=-1)
        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds


class RankingLossFunc(nn.Module):
    def __init__(self, delta):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0] > 0)
        loss = 0
        num_of_samples = X.shape[0]

        mask = torch.eye(num_of_samples)
        for idx in range(num_of_samples):
            negative_sample_ids = [j for j in range(num_of_samples) if mask[idx][j] < 1]

            loss += sum([max(0, self.delta
                             - nn.functional.cosine_similarity(X[idx], Y[idx], dim=-1)
                             + nn.functional.cosine_similarity(X[idx], Y[j], dim=-1)) for j in negative_sample_ids])
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temp, dev):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.dev = dev

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0] > 0)
        loss = 0
        num_of_samples = X.shape[0]

        mask = torch.eye(num_of_samples)
        for idx in range(num_of_samples):
            negative_sample_id = [j for j in range(num_of_samples) if mask[idx][j] < 1]
            pos_logit = torch.matmul(X[idx], Y[idx])
            neg_logits = []
            for count, j in enumerate(negative_sample_id):
                if count == 1:
                    neg_logits = torch.cat([neg_logits.view(1), torch.matmul(X[idx], Y[j]).view(1)])
                elif count > 1:
                    neg_logits = torch.cat([neg_logits, torch.matmul(X[idx], Y[j]).view(1)])
                else:
                    neg_logits = torch.matmul(X[idx], Y[j])

            logits = torch.cat([pos_logit.view(1), neg_logits])
            loss += F.cross_entropy(logits.view((1, logits.size(0)))/self.temp, torch.zeros(1, dtype=torch.long, device=self.dev))

        return loss

    def forward2(self, X, Y):
        assert (X.shape[0] == Y.shape[0] > 0)
        loss = 0
        N = X.shape[0]
        C = X.shape[1]

        mask = torch.eye(N)
        negative_sample_ids = []
        for idx in range(N):
            negative_sample_ids.append([j for j in range(N) if mask[idx][j] < 1])

        l_pos = torch.matmul(X.view((N, 1, C)), Y.view((N, C, 1)))
        l_neg = torch.dot(X.view((N, C)), Y[negative_sample_ids, :].view((C, N-1)))

        print(Y)
        print(Y[negative_sample_ids, :].view((C, N-1)))

        return loss


if __name__ == "__main__":
    u1 = torch.rand((4, 4))
    u2 = torch.rand((4, 4))
    loss = ContrastiveLoss(1)
    out = loss.forward2(u1, u2)
    print(out)
