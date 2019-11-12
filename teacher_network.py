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
        print(embedding_loss)
        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds


class RankingLossFunc(nn.Module):
    def __init__(self, delta):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0])
        loss = 0
        num_of_samples = X.shape[0]

        mask = torch.eye(num_of_samples)
        for idx in range(num_of_samples):
            negative_sample_ids = [j for j in range(num_of_samples) if mask[idx][j] < 1]

            loss += sum([max(0, self.delta
                             - nn.functional.cosine_similarity(X[idx], Y[idx], dim=-1)
                             + nn.functional.cosine_similarity(X[idx], Y[j], dim=-1)) for j in negative_sample_ids])
        return loss
