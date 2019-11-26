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
        out = F.relu(self.linear3(out))
        return out


class RankingLossFunc(nn.Module):
    def __init__(self, delta):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, q, k, queue):
        assert (q.shape[0] == k.shape[0] > 0)
        loss = 0
        num_of_samples = q.shape[0]
        num_of_neg_samples = queue.size(0)

        for idx in range(num_of_samples):
            loss += sum([max(0, self.delta
                             - nn.functional.cosine_similarity(q[idx], k[idx], dim=-1)
                             + nn.functional.cosine_similarity(q[idx], queue[j], dim=-1))
                         for j in range(num_of_neg_samples)])
        return loss

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = 1 - nn.functional.cosine_similarity(x_reprets[i], y_reprets[j], dim=-1)
        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds


class ContrastiveLoss(nn.Module):
    def __init__(self, temp, dev):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.dev = dev

    def norm(self, vec):
        mean_vec = torch.mean(vec, dim=1)
        std_vec = torch.std(vec, dim=1)
        for i in range(vec.size(0)):
            vec[i] -= mean_vec[i]
            vec[i] /= std_vec[i]
        return vec

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        x_reprets = self.norm(x_reprets)
        y_reprets = self.norm(y_reprets)
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = torch.matmul(x_reprets[i], y_reprets[j])
        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds

    def forward(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.size(0)
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.view(C, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = F.cross_entropy(logits/self.temp, labels)
        return loss

    def forward3(self, X, Y, queue):
        assert (X.shape[0] == Y.shape[0] > 0)
        loss = 0
        num_of_samples = X.shape[0]

        for idx in range(num_of_samples):
            pos_logit = torch.matmul(X[idx], Y[idx])
            neg_logits = []
            for count in range(queue.size(0)):
                if count == 1:
                    neg_logits = torch.cat([neg_logits.view(1), torch.matmul(X[idx], queue[count]).view(1)])
                elif count > 1:
                    neg_logits = torch.cat([neg_logits, torch.matmul(X[idx], queue[count]).view(1)])
                else:
                    neg_logits = torch.matmul(X[idx], queue[count])

            logits = torch.cat([pos_logit.view(1), neg_logits])
            loss += F.cross_entropy(logits.view((1, logits.size(0)))/self.temp,
                                    torch.zeros(1, dtype=torch.long, device=self.dev))

        return loss/num_of_samples


if __name__ == "__main__":
    u1 = torch.rand((3, 4))
    u2 = torch.rand((3, 4))
    du = torch.rand((6, 4))
    du2 = torch.transpose(du, 0, 1)
    print(du2.size())
    loss = ContrastiveLoss(1, "cpu")
    out = loss.forward(u1, u2, du2)

    out = loss.forward3(u1, u2, du)
    print(u1)
    print(loss.norm(u1))
    print(loss.norm(loss.norm(u1)))