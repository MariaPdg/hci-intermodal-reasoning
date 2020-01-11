import torch.nn as nn
import torch.nn.functional as F
import torch


class TeacherNet3query(nn.Module):
    def __init__(self):
        super(TeacherNet3query, self).__init__()
        self.linear0 = nn.Linear(in_features=2048, out_features=768)
        self.linear1 = nn.Linear(in_features=768, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=100)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, inputs):
        out = F.leaky_relu(self.linear0(inputs))
        out = F.leaky_relu(self.linear1(out))
        out = self.dropout1(out)
        out = F.leaky_relu(self.linear2(out))
        out = self.dropout2(out)
        out = self.linear3(out)
        out = F.normalize(out)
        return out


class TeacherNet3key(nn.Module):
    def __init__(self):
        super(TeacherNet3key, self).__init__()
        self.linear1 = nn.Linear(in_features=768, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=100)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, inputs):
        out = F.leaky_relu(self.linear1(inputs))
        out = self.dropout1(out)
        out = F.leaky_relu(self.linear2(out))
        out = self.dropout2(out)
        out = self.linear3(out)
        out = F.normalize(out)
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
        self.temp = 0.07
        self.dev = dev
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def return_logits(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.T.view(-1, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        sim_diff = l_pos.squeeze()-torch.max(l_neg, dim=1).values
        return logits, torch.argmax(logits, dim=1), torch.mean(sim_diff).item()

    def forward(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.T.view(-1, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = self.loss_fn(logits/self.temp, labels)
        return loss


class ContrastiveLossInBatch(nn.Module):
    def __init__(self, temp, dev):
        super(ContrastiveLossInBatch, self).__init__()
        self.temp = 0.07
        self.dev = dev
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def return_logits(self, q, k):
        N = q.size(0)
        C = q.size(1)
        mask = torch.eye(N)
        l_neg = None
        for idx in range(N):
            negative_sample_ids = [j for j in range(N) if mask[idx][j] < 1]
            if l_neg is None:
                l_neg = torch.mv(k[negative_sample_ids, :], q[idx]).view(1, N-1)
            else:
                l_neg = torch.cat([l_neg, torch.mv(k[negative_sample_ids, :], q[idx]).view(1, N-1)], dim=0)
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        sim_diff = l_pos.squeeze() - torch.max(l_neg, dim=1).values
        return logits, torch.argmax(logits, dim=1), torch.mean(sim_diff).item()

    def forward(self, q, k):
        N = q.size(0)
        C = q.size(1)
        mask = torch.eye(N)
        l_neg = None
        for idx in range(N):
            negative_sample_ids = [j for j in range(N) if mask[idx][j] < 1]
            if l_neg is None:
                l_neg = torch.mv(k[negative_sample_ids, :], q[idx]).view(1, N-1)
            else:
                l_neg = torch.cat([l_neg, torch.mv(k[negative_sample_ids, :], q[idx]).view(1, N-1)], dim=0)
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = self.loss_fn(logits/self.temp, labels)
        return loss


class ContrastiveLossReRank(nn.Module):
    def __init__(self, temp, dev):
        super(ContrastiveLossReRank, self).__init__()
        self.temp = 0.07
        self.dev = dev
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def return_logits(self, q, k, neg):
        N = q.size(0)
        C = q.size(1)
        K = neg[0].size(0)
        l_neg = None
        for idx in range(N):
            if l_neg is None:
                l_neg = torch.mv(neg[idx], q[idx]).view(1, K)
            else:
                l_neg = torch.cat([l_neg, torch.mv(neg[idx], q[idx]).view(1, K)], dim=0)
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        sim_diff = l_pos.squeeze() - torch.max(l_neg, dim=1).values
        return logits, torch.argmax(logits, dim=1), torch.mean(sim_diff).item()

    def forward(self, q, k, neg):
        N = q.size(0)
        C = q.size(1)
        K = neg[0].size(0)
        l_neg = None
        for idx in range(N):
            if l_neg is None:
                l_neg = torch.mv(neg[idx], q[idx]).view(1, K)
            else:
                l_neg = torch.cat([l_neg, torch.mv(neg[idx], q[idx]).view(1, K)], dim=0)
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = self.loss_fn(logits/self.temp, labels)
        return loss


class IdentificationLossInBatch(nn.Module):
    def __init__(self, dev="cpu"):
        super(IdentificationLossInBatch, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dev = dev

    def forward(self, q):
        N = q.size(0)
        C = q.size(1)
        mask = torch.eye(N)
        l_neg = None
        for idx in range(N):
            negative_sample_ids = [j for j in range(N) if mask[idx][j] < 1]
            if l_neg is None:
                l_neg = torch.mv(q[negative_sample_ids, :], q[idx]).view(1, N-1)
            else:
                l_neg = torch.cat([l_neg, torch.mv(q[negative_sample_ids, :], q[idx]).view(1, N-1)], dim=0)
        l_pos = torch.bmm(q.view(N, 1, C), q.view(N, C, 1))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = self.loss_fn(logits/0.07, labels)
        return loss

    def compute_diff(self, q):
        N = q.size(0)
        C = q.size(1)
        mask = torch.eye(N)
        l_neg = None
        for idx in range(N):
            negative_sample_ids = [j for j in range(N) if mask[idx][j] < 1]
            if l_neg is None:
                l_neg = torch.mv(q[negative_sample_ids, :], q[idx]).view(1, N - 1)
            else:
                l_neg = torch.cat([l_neg, torch.mv(q[negative_sample_ids, :], q[idx]).view(1, N - 1)], dim=0)
        l_pos = torch.bmm(q.view(N, 1, C), q.view(N, C, 1))
        sim_diff = l_pos.squeeze() - torch.max(l_neg, dim=1).values
        return torch.mean(sim_diff).item()


class CustomedQueue:
    def __init__(self, max_size=1024):
        self.neg_keys = []
        self.size = 0
        self.max_size = max_size

    def empty(self):
        return self.size == 0

    def enqueue(self, new_tensor):
        if self.size == 0:
            self.neg_keys = new_tensor
        else:
            self.neg_keys = torch.cat([self.neg_keys, new_tensor])
        self.size = self.neg_keys.size(0)

    def dequeue(self, howmany=1):
        if self.size > self.max_size:
            self.neg_keys = self.neg_keys[-self.max_size:]
            self.size = self.neg_keys.size(0)
            # print("m",self.neg_keys[-howmany:])
            # print("p",self.neg_keys[howmany:])

    def get_tensor(self, transpose=False):
        if transpose:
            return torch.transpose(self.neg_keys, 0, 1)
        else:
            return self.neg_keys


if __name__ == "__main__":
    u1 = torch.rand((3, 1))
    print(u1)
    loss2 = IdentificationLossInBatch()
    print(loss2(u1))
