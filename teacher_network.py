import torch.nn as nn
import torch.nn.functional as F
import torch


class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.linear1 = nn.Linear(in_features=2048, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=100)

    def forward(self, inputs):
        out = F.leaky_relu(self.linear1(inputs))
        out = F.leaky_relu(self.linear2(out))
        out = F.leaky_relu(self.linear3(out))
        out = F.normalize(out)
        return out


class TeacherNet2(nn.Module):
    def __init__(self):
        super(TeacherNet2, self).__init__()
        self.linear1 = nn.Linear(in_features=2048, out_features=100)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = F.normalize(out)
        return out


class TeacherNet3(nn.Module):
    def __init__(self):
        super(TeacherNet3, self).__init__()
        self.linear1 = nn.Linear(in_features=2048, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=100)

    def forward(self, inputs):
        out = F.leaky_relu(self.linear1(inputs))
        out = F.leaky_relu(self.linear2(out))
        out = self.linear3(out)
        out = F.normalize(out)
        return out


class TeacherNet4(nn.Module):
    def __init__(self):
        super(TeacherNet4, self).__init__()
        self.linear1 = nn.Linear(in_features=2048, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=100)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.linear2(out)
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


def norm(vec):
    mean_vec = torch.mean(vec, dim=1)
    std_vec = torch.std(vec, dim=1)
    for i in range(vec.size(0)):
        vec[i] -= mean_vec[i]
        vec[i] /= std_vec[i]
    return vec


class ContrastiveLoss(nn.Module):
    def __init__(self, temp, dev):
        super(ContrastiveLoss, self).__init__()
        self.temp = 0.07
        self.dev = dev
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        vec_size = x_reprets.shape[1]
        # x_reprets = norm(x_reprets)
        # y_reprets = norm(y_reprets)
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = nn.functional.cosine_similarity(x_reprets[i], y_reprets[j], dim=-1)
                # print(x_reprets[i], y_reprets[j], torch.matmul(x_reprets[i], y_reprets[j]))
        # print(embedding_loss)
        preds = torch.argmax(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds

    def return_logits(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.T.view(-1, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        return logits, torch.argmax(logits, dim=1)

    def forward(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.T.view(-1, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = self.loss_fn(logits/self.temp, labels)
        # print("loss", loss)
        # print("inside forward", logits.size())
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
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler

    train_img = torch.rand((100, 3))
    train_data = TensorDataset(train_img)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=2, num_workers=2)
    q = CustomedQueue(10)
    for step, batch in enumerate(train_dataloader):
        batch = batch[0]
        if q.empty():
            q.enqueue(batch)
            continue
        else:
            print("batch")
            print(batch)
            print("queue")
            print(q.get_tensor(False), q.get_tensor(False).size(), q.size)
            input()
            q.enqueue(batch)
        q.dequeue(2)

    # u1 = torch.rand((4, 3))
    # u2 = torch.rand((4, 3))
    # du = torch.rand((6, 3))
    # loss = ContrastiveLoss(1, "cpu")
    #
    # print(u1)
    # print(u2)
    # loss.predict(u1, u2)
    # print(loss.return_logits(u1, u2, du))

    # crossent = torch.nn.CrossEntropyLoss()
    # inp = torch.rand((4, 3), requires_grad=True)
    # print(inp)
    # print()
    # label = torch.zeros(4, dtype=torch.long)
    # for i in range(5):
    #     out = crossent(inp, label)
    #     print("loss is", out.item())
    #     out.backward()
    #     with torch.no_grad():
    #         inp -= inp.grad
    #         out = crossent(inp, label)
    #         print("loss is", out.item())
    #         print(inp)
    #         inp.grad.zero_()
    #         print()
