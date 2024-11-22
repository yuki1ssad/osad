import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM


class HolisticHead(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(HolisticHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)


class PlainHead(nn.Module):
    def __init__(self, in_dim, topk_rate=0.1):
        super(PlainHead, self).__init__()
        self.scoring = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)
        self.topk_rate = topk_rate

    def forward(self, x):
        x = self.scoring(x)
        x = x.view(int(x.size(0)), -1)
        topk = max(int(x.size(1) * self.topk_rate), 1)
        x = torch.topk(torch.abs(x), topk, dim=1)[0]
        x = torch.mean(x, dim=1).view(-1, 1)
        return x


class CompositeHead(PlainHead):
    def __init__(self, in_dim, topk=0.1):
        super(CompositeHead, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU())

    def forward(self, x, ref):
        ref = torch.mean(ref, dim=0).repeat([x.size(0), 1, 1, 1])
        x = ref - x
        x = self.conv(x)
        x = super().forward(x)
        return x

class subNet(nn.Module):
    def __init__(self, in_dim, emb_dim=128):
        super(subNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(14 * 14, emb_dim)
        self.fc2 = nn.Linear(7 * 7, emb_dim) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(int(x.size(0)), -1)
        if x.shape[-1] == 14 * 14:
            x = self.fc1(x)
        elif  x.shape[-1] == 7 * 7:
            x = self.fc2(x)
        else:
            raise NotImplementedError
        return F.normalize(x)
    
class AssistNet(nn.Module):
    def __init__(self, input_size=128, hidden_size=512, output_size=1):
        super(AssistNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.abs(x)

class DRA(nn.Module):
    def __init__(self, cfg, backbone="resnet18"):
        super(DRA, self).__init__()
        self.cfg = cfg
        self.feature_extractor = build_feature_extractor(backbone, cfg)
        self.in_c = NET_OUT_DIM[backbone]
        self.holistic_head = HolisticHead(self.in_c)
        self.seen_head = PlainHead(self.in_c, self.cfg.topk)
        self.pseudo_head = PlainHead(self.in_c, self.cfg.topk)
        self.composite_head = CompositeHead(self.in_c, self.cfg.topk)

        self.exchangeNet = subNet(self.in_c)
        self.protos = F.normalize(torch.randn(size=(3, 128), requires_grad=True)).cuda()
        self.assist1 = AssistNet()
        self.assist2 = AssistNet()
        self.assist3 = AssistNet()
        self.criterian = nn.BCEWithLogitsLoss()

    def forward(self, image, label):
        image_pyramid = list()
        for i in range(self.cfg.total_heads):
            image_pyramid.append(list())

        embedsList = list()
        for s in range(self.cfg.n_scales):
            image_scaled = F.interpolate(image, size=self.cfg.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)  # torch.Size([53, 512, 14, 14]) torch.Size([53, 512, 7, 7])

            embedsList.append(self.exchangeNet(feature))

            ref_feature = feature[:self.cfg.nRef, :, :, :]
            feature = feature[self.cfg.nRef:, :, :, :]

            if self.training:
                normal_scores = self.holistic_head(feature)                     # torch.Size([48, 1])
                abnormal_scores = self.seen_head(feature[label != 2])           # torch.Size([28, 1])
                dummy_scores = self.pseudo_head(feature[label != 1])            # torch.Size([32, 1])
                comparison_scores = self.composite_head(feature, ref_feature)   # torch.Size([48, 1])
            else:
                normal_scores = self.holistic_head(feature)
                abnormal_scores = self.seen_head(feature)
                dummy_scores = self.pseudo_head(feature)
                comparison_scores = self.composite_head(feature, ref_feature)
            for i, scores in enumerate([normal_scores, abnormal_scores, dummy_scores, comparison_scores]):
                image_pyramid[i].append(scores)

        for i in range(self.cfg.total_heads):
            image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
            image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)

        if self.training:
            embeddings = F.normalize(torch.cat(embedsList, dim=0))
            embedLabel = self.getEmbedLabel(label)
            embedGroups, lossComp = self.updateProtos(embeddings, embedLabel)
            assistScores, assistloss = self.assistScore(embedGroups)
            lossProto = self.lossProto()
            tmpLoss = lossComp + lossProto + assistloss
            # image_pyramid.append(assistScore)

            return image_pyramid, tmpLoss
        else:
            return image_pyramid
    
    def getEmbedLabel(self, label):
        label1 = torch.cat((label, torch.zeros(self.cfg.nRef, dtype=label.dtype).cuda()))
        label1_combined = torch.cat((label1, label1))
        return label1_combined


    def updateProtos(self, embeddings, embedLabel):
        # updateProtos
        n_embeddings = embeddings[embedLabel==0]
        a_embeddings = embeddings[embedLabel!=0]
        beta = 0.1
        protoNum = self.protos.shape[0]
        similarity_matrix = torch.cosine_similarity(self.protos.unsqueeze(1), n_embeddings.unsqueeze(0), dim=2)
        # closest_proto_indices = torch.argmax(similarity_matrix, dim=0)
        # 防止某个proto没有对应的embedding
        _, closest_proto_indices = torch.max(similarity_matrix, dim=0)
        assigned_protos = torch.zeros(protoNum, dtype=torch.bool)
        assigned_protos[closest_proto_indices] = True
        unassigned_protos = torch.where(~assigned_protos)[0]
        while unassigned_protos.numel() > 0:
            unassigned_similarity = similarity_matrix[unassigned_protos]
            _, proto_to_example = torch.max(unassigned_similarity, dim=1)
            for i, proto_idx in enumerate(unassigned_protos):
                example_idx = proto_to_example[i]
                if (closest_proto_indices==closest_proto_indices[example_idx]).sum() > 1:
                    closest_proto_indices[example_idx] = proto_idx
                else:
                    similarity_matrix[proto_idx][example_idx] = -2.
            
            assigned_protos = torch.zeros(protoNum, dtype=torch.bool)
            assigned_protos[closest_proto_indices] = True
            unassigned_protos = torch.where(~assigned_protos)[0]
        # 防止某个proto没有对应的embedding
        embedGroups = list()    # (normal, abnormal)
        lossComp = torch.zeros_like(self.protos).sum()
        for i in range(protoNum):
            mask = closest_proto_indices==i
            num_i = mask.sum()
            if num_i > 0:
                nEmbed_i = n_embeddings[mask]
                self.protos[i] = F.normalize((1 - beta) * self.protos[i].detach() + beta * nEmbed_i.mean(dim=0), p=2, dim=0)
                loss = 1 - torch.cosine_similarity(self.protos[i].unsqueeze(0).unsqueeze(1).detach(), nEmbed_i.unsqueeze(0), dim=2).sum() / num_i
                lossComp += loss
                aEmbed_i = self.getaEmbeddings(a_embeddings, num_i)
                embedGroups.append([nEmbed_i, aEmbed_i])
            else:
                embedGroups.append([])

        return embedGroups, lossComp / protoNum
    
    def getaEmbeddings(self, aEmbeddings, num):
        n = aEmbeddings.shape[0]
        if num <= n:
            random_indices = (torch.randint(0, n, (num,)))
        else:
            new_n = n
            while num > new_n:
                new_n *= 2
            random_indices = (torch.randint(0, new_n, (num,)) % n)
        
        return aEmbeddings[random_indices]
    
    def lossProto(self):
        protoNum = self.protos.shape[0]
        sim = torch.cosine_similarity(self.protos.unsqueeze(1).detach(), self.protos.unsqueeze(0).detach(), dim=2)
        mask1 = torch.eye(protoNum).cuda()
        mask2 = 1 - mask1
        loss = torch.abs((sim * mask2).sum() / (protoNum * protoNum - protoNum))
        return loss
    
    
    def assistScore(self, embedGroups):
        nGroup = len(embedGroups)
        
        loss = torch.tensor(0.).cuda()
        if nGroup == 3:
            score1 = self.assist1(torch.cat(embedGroups[0], dim=0)).squeeze()
            score2 = self.assist1(torch.cat(embedGroups[1], dim=0)).squeeze()
            score3 = self.assist1(torch.cat(embedGroups[1], dim=0)).squeeze()
            scores = [score1, score2, score3]

            for i in range(3):
                score = scores[i]
                n = score.shape[0]
                label = torch.zeros(n).cuda()
                label[n//2:] = 1
                loss += self.criterian(score, label)

            return scores, loss

        

    


