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
        x = F.adaptive_avg_pool2d(x, (1, 1))    # 全局平均池化
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
    def __init__(self, in_dim):
        super(subNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(14 * 14, 128)
        self.fc2 = nn.Linear(7 * 7, 128) 
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
        # self.initProtos()
        self.exchangeNet = subNet(self.in_c)
        self.protos = torch.randn(size=(3, 128), requires_grad=True).cuda()

    def initProtos(self, protoNum = 3):
        sizes = [(protoNum, self.in_c, self.cfg.img_size // (2 ** s) // 32, self.cfg.img_size // (2 ** s) // 32) for s in range(self.cfg.n_scales)]
        self.protos = [torch.randn(size=sz).cuda() for sz in sizes]

    def forward(self, image, label):
        image_pyramid = list()
        for i in range(self.cfg.total_heads):
            image_pyramid.append(list())
        
        # tmpLossList = list()        
        for s in range(self.cfg.n_scales):
            image_scaled = F.interpolate(image, size=self.cfg.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)  # torch.Size([53, 512, 14, 14]) torch.Size([53, 512, 7, 7])

            n_features = feature[label==0]
            a_features = feature[label==1]

            if self.training:
                self.updateProtos(n_features)
                p_features = self.genPseudo(n_features, a_features)

                normal_scores = self.holistic_head(feature)                             # torch.Size([48, 1])
                abnormal_scores = self.seen_head(feature)                               # torch.Size([48, 1])
                dummy_scores = self.pseudo_head(torch.concat((n_features, p_features))) # torch.Size([48, 1])
            else:
                normal_scores = self.holistic_head(feature)
                abnormal_scores = self.seen_head(feature)
                dummy_scores = self.pseudo_head(feature)

            for i, scores in enumerate([normal_scores, abnormal_scores, dummy_scores]):
                image_pyramid[i].append(scores)
        for i in range(self.cfg.total_heads):
            image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
            image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)
        return image_pyramid
    

    def updateProtos(self, n_features):
        # updateProtos
        beta = 0.1
        protoNum = self.protos.shape[0]
        n_embeddings = self.exchangeNet(n_features)
        similarity_matrix = torch.cosine_similarity(self.protos.unsqueeze(1), n_embeddings.unsqueeze(0), dim=2)
        closest_proto_indices = torch.argmax(similarity_matrix, dim=0)

        nEmbedGroups = list()
        for i in range(protoNum):
            mask = closest_proto_indices==i
            nEmbedGroups.append(n_embeddings[mask])
            self.protos[i] = (1 - beta) * self.protos[i].data + beta * n_embeddings[mask].mean(dim=0)
        
   
    def genPseudo(self, n_features, a_features):
        gama = 0.5
        b, c, h, w = a_features.shape
        torch.manual_seed(0)
        random_indices = torch.randperm(n_features.shape[0])
        n_selected = n_features[random_indices[:b]]
        p_features = gama * a_features + (1 - gama) * n_selected
        return p_features



