import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM
from modeling.layers import build_criterion


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
    def __init__(self, input_size=128, hidden_size=256, output_size=1):
        super(AssistNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(F.normalize(x)))
        x = self.fc2(x)
        return x


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

        self.numProtos = cfg.numProtos
        self.protos = nn.Parameter(torch.randn(size=(self.numProtos, 128)), requires_grad=True).cuda()
        if self.numProtos == 2:
            self.assist1 = PlainHead(self.in_c, self.cfg.topk)
            self.assist2 = PlainHead(self.in_c, self.cfg.topk)
        elif self.numProtos == 3:
            self.assist1 = PlainHead(self.in_c, self.cfg.topk)
            self.assist2 = PlainHead(self.in_c, self.cfg.topk)
            self.assist3 = PlainHead(self.in_c, self.cfg.topk)
        elif self.numProtos == 4:
            self.assist1 = PlainHead(self.in_c, self.cfg.topk)
            self.assist2 = PlainHead(self.in_c, self.cfg.topk)
            self.assist3 = PlainHead(self.in_c, self.cfg.topk)
            self.assist4 = PlainHead(self.in_c, self.cfg.topk)
        elif self.numProtos == 5:
            self.assist1 = PlainHead(self.in_c, self.cfg.topk)
            self.assist2 = PlainHead(self.in_c, self.cfg.topk)
            self.assist3 = PlainHead(self.in_c, self.cfg.topk)
            self.assist4 = PlainHead(self.in_c, self.cfg.topk)
            self.assist5 = PlainHead(self.in_c, self.cfg.topk)

        self.beta = cfg.beta
        
        self.criterionType = 'deviation' # CE deviation
        self.criterian = build_criterion(self.criterionType)

        self.cdfl = cfg.cdfl    # ablation of cdfl module
        # self.sf = cfg.sf
        # self.dir = cfg.experiment_dir
        self.epochs = cfg.epochs


    def forward(self, image, label, epoch=0):
        image_pyramid = list()
        for i in range(self.cfg.total_heads):
            image_pyramid.append(list())

        embedsList = list()
        featsList = list()
        for s in range(self.cfg.n_scales):
            image_scaled = F.interpolate(image, size=self.cfg.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)  # torch.Size([53, 512, 14, 14]) torch.Size([53, 512, 7, 7])

            embeds = self.exchangeNet(feature)
            embedsList.append(embeds)
            featsList.append(feature)

            # save features
            if epoch == self.epochs:
                return embeds

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
            embeds, feats0, feats1 = embedsList[0], featsList[0], featsList[1]
            embeddings = F.normalize(embeds)
            embedLabel = self.getEmbedLabel(label)
            embedGroups0, embedGroups1, lossComp = self.updateProtos(feats0, feats1, embeddings, embedLabel)
            _, assistloss0 = self.assistScore(embedGroups0)
            _, assistloss1 = self.assistScore(embedGroups1)
            assistloss = (assistloss0 + assistloss1) / 2

            if self.cdfl:
                assistloss = assistloss * 0.

            lossProto = self.lossProto()

            alpha = 0.1
            tmpLoss = alpha * (lossComp + lossProto) + assistloss            
            return image_pyramid, tmpLoss
        else:
            testfeatsList = [x[self.cfg.nRef:] for x in featsList]
            assistScores = list()
            for feat in testfeatsList:
                score = self.assistScore(feat)
                assistScores.append(score)

            assistScore = torch.stack(assistScores).mean(dim=0)

            if self.cdfl:
                assistScore = assistScore * 0.
            
            return image_pyramid, assistScore   #  assistScore: 来自辅助网络的异常分数预测
    
    def getEmbedLabel(self, label):
        label1 = torch.cat((torch.zeros(self.cfg.nRef, dtype=label.dtype).cuda(), label))
        # label1_combined = torch.cat((label1, label1))
        # return label1_combined
        return label1


    def updateProtos(self, feats0, feats1, embeddings, embedLabel):
        # updateProtos
        n_embeddings = embeddings[embedLabel==0]
        n_feats0 = feats0[embedLabel==0]
        n_feats1 = feats1[embedLabel==0]
        a_embeddings = embeddings[embedLabel!=0]
        a_feats0 = feats0[embedLabel!=0]
        a_feats1 = feats1[embedLabel!=0]
        # beta = 0.01
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
        featsGroups0 = list()    # (normal, abnormal)
        featsGroups1 = list()    # (normal, abnormal)
        lossComp = torch.zeros_like(self.protos).sum()
        for i in range(protoNum):
            mask = closest_proto_indices==i
            num_i = mask.sum()
            if num_i > 0:
                nEmbed_i = n_embeddings[mask]
                nFeats_i0 = n_feats0[mask]
                nFeats_i1 = n_feats1[mask]
                self.protos[i] = F.normalize((1 - self.beta) * self.protos[i].detach() + self.beta * nEmbed_i.mean(dim=0), p=2, dim=0)
                loss = 1 - torch.cosine_similarity(self.protos[i].unsqueeze(0).unsqueeze(1).detach(), nEmbed_i.unsqueeze(0), dim=2).mean()
                lossComp += loss
                # aEmbed_i = self.getaEmbeddings(a_embeddings, num_i)
                aFeats_i0 = self.getaFeats(a_feats0, num_i)
                aFeats_i1 = self.getaFeats(a_feats1, num_i)
                # embedGroups.append([nEmbed_i, aEmbed_i])
                featsGroups0.append([nFeats_i0, aFeats_i0])
                featsGroups1.append([nFeats_i1, aFeats_i1])
            else:
                embedGroups.append([])

        # return embedGroups, lossComp / protoNum
        return featsGroups0, featsGroups1, lossComp / protoNum
    
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

    def getaFeats(self, aFeats, num):
        n = aFeats.shape[0]
        if num <= n:
            random_indices = (torch.randint(0, n, (num,)))
        else:
            new_n = n
            while num > new_n:
                new_n *= 2
            random_indices = (torch.randint(0, new_n, (num,)) % n)
        
        return aFeats[random_indices]
    
    
    def lossProto(self):
        protoNum = self.protos.shape[0]
        sim = torch.cosine_similarity(self.protos.unsqueeze(1).detach(), self.protos.unsqueeze(0).detach(), dim=2)
        mask1 = torch.eye(protoNum).cuda()
        mask2 = 1 - mask1
        loss = torch.abs((sim * mask2).sum() / (protoNum * protoNum - protoNum))
        return loss
    
    
    def assistScore(self, embedGroups):
        nGroup = self.protos.shape[0]
        scores = list()
        if self.training:
            loss = torch.tensor(0.).cuda()
            if nGroup == 2:
                score1 = self.assist1(torch.cat(embedGroups[0], dim=0)).squeeze()
                score2 = self.assist2(torch.cat(embedGroups[1], dim=0)).squeeze()
                scores.append(score1)
                scores.append(score2)
            elif nGroup == 3:
                score1 = self.assist1(torch.cat(embedGroups[0], dim=0)).squeeze()
                score2 = self.assist2(torch.cat(embedGroups[1], dim=0)).squeeze()
                score3 = self.assist3(torch.cat(embedGroups[2], dim=0)).squeeze()
                scores.append(score1)
                scores.append(score2)
                scores.append(score3)
            elif nGroup == 4:
                score1 = self.assist1(torch.cat(embedGroups[0], dim=0)).squeeze()
                score2 = self.assist2(torch.cat(embedGroups[1], dim=0)).squeeze()
                score3 = self.assist3(torch.cat(embedGroups[2], dim=0)).squeeze()
                score4 = self.assist4(torch.cat(embedGroups[2], dim=0)).squeeze()
                scores.append(score1)
                scores.append(score2)
                scores.append(score3)
                scores.append(score4)
            elif nGroup == 5:
                score1 = self.assist1(torch.cat(embedGroups[0], dim=0)).squeeze()
                score2 = self.assist2(torch.cat(embedGroups[1], dim=0)).squeeze()
                score3 = self.assist3(torch.cat(embedGroups[2], dim=0)).squeeze()
                score4 = self.assist4(torch.cat(embedGroups[2], dim=0)).squeeze()
                score5 = self.assist5(torch.cat(embedGroups[2], dim=0)).squeeze()
                scores.append(score1)
                scores.append(score2)
                scores.append(score3)
                scores.append(score4)
                scores.append(score5)

            for i in range(nGroup):
                score = scores[i]
                n = score.shape[0]
                label = torch.zeros(n).cuda()
                label[n//2:] = 1
                # loss += self.criterian(score, label)
                if self.criterionType == 'CE':
                    prob = F.softmax(score)
                    loss += self.criterian(prob, label.float())
                else:
                    loss += self.criterian(score, label.float())
            return scores, loss / nGroup
        else :
            if nGroup == 2:
                score1 = self.assist1(embedGroups).squeeze()
                score2 = self.assist2(embedGroups).squeeze()
                scores.append(score1)
                scores.append(score2)
            elif nGroup == 3:
                score1 = self.assist1(embedGroups).squeeze()
                score2 = self.assist2(embedGroups).squeeze()
                score3 = self.assist3(embedGroups).squeeze()
                scores.append(score1)
                scores.append(score2)
                scores.append(score3)
            elif nGroup == 4:
                score1 = self.assist1(embedGroups).squeeze()
                score2 = self.assist2(embedGroups).squeeze()
                score3 = self.assist3(embedGroups).squeeze()
                score4 = self.assist4(embedGroups).squeeze()
                scores.append(score1)
                scores.append(score2)
                scores.append(score3)
                scores.append(score4)
            elif nGroup == 5:
                score1 = self.assist1(embedGroups).squeeze()
                score2 = self.assist2(embedGroups).squeeze()
                score3 = self.assist3(embedGroups).squeeze()
                score4 = self.assist4(embedGroups).squeeze()
                score5 = self.assist5(embedGroups).squeeze()
                scores.append(score1)
                scores.append(score2)
                scores.append(score3)
                scores.append(score4)
                scores.append(score5)
                
            # score = torch.stack(scores).max(dim=0)[0]
            score = torch.stack(scores).mean(dim=0)
            return score
                
        

    


