import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
from sklearn import metrics
import config
import utils
from tqdm import tqdm


def loss_fn(outputs, targets):
    o1, o2 = outputs
    t1, t2 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    return (l1 + l2) / 2


def focal_loss(outputs, targets):
    o1, o2 = outputs
    t1, t2 = targets
    l1 = utils.FocalLoss(gamma=0)(o1, t1)
    l2 = utils.FocalLoss(gamma=0)(o2, t2)
    return (l1 + l2) / 2


def train(data_loader, model, optimizer, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, data in enumerate(tk0):
        image = data["image"]
        target_1 = data["Target 1"]
        target_2 = data["Target 2"]

        image = image.to(config.DEVICE, dtype=torch.float)
        target_1 = target_1.to(config.DEVICE, dtype=torch.long)
        target_2 = target_2.to(config.DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (target_1, target_2)

        loss = focal_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        losses.update(loss.item(), image.size(0))
        tk0.set_postfix(loss=losses.avg)


def evaluate(data_loader, model):
    model.eval()
    final_loss = 0
    final_targets = []
    final_outputs = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk0:
            image = data["image"]
            target_1 = data["Target 1"]
            target_2 = data["Target 2"]

            image = image.to(config.DEVICE, dtype=torch.float)
            target_1 = target_1.to(config.DEVICE, dtype=torch.long)
            target_2 = target_2.to(config.DEVICE, dtype=torch.long)

            outputs = model(image)
            targets = (target_1, target_2)

            o1, o2 = outputs
            t1, t2 = targets
            o1 = torch.sigmoid(o1)
            o2 = torch.sigmoid(o2)

            final_outputs.append(torch.cat((o1, o2), dim=1))
            final_targets.append(torch.stack((t1, t2), dim=1))  # stack
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        return final_outputs, final_targets
