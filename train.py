import torch
import itertools
import random
import pickle
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import PredefinedSplit,GridSearchCV,KFold

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from dataset import *
from models import *

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import argparse
parser = argparse.ArgumentParser(description = "Template")
parser.add_argument("-g","--gpu-index",default=0,type=int,help="the gpu index")
parser.add_argument("-p","--data-path",default="./data", type=str,help="the root of datasets")
parser.add_argument("-d","--dataset",default="Amyloid",type=str,help="between")
parser.add_argument("-hl","--hidden-layers", default=[64, 64, 64, 32, 32, 32],type=int, nargs="+", help="the sizes of hidden layers")
parser.add_argument("-f","--folds",default=5,type=int,help="the number of folds")
parser.add_argument("-t","--task",default="NC-EMCI",type=str,help="the binary task that we focuses")
parser.add_argument("-bs","--batch-size",default=32,type=int,help="the batch size")
parser.add_argument("-n","--n-epoch",default=20,type=int,help="the number of epochs")
parser.add_argument("-lr","--learning-rate",default=5e-4,type=float,help="the learning rate")
parser.add_argument("-ns","--n-show",default=1,type=int,help="for how many epochs to show the testing")
parser.add_argument("--save",default=False,action='store_true')
parser.add_argument("--regularize",default=False,action = "store_true")
parser.add_argument("--wrt-subjects",default=False,action='store_true')
parser.add_argument("--lambda-value",default=5e-4,type=float,help="the weight of the regularization term")


options = parser.parse_args()

device = torch.device(f'cuda:{options.gpu_index}')

if options.dataset == "Amyloid":
    dataset = Amyloid_dataset(options.data_path,normalized=True)
elif options.dataset == "FreeSurfer":
    dataset = FreeSurfer_dataset(options.data_path,normalized=True)
else:
    raise ValueError("Unknown dataset name! The name needs to be 'Amyloid' or 'FreeSurfer'! ")

splits = list(KFold(n_splits=options.folds,shuffle=True,random_state=SEED).split(dataset.RIDs))

classes = {'NC':0,'EMCI':1,'LMCI':2,'AD':3}
if options.task == "all":
    tasks = [
        "NC-EMCI","EMCI-LMCI","LMCI-AD","NC-AD"
    ]
else:
    tasks = [options.task]


def train_net(model, trainset, testset, wrt_subjects=False):
    if not options.regularize:

        if wrt_subjects:
            train_net_wrt_subjects(model, trainset, testset)
        else:
            train_net_wrt_samples(model, trainset, testset)
    else:
        train_net_with_regularization(model, trainset, testset)

def train_net_wrt_samples(model, trainset, testset):
    dataloader = DataLoader(trainset, batch_size=options.batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(options.n_epoch):
        total_loss = 0
        model.train()
        for idx, (input, label) in enumerate(dataloader):
            input = input.to(device)
            label = label.to(device).view(-1, 1).float()
            pred = model(input)[0]
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        if epoch % 100 == 0:
            validate_net(model, "trainset", epoch)
            validate_net(model, "testset", epoch)


def train_net_wrt_subjects(model, trainset, testset):
    if options.dataset == "Amyloid":
        n_splits = 20
    else:
        n_splits = 40
    batches = list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED
                         ).split(trainset.RIDs))
    batches = [batch[1] for batch in batches]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(options.n_epoch):
        total_loss = 0
        model.train()
        for idx in range(len(batches)):
            RIDs = trainset.RIDs[batches[idx]]
            indices = []
            for RID in RIDs:
                indices += trainset.RIDs_to_indices[RID]
            input = torch.FloatTensor(trainset.data[indices]).to(device)
            label = torch.FloatTensor(trainset.DXGrp[indices]).to(device).view(-1, 1).float()
            pred = model(input)[0]
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        if epoch % options.n_show == 0:
            validate_net_with_regularization(model, "trainset", epoch)
            validate_net_with_regularization(model, "testset", epoch)


def train_net_with_regularization(model, trainset, testset):
    if options.dataset == "Amyloid":
        n_splits = 20
    else:
        n_splits = 40
    batches = list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(trainset.RIDs))
    batches = [batch[1] for batch in batches]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=options.learning_rate)

    for epoch in range(options.n_epoch):
        model.train()
        for idx in range(len(batches)):
            RIDs = trainset.RIDs[batches[idx]]
            indices = []
            RIDs_to_indices_local = {}
            for RID in RIDs:
                indices_of_RID = trainset.RIDs_to_indices[RID]
                indices += indices_of_RID
                RIDs_to_indices_local[RID] = [indices.index(idx) for idx in indices_of_RID]

            input = torch.FloatTensor(trainset.data[indices]).to(device)
            label = torch.FloatTensor(trainset.DXGrp[indices]).to(device).view(-1, 1).float()
            pred, feature = model(input)
            loss = criterion(pred, label) + options.lambda_value * regularize(trainset, pred, feature, RIDs,
                                                                              RIDs_to_indices_local)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        if epoch % options.n_show == 0:
            validate_net_with_regularization(model, "trainset", epoch)
            validate_net_with_regularization(model, "testset", epoch)


def regularize(
        trainset,
        pred,
        feature,
        RIDs,
        RIDs_to_indices_local,
):
    loss_reg = 0
    for RID in RIDs:
        if len(RIDs_to_indices_local[RID]) < 2:
            continue
        indices_local = RIDs_to_indices_local[RID]
        c_n_2 = np.array(list(itertools.combinations(range(len(indices_local)), 2)))

        feature_subject = feature[indices_local]
        ages = torch.FloatTensor(trainset.ages[trainset.RIDs_to_indices[RID]]).to(device)

        feature_new = feature_subject[c_n_2[:, 1]]  # the embedding feature of the newer samples of the same subject
        feature_old = feature_subject[c_n_2[:, 0]]  # the embedding feature of the older samples of the same subject

        age_new = ages[c_n_2[:, 1]]  # the age of the subject at the newer samples
        age_old = ages[c_n_2[:, 0]]  # the age of the subject at the older samples

        target = (feature_old - feature_new) / ((feature_old - feature_new).norm(dim=1, keepdim=True) + 1e-7)
        target = target.mm(model.prediction.weight.T / (model.prediction.weight.norm() + 1e-7)).flatten() * (
                    age_new - age_old)

        loss_reg = loss_reg + target.sum()  # add to the regularization loss
    return loss_reg


def validate_net_with_regularization(model, set_choice, epoch):
    if set_choice == "trainset":
        valset = trainset
    elif set_choice == "testset":
        valset = testset
    else:
        raise ValueError("Unknown validation set choice!")
    criterion = nn.BCEWithLogitsLoss()
    input = torch.FloatTensor(valset.data).to(device)
    label = torch.FloatTensor(valset.DXGrp).to(device).view(-1, 1)
    with torch.no_grad():
        pred, feature = model(input)
        loss = criterion(pred, label).item()
        loss_reg = regularize(
            valset,
            pred,
            feature,
            valset.RIDs,
            valset.RIDs_to_indices
        ).item()
        auc = roc_auc_score(label.detach().cpu(), torch.sigmoid(pred).detach().cpu())

        violations, total = get_violations(pred, valset)
    print("Epoch: %d/%d Validation %s: Loss: %.4f,%.4f, AUC: %.4f, Vio: %d/%d=%.4f" % (
        epoch, options.n_epoch, set_choice, loss, options.lambda_value * loss_reg, auc, violations, total,
        violations / total))



def get_violations(pred, testset):
    violations = 0
    total = 0
    for RID in testset.RIDs:
        indices = testset.RIDs_to_indices[RID]
        if len(indices) > 1:
            pred_subject = pred[indices]
            c_n_2 = np.array(list(itertools.combinations(range(len(indices)), 2)))
            violations += (pred_subject[c_n_2[:,1]] <= pred_subject[c_n_2[:,0]]).sum().item()
            total += len(c_n_2)
    return violations, total


def test_net(model, testset):
    input = torch.FloatTensor(testset.data).to(device)
    label = torch.FloatTensor(testset.DXGrp)
    with torch.no_grad():
        pred, _ = model(input)
        pred = torch.sigmoid(pred)
        violations, total = get_violations(pred, testset)
    return pred.detach().cpu().numpy().flatten(), label.detach().cpu().numpy(), violations, total

AllResults = {
}
for task in tasks:
    print(f"================{task}================")
    negative_class = int(classes[task.split("-")[0]])
    positive_class = int(classes[task.split("-")[1]])

    Trials = []
    Violations = []
    for fold in range(options.folds):
        print(f"================fold-{fold}================")

        N = len(dataset.RIDs)
        training_RIDs = dataset.RIDs[splits[fold][0]]
        testing_RIDs = dataset.RIDs[splits[fold][1]]

        training_indices = []
        testing_indices = []
        for RID in training_RIDs:
            training_indices += dataset.RIDs_to_indices[RID]
        for RID in testing_RIDs:
            testing_indices += dataset.RIDs_to_indices[RID]

        # Select the subset for the corresponding binary task
        training_indices, testing_indices = np.array(training_indices), np.array(testing_indices)
        testing_indices = testing_indices[
            (dataset.DXGrp[testing_indices] == positive_class) + \
            (dataset.DXGrp[testing_indices] == negative_class)]
        training_indices = training_indices[
            (dataset.DXGrp[training_indices] == positive_class) + \
            (dataset.DXGrp[training_indices] == negative_class)]

        trainset = Subset(dataset, training_indices, binary=True)
        testset = Subset(dataset, testing_indices, binary=True)

        model = MLP(input_size=dataset.data.shape[1],
                    output_size=1, hidden_layers=list(options.hidden_layers)).to(device)
        print("Layers:", options.hidden_layers)
        train_net(model, trainset, testset, options.wrt_subjects)

        pred, label, violation, total = test_net(model, testset)

        Trials.append(np.array([pred, label]).T)
        Violations.append([violation, total])

    Trials = np.concatenate(Trials, axis=0)
    Violations = np.stack(Violations).sum(0)

    AllResults[task] = {
        'Trials': Trials,
        'Violations': Violations,
    }


if not os.path.exists("results"):
    os.mkdir("results")
if options.regularize:
    with open ("results/RMLP_%s.pickle"%options.dataset, "wb") as f:
        pickle.dump(AllResults, f)
else:
    with open ("results/MLP_%s.pickle"%options.dataset, "wb") as f:
        pickle.dump(AllResults, f)
