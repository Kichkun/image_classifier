import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models, transforms


class Model(object):
    def __init__(self, to_load=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=to_load)
        self.num_in_features = 25088
        self.hidden_layers = [256]
        self.num_out_features = 102
        for param in self.model.parameters():
            param.requires_grad = False

        classifier = self.build_nn_classifier()

        self.model.classifier = classifier

    def load_model(self, saved_model_path):
        self.model.load_state_dict(torch.load(saved_model_path))

    def build_nn_classifier(self, dropout=0.25):
        classifier = nn.Sequential()

        if self.hidden_layers == None:
            classifier.add_module('fc0', nn.Linear(self.num_in_features, 102))
        else:
            layer_sizes = zip(self.hidden_layers[:-1], self.hidden_layers[1:])
            classifier.add_module('fc0', nn.Linear(self.num_in_features, self.hidden_layers[0]))
            classifier.add_module('relu0', nn.ReLU())
            classifier.add_module('drop0', nn.Dropout(dropout))
            for i, (h1, h2) in enumerate(layer_sizes):
                classifier.add_module('fc' + str(i + 1), nn.Linear(h1, h2))
                classifier.add_module('relu' + str(i + 1), nn.ReLU())
                classifier.add_module('drop' + str(i + 1), nn.Dropout(dropout))
            classifier.add_module('output', nn.Linear(self.hidden_layers[-1], self.num_out_features))
        return classifier

    def process_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = preprocess(image)
        return image

    def imshow(self, image, ax=None, ):
        if ax is None:
            fig, ax = plt.subplots()

        image = image.numpy().transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax

    def predict(self, image_path, topk=2):
        img = Image.open(image_path)
        img = self.process_image(img)

        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img)

        self.model.eval()
        inputs = Variable(img).to(self.device)
        logits = self.model.forward(inputs)

        ps = F.softmax(logits, dim=1)
        topk = ps.cpu().topk(topk)

        return (e.data.numpy().squeeze().tolist() for e in topk)

    def train_model(self, dataloaders, dataset_sizes, criterion=nn.CrossEntropyLoss(), _lr=0.01, _momentum=0.9,
                    _step_size=7, _gamma=0.1, num_epochs=25, st=0,
                    save_model_folder=None, _log_dir=None):
        self.model.to(self.device)
        if _log_dir is not None:
            writer = SummaryWriter(_log_dir)
        optimizer = optim.SGD(self.model.parameters(), lr=_lr, momentum=_lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=_step_size, gamma=_gamma)
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(st, num_epochs + st):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for niter, (inputs, labels) in tqdm.tqdm_notebook(enumerate(dataloaders[phase])):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if _log_dir is not None:
                    if phase == 'train':
                        writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
                        writer.add_scalar('Train/EpochAcc', epoch_acc, epoch)
                    else:
                        writer.add_scalar('Valid/EpochLoss', epoch_loss, epoch)
                        writer.add_scalar('Valid/EpochAcc', epoch_acc, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if save_model_folder is not None:
                    s_pt = os.path.join(save_model_folder, '{}_{}.pt'.format('vgg_pr', str(epoch)))
                    if not os.path.exists(s_pt):
                        self.save_model(s_pt)

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def save_model(self, _path):
        if not os.path.exists(_path):
            torch.save(self.model.state_dict(), _path)

    def score_model(self, dataloaders, part='valid'):
        correct = 0
        total = 0
        predicted_all = []
        true_all = []
        with torch.no_grad():
            for inputs, labels in tqdm.notebook.tqdm(dataloaders[part]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_all.extend(predicted)
                true_all.extend(labels)
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        return predicted_all, true_all