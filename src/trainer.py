import torch
from collections import OrderedDict
import src.utils as utils


class Trainer:
    def __init__(self, config, tb_writer=None):
        self.config = config
        self.tb_writer = tb_writer
        self.optimizer = None
        self.criterion = None
        self.model = None

    def build_model(self, model_name="fcn", pretrain=None, num_classes=2):
        """ Get model architecture """
        raise NotImplementedError("model {} not implemented".format(model_name))

    def train_model(self, dataloader, epoch):
        """ Train a single epoch.

        Args:
            dataloader (torch Dataloader): initialized dataloader containing training data
            epoch (int): current epoch
        """
        self.model.train()

        running_correct = 0
        loss_idx_value = (epoch * len(dataloader))
        for i, data in enumerate(dataloader, 0):
            _, inputs, labels = data
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            if type(outputs) == OrderedDict:
                outputs = outputs["out"]

            loss = self.criterion(outputs, labels.squeeze().long())
            predict = torch.argmax(outputs, dim=1)
            correct = (predict == labels.squeeze()).sum() / torch.numel(labels)
            running_correct += correct

            loss.backward()
            self.optimizer.step()

            # print statistics
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.6f}, acc: {correct:.6f}')

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("Loss/Minibatches", loss, loss_idx_value)
                loss_idx_value += 1

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/Epochs", loss, epoch)
            self.tb_writer.add_scalar("Accuracy/Epochs", running_correct / len(dataloader), epoch)

    def validate_model(self, dataloader, epoch):
        """ Evaluate dice on validation set.

        Args:
            dataloader (torch Dataloader): initialized dataloader containing validation data
            epoch (int): current epoch
        """
        self.model.eval()
        correct = 0.0
        for i, data in enumerate(dataloader, 0):
            _, inputs, labels = data
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)

            outputs = self.model(inputs)
            if type(outputs) == OrderedDict:
                outputs = outputs["out"]

            predict = torch.argmax(outputs, dim=1)
            correct += (predict == labels.squeeze()).sum() / torch.numel(labels)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Accuracy/Validation", correct / len(dataloader), epoch)

    def run(self, train_dataloader, val_dataloader):
        """ Main function for training and validating model.

        Args:
            train_dataloader (torch DataLoader): iterable function for training data
            val_dataloader (torch DataLoader): iterable function for validation data
        """
        if self.model is None:
            self.model = self.build_model(model_name=self.config.model_name, pretrain=self.config.model_weights)
            self.model.to(self.config.device)

        if self.optimizer is None and "optimizer_name" in self.config:
            self.optimizer = utils.get_optimizer(self.model, self.config.optimizer_name)

        if self.criterion is None:
            self.criterion = utils.get_criterion()

        print("Training...")
        for _e in range(int(self.config.num_epochs)):
            self.train_model(train_dataloader, _e)
            self.validate_model(val_dataloader, _e)
