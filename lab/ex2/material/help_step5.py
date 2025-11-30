import torch
import torch.nn as nn

def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device):
    print('Training in overfitting mode...')
    epochs = 400

    # get only the 1st batch
    x_b1, y_b1, lengths_b1 = next(iter(train_loader))
    model.train()
    for epoch in range(epochs):
        loss, logits = model(x_b1.float().to(device), y_b1.to(device), lengths_b1.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()

        if epoch == 0 or (epoch+1)%20 == 0:
            print(f'Epoch {epoch+1}, Loss at training set: {loss.item()}')


def train(model, train_loader, val_loader, optimizer, epochs, device="cuda", overfit_batch=False):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device)
    else:
        pass
#         early_stopper = EarlyStopper(patience=5)
#         for epoch in range(epochs):
#             train_loss = train_one_epoch(model, train_loader)
#             validation_loss = validate_one_epoch(model, val_loader)
#             if early_stopper.early_stop(validation_loss):
#                 break


class Classifier(nn.Module):
    def __init__(self, num_classes, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Classifier, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits

# run Training in overfitting mode
from lstm import LSTMBackbone

DEVICE = 'cuda'
RNN_HIDDEN_SIZE = 64
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 10

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, device=DEVICE, overfit_batch=True)

# we need EarlyStopping that we'll adopt from: https://stackoverflow.com/a/73704579/19306080 and add some customizations

class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# implemetation of the training process

def train_one_epoch(model, train_loader, optimizer, device=DEVICE):
    model.train()
    total_loss = 0
    for x, y, lengths in train_loader:
        loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_one_epoch(model, val_loader, device=DEVICE):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train(model, train_loader, val_loader, optimizer, epochs, save_path='checkpoint.pth', device="cuda", overfit_batch=False):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device)
    else:
        print(f'Training started for model {save_path.replace(".pth", "")}...')
        early_stopper = EarlyStopper(model, save_path, patience=5)
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            validation_loss = validate_one_epoch(model, val_loader)
            if epoch== 0 or (epoch+1)%5==0:
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')

            if early_stopper.early_stop(validation_loss):
                print('Early Stopping was activated.')
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')
                print('Training has been completed.\n')
                break

# run Training
DEVICE = 'cuda'
RNN_HIDDEN_SIZE = 128
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 40
save_path='lstm_genre_mel.pth'

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)
