from sklearn.metrics import classification_report

def get_labels_predictions(model, data_loader, device):
    model.eval()
    y = []
    y_ = []
    with torch.no_grad():
        for x, labels, lengths in data_loader:
            loss, logits = model(x.float().to(device), labels.to(device), lengths.to(device))
            y.append(labels)
            y_.append(logits.argmax(dim=-1).detach().cpu().numpy())

    return y, y_

data_dir = 'fma_genre_spectrograms'
saved_model_path = 'lstm_genre_mel.pth'

print(f'Evaluation of model {saved_model_path.replace(".pth", "")} on the test set...')

test_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), class_mapping=CLASS_MAPPING,
    train=False, feat_type='mel', max_length=MAX_LENGTH
)

test_loader, _ = torch_train_val_split(
    test_dataset, BATCH_SIZE, BATCH_SIZE, val_size=0.0
)

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(test_loader))
input_shape = x_b1[0].shape

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)
model.load_state_dict(torch.load(saved_model_path))

y, y_ = get_labels_predictions(model, test_loader, DEVICE)
print(classification_report(np.hstack(y), np.hstack(y_)))
