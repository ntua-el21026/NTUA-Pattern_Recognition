# Step 4: Data loading and analysis
from dataset import CLASS_MAPPING, SpectrogramDataset

PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, train=True
)

print(train_dataset[10])
print(f"Input: {train_dataset[10][0].shape}")
print(f"Label: {train_dataset[10][1]}")
print(f"Original length: {train_dataset[10][2]}")

def plot_class_histogram(dataset, title):
    plt.figure()
    plt.hist(dataset.labels, rwidth=0.5, align='mid')
    plt.xticks(dataset.labels, dataset.labels.astype(str))
    plt.title(title)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequencies')

plot_class_histogram(train_dataset, title='Histogram of classes with merged classes')

train_dataset_all = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=None, train=True
)
plot_class_histogram(train_dataset_all, title='Histogram of classes with all classes')

# dataloaders for specific features type, e.g. 'mel'
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING,
    train=True, feat_type='mel', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)
