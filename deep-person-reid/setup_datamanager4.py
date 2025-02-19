import torchreid
import pickle

# Initialize datamanager
datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="market1501",
    targets="market1501",
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=64,
    transforms=["random_flip", "random_erase"]
)

# Save datamanager to a file using pickle
with open('datamanager.pkl', 'wb') as f:
    pickle.dump(datamanager, f)

