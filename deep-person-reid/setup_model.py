import torchreid
import pickle
import os
import torch

# Load datamanager from file
with open('datamanager.pkl', 'rb') as f:
    datamanager = pickle.load(f)

# Build model using datamanager
model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=False
)


# Build optimizer
optimizer = torchreid.optim.build_optimizer(
    model,
    optim="sgd",
    lr=0.065,
    weight_decay=0.0005
)


# Build learning rate scheduler
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="multi_step",
    stepsize=[150, 225, 300],
    gamma=0.1
)


# Build engine
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)


def convert_tar_pth_to_pt(tar_pth_file, pt_file):
    loaded_model = torch.load(tar_pth_file, map_location=torch.device('cpu'))
    model_state_dict = loaded_model['state_dict']
    torch.save(model_state_dict, pt_file)

# Main function to start training
if __name__ == '__main__':
    save_dir = "log/osnet_x0_25"
 
    # Run engine
    engine.run(
        save_dir=save_dir,
        max_epoch=350,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

    # After training completes, find the checkpoint path
    checkpoint_path = os.path.join(save_dir, 'model_best.pth.tar')  # Adjust as per your checkpoint naming convention
    print(f'Training completed. Checkpoint saved at: {checkpoint_path}')

    # Convert the checkpoint to .pt format
    pt_file_path = os.path.join(save_dir, 'model.pt')
    convert_tar_pth_to_pt(checkpoint_path, pt_file_path)
    print(f'Checkpoint converted to .pt format and saved at: {pt_file_path}')

