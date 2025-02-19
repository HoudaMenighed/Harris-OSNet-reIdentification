import torch

def convert_tar_pth_to_pt(tar_pth_file, pt_file):
    loaded_model = torch.load(tar_pth_file, map_location=torch.device('cpu'))

    model_state_dict = loaded_model['state_dict']

    torch.save(model_state_dict, pt_file)

convert_tar_pth_to_pt('model3.pth.tar-60', 'osnet_x0_25_avecHarris.pt')