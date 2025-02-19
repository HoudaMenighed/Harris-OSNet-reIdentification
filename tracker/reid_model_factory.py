__model_types = [
    'resnet50', 'mlfn', 'hacnn', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4',
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
    'osnet_ibn_x1_0', 'osnet_ain_x1_0']

__trained_urls = {

    'osnet_x1_0_market1501.pt':
    'https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA',
    'osnet_x1_0_dukemtmcreid.pt':
    'https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq',
    'osnet_x1_0_msmt17.pt':
    'https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M',

}


def show_downloadeable_models():
    print('\nAvailable .pt ReID models for automatic download')
    print(list(__trained_urls.keys()))


def get_model_url(model):
    model = str(model).rsplit('/', 1)[-1]
    if model in __trained_urls:
        return __trained_urls[model]
    else:
        None


def is_model_in_model_types(model):
    model = str(model).rsplit('/', 1)[-1].split('.')[0]
    if model in __model_types:
        return True
    else:
        return False


def get_model_name(model):
    model = str(model).rsplit('/', 1)[-1].split('.')[0]
    for x in __model_types:
        if x in model:
            return x
    return None

