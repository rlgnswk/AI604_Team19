import torch
from collections import OrderedDict

def normalize_activation(x):
	norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
	return x / (norm_factor + 1e-10)

def get_state_dict(net_type='alex', version='0.1'):
	url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
			+ f'master/lpips/weights/v{version}/{net_type}.pth'

	map_location = torch.device('cpu')
	if torch.cuda.is_available():
		map_location = None

	old_state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location=map_location)

	new_state_dict = OrderedDict()
	for key, val in old_state_dict.items():
		new_key = key
		new_key = new_key.replace('lin', '')
		new_key = new_key.replace('model.', '')
		new_state_dict[new_key] = val

	return new_state_dict