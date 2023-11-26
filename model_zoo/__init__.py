from model_zoo.pinnmodel import PINNmodel
from model_zoo.gabornetl import GaborNetL

def build_model(model_type, layer_type, in_channels, out_channels, hidden_layers, **kwargs):
	if model_type == 'pinn':
		model = PINNmodel(layer_type, in_channels, out_channels, hidden_layers, **kwargs)
	elif model_type == 'gaborl':
		model = GaborNetL(in_channels, out_channels, hidden_layers, **kwargs)
	else:
		raise NotImplementedError
	return model