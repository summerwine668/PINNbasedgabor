import torch

# Positional encoding
class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(enc_config):
	
    if enc_config.encoding_type == -1:
        return torch.nn.Identity(), enc_config.embed_dim

    elif enc_config.encoding_type==0:
        embed_kwargs = {
            'include_input': True,
            'input_dims': enc_config.embed_dim,
            'max_freq_log2': enc_config.multires-1,
            'num_freqs': enc_config.multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)
        def embed(x, eo=embedder_obj): return eo.embed(x)
        out_dim = embedder_obj.out_dim
    return embed, out_dim

