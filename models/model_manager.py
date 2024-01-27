from utils.register import register
import torch
from .gcc import change_params_key


def load_model(input_dim: int, output_dim: int, config):
    if config.model in ['GCC', 'GCC_GraphControl']:
        state_dict = torch.load('checkpoint/gcc.pth', map_location='cpu')
        opt = state_dict['opt']
        model = register.models[config.model](
            positional_embedding_size=opt.positional_embedding_size,
            max_node_freq=opt.max_node_freq,
            max_edge_freq=opt.max_edge_freq,
            max_degree=opt.max_degree,
            freq_embedding_size=opt.freq_embedding_size,
            degree_embedding_size=opt.degree_embedding_size,
            output_dim=opt.hidden_size,
            node_hidden_dim=opt.hidden_size,
            edge_hidden_dim=opt.hidden_size,
            num_layers=opt.num_layer,
            num_step_set2set=opt.set2set_iter,
            num_layer_set2set=opt.set2set_lstm_layer,
            gnn_model=opt.model,
            norm=opt.norm,
            degree_input=True,
            num_classes = output_dim
        )
        params = state_dict['model']
        change_params_key(params)

        if config.model == 'GCC':
            model.load_state_dict(params)
            return model
        elif config.model == 'GCC_GraphControl':
            model.encoder.load_state_dict(params)
            model.trainable_copy.load_state_dict(params)
            return model
    else:
        return register.models[config.model](input_dim=input_dim, output_dim=output_dim, **vars(config))