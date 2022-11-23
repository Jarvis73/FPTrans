from networks.FPTrans import FPTrans


__networks = {
    'fptrans': FPTrans,
}


def load_model(opt, logger, *args, **kwargs):
    if opt.network.lower() in __networks:
        model = __networks[opt.network.lower()](opt, logger, *args, **kwargs)
        if opt.print_model:
            print(model)
        return model
    else:
        raise ValueError(f'Not supported network: {opt.network}. {list(__networks.keys())}')
