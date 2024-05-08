import importlib as imp

def make_network(opt):
    network = imp.machinery.SourceFileLoader(
        opt.module, opt.path).load_module().ContNetwork
    _network = network(opt)
    return _network