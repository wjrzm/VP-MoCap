import importlib as imp

def make_dataset(opt, phase):
    dataset = imp.machinery.SourceFileLoader(
        opt.module, opt.path).load_module().ContDataset
    dataset = dataset(opt, phase)
    return dataset