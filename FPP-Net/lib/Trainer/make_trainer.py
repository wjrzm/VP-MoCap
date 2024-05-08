import importlib as imp

def make_trainer(dataloader, train_net, optimizer, recorder, gpu_ids,opt):
    trainer = imp.machinery.SourceFileLoader(
        opt.module, opt.path).load_module().ContTrainer
    trainer = trainer(dataloader, train_net, optimizer, recorder, gpu_ids,opt)
    return trainer