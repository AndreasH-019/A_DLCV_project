import lightning as L
from model import LitMaskRCNN
import argparse

def parserargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Aktiver debugging-tilstand (sandhedsværdi)')
    args = parser.parse_args()
    return args

def getTrainer(debug):
    if debug:
        return L.Trainer(max_epochs=4, check_val_every_n_epoch=1, log_every_n_steps=1)
    else:
        return L.Trainer(max_epochs=100)

args = parserargs()
pl_model = LitMaskRCNN()
pl_model.set_debug(args.debug)
trainer = getTrainer(args.debug)
trainer.fit(model=pl_model)