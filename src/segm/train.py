import lightning as L
from model import LitMaskRCNN
import argparse

def parserargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Aktiver debugging-tilstand')
    parser.add_argument('--paste_mode', help='one of "gen", "real", "none"')
    args = parser.parse_args()
    return args

def getTrainer(debug):
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='mAP', mode='max',
                                                              save_top_k=1, filename='best', save_last=True)
    if debug:
        return L.Trainer(max_epochs=1, check_val_every_n_epoch=1, log_every_n_steps=1, callbacks=[checkpoint_callback])
    else:
        return L.Trainer(max_epochs=100, callbacks=[checkpoint_callback])

args = parserargs()
pl_model = LitMaskRCNN(args.paste_mode, args.debug)
# pl_model.set_debug(args.debug)
trainer = getTrainer(args.debug)
trainer.fit(model=pl_model)