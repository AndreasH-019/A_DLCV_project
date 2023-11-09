import argparse
from model import get_model, LitMaskRCNN
import lightning as L

def parserargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None, help='Sti til checkpoint-filen (default: None)')
    parser.add_argument('--debug', action='store_true', help='Aktiver debugging-tilstand')
    args = parser.parse_args()
    return args

args = parserargs()
checkpoint_path = args.checkpoint_path
model = LitMaskRCNN.load_from_checkpoint(checkpoint_path=checkpoint_path) if checkpoint_path is not None else get_model()
model.set_debug(args.debug)
trainer = L.Trainer(log_every_n_steps=1)
result = trainer.test(model=model)