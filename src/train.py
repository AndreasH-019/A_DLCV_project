import lightning as L
from model import get_model

pl_model = get_model()
trainer = L.Trainer(max_epochs=4, check_val_every_n_epoch=1, log_every_n_steps=1)
trainer.fit(model=pl_model)