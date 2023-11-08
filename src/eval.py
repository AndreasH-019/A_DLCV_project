from model import get_model, LitMaskRCNN
import lightning as L

# model = LitMaskRCNN.load_from_checkpoint(checkpoint_path="D:/ADLCV/projekt/A_DLCV_project/src/lightning_logs/version_3/checkpoints/epoch=0-step=1.ckpt")
model = get_model()
trainer = L.Trainer(log_every_n_steps=1)
result = trainer.test(model=model)
print(result)


