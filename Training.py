from src.data_loader import load_data
from src.model import build_attention_efficientunet, compile_for_regression

X_train, Y_train, X_test, Y_test = load_data(band="5GHz", split="High")

model = build_attention_efficientunet(
    input_shape=(256, 256, 3),
    backbone="EfficientNetB5",     # or "EfficientNetB2" for the smaller variant
    dropout_rate=0.5,
    encoder_trainable=True,        # False = fine-tuning mode
)
model = compile_for_regression(model, optimizer="sgd",
                                lr_start=1e-2, lr_end=1e-4, decay_steps=1500)

print(X_train.shape, Y_train.shape, model.output_shape)
