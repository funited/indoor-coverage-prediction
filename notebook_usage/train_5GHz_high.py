from src.data_loader import load_data
from src.model       import build_attention_efficientunet, compile_for_regression
from src.training    import train_leg, predict_coarse, build_second_leg_input
from src.evaluation  import (overall_rmse_mae, plot_prediction_vs_truth,
                              plot_error_cdf, plot_training_curves)

# ---------- data ----------
X_train, Y_train, X_test, Y_test = load_data(band="5GHz", split="High", seed=42)

# ---------- LEG 1 ----------
leg1 = build_attention_efficientunet(input_shape=(256, 256, 3),
                                     backbone="EfficientNetB5",
                                     dropout_rate=0.5)
leg1 = compile_for_regression(leg1, optimizer="sgd",
                               lr_start=1e-2, lr_end=1e-4, decay_steps=1500)

hist1 = train_leg(leg1, X_train, Y_train,
                  epochs=30, batch_size=32, validation_split=0.3,
                  checkpoint_path="leg1.keras")
plot_training_curves(hist1, threshold=1800)

# Coarse predictions for both train (for leg-2 input) and test
coarse_train = predict_coarse(leg1, X_train, save_dir="artifacts/Coarse_Pred")
coarse_test  = predict_coarse(leg1, X_test,  save_dir="artifacts/SR_Pred",
                               name_prefix="SR_Pred")

# Sanity-check leg-1 on the test set
print("Leg-1 test:", overall_rmse_mae(leg1.predict(X_test), Y_test))

# ---------- LEG 2 ----------
X_train_2 = build_second_leg_input(X_train, coarse_train)
X_test_2  = build_second_leg_input(X_test,  coarse_test)

leg2 = build_attention_efficientunet(input_shape=(256, 256, 3),
                                     backbone="EfficientNetB5",
                                     dropout_rate=0.5)
leg2 = compile_for_regression(leg2, optimizer="sgd",
                               lr_start=1e-4, lr_end=1e-6, decay_steps=1500)

hist2 = train_leg(leg2, X_train_2, Y_train,
                  epochs=30, batch_size=32, validation_split=0.3,
                  checkpoint_path="leg2.keras")
plot_training_curves(hist2, threshold=1800)

# ---------- final evaluation ----------
preds_test = leg2.predict(X_test_2)
print("Leg-2 test:", overall_rmse_mae(preds_test, Y_test))

i = 2
plot_prediction_vs_truth(preds_test[i], Y_test[i])
plot_error_cdf(preds_test[i], Y_test[i])
