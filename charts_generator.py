from utils.plotting_util import plot_line_chart
from utils.trainutil import load_json


def generate_plot_for_resnet_with_upproj_best_model():
    lr_vals = load_json('plots_data/learning_rate_rn50up-v4_default.json')
    train_vals = load_json('plots_data/train_rn50up-v4_berHu_default.json')
    val_vals = load_json('plots_data/val_rn50up-v4_berHu_default.json')
    plot_line_chart([i for i in range(len(train_vals))], [('TrainLoss-ResnetWithUpProj', train_vals), ('ValLoss-ResnetWithUpProj', val_vals)], 'Train vs Validation Loss for Resnet With UpProjection Model', 'plots/train_vs_val_rn50upproj')


def generate_plot_for_resnet_with_upconv_best_model():
    lr_vals = load_json('plots_data/learning_rate_rn50-v5_default.json')
    train_vals = load_json('plots_data/train_rn50-v5_berHu_default.json')
    val_vals = load_json('plots_data/val_rn50-v5_berHu_default.json')
    plot_line_chart([i for i in range(len(train_vals))], [('TrainLoss-ResnetWithUpConv', train_vals), ('ValLoss-ResnetWithUpConv', val_vals)], 'Train vs Validation Loss for Resnet With UpConvolution Model', 'plots/train_vs_val_rn50upconv')


if __name__ == '__main__':
    generate_plot_for_resnet_with_upproj_best_model()
    generate_plot_for_resnet_with_upconv_best_model()
