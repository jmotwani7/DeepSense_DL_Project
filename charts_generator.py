from utils.plotting_util import plot_line_chart
from utils.trainutil import load_json

# ('solid', 'solid'),  # Same as (0, ()) or '-'
# ('dotted', 'dotted'),  # Same as (0, (1, 1)) or ':'
# ('dashed', 'dashed'),  # Same as '--'
# ('dashdot', 'dashdot')
SOLID = 'solid'
DOTTED = 'dotted'
DASHED = 'dashed'
DASHDOT = 'dashdot'


def generate_plot_for_resnet_with_upproj_best_model():
    lr_vals = load_json('plots_data/learning_rate_rn50up-v4_default.json')
    train_vals_conv = load_json('plots_data/train_rn50-v5_berHu_default.json')
    val_vals_cpmv = load_json('plots_data/val_rn50-v5_berHu_default.json')
    train_vals = load_json('plots_data/train_rn50up-v4_berHu_default.json')
    val_vals = load_json('plots_data/val_rn50up-v4_berHu_default.json')
    epoch_cap = 80 #min([len(train_vals_conv), len(val_vals_cpmv), len(train_vals), len(val_vals)])
    plot_line_chart(list(range(epoch_cap)),
                    [('TrainLoss-ResnetWithUpProj', train_vals[:epoch_cap], SOLID), ('ValLoss-ResnetWithUpProj', val_vals[:epoch_cap], SOLID),
                     ('ValLoss-ResnetWithUpConv', val_vals_cpmv[:epoch_cap], DASHDOT), ('TrainLoss-ResnetWithUpConv', train_vals_conv[:epoch_cap], DASHDOT)],
                    'Learning curve for Resnet With up-convolution & up-projection models', 'plots/train_vs_val_rn50_upproj_vs_upconv')


def generate_plot_for_resnet_with_upconv_best_model():
    lr_vals = load_json('plots_data/learning_rate_rn50-v5_default.json')
    train_vals = load_json('plots_data/train_rn50-v5_berHu_default.json')
    val_vals = load_json('plots_data/val_rn50-v5_berHu_default.json')
    plot_line_chart([i for i in range(len(train_vals))], [('TrainLoss-ResnetWithUpConv', train_vals), ('ValLoss-ResnetWithUpConv', val_vals)], 'Train vs Validation Loss for Resnet With UpConvolution Model', 'plots/train_vs_val_rn50upconv')


if __name__ == '__main__':
    generate_plot_for_resnet_with_upproj_best_model()
    # generate_plot_for_resnet_with_upconv_best_model()
