import matplotlib.pyplot as plt

color_pallete = ['red', 'blue', 'green', 'yellow', 'black', 'purple']


def plot_line_chart(X_axis_index: list, Y_axis_data, title, file_path, x_label='Epochs', y_label='Loss', legend_loc=1):
    """

    Parameters
    ----------
    X_axis_val : should a list of indexes on x-axis
        for example : [0, 1, 2, 3, 4 ....] epochs
    Y_axis_vals : should be a list of tuples of the two things
        (Legend_value, Y_axis_vals)
        where Legend_value -> value to show in legend
        and Y_axis_vals -> list of values to plot on Y-axis against corresponding x-indices. For example [0.001, 0.003, 0.005...]. This should match the number of values in X_axis_index argument
    title : title of the figure
    file_path : filename where figure needs to be stored

    Returns
    -------

    """
    plt.figure(figsize=(16, 12))
    legend_vals = []
    for i, (legend, Y_axis_val) in enumerate(Y_axis_data):
        plt.plot(X_axis_index, Y_axis_val, color_pallete[i])
        legend_vals.append(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', fillstyle='left')
    plt.legend(legend_vals, loc=legend_loc)
    # plt.xlim((in_sd, in_ed))
    plt.minorticks_on()
    plt.xticks()
    plt.savefig(f'{file_path}.png')
    plt.clf()
