import numpy as np
import matplotlib.pyplot as plt

data_performance = {
'read_csv': ((10.324, 20.793, 41.413, 84.63), 
    (0.914,  1.925, 3.988, 8.149)),
'describe': ((0.374, 0.681, 1.426, 2.875), 
    (0.235, 0.597, 1.192, 2.264)),
'v_counts': ((0.867, 1.777, 3.447, 6.799), 
    (0.699, 1.401, 2.914, 6.276)),
'statistics': ((2.1, 4.6, 8.3, 20.2), 
    (0.3, 0.7, 1.6, 3.2)),
'sum': ((0.9, 1.2, 2.7, 5.9), 
    (0.1, 0.4, 0.8, 1.9)),
}

plot_params = {
    'read_csv':(('1m', '2m', '4m', '8m'), 'Data size', 'Performance: Pandas vs SDC', True, 'upper left'),
    'describe':(('1m', '2m', '4m', '8m'), 'Data size', 'Performance: Pandas vs SDC', True, 'upper left'),
    'v_counts':(('1m', '2m', '4m', '8m'), 'Data size', 'Data size', 'Performance: Pandas vs SDC', True, 'upper left'),
    'statistics':(('10m', '20m', '40m', '80m'), 'Data size', 'Performance: Pandas vs SDC', True, 'upper left'),
    'sum':(('10m', '20m', '40m', '80m'), 'Data size', 'Performance: Pandas vs SDC', True, 'upper left'),
}


class Plotter:
    def __init__(self, func_id='read_parquet'):
        self.func_id = func_id
        self.x_labels, self.x_title, self.title, self.is_compared, self.label_position = plot_params[self.func_id]
        self.ngroups = len(self.x_labels)

    def autolabel(self, rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

    def plot_performance(self):

        plt.figure(figsize = (16, 8))
        # create plot
        index = np.arange(self.ngroups)
        bar_width = 0.35
        opacity = 0.8

        plt.xlabel(self.x_title, fontsize=16)
        plt.ylabel('Time, s', fontsize=16)
        plt.title(self.title, fontsize=18)
        
        plt.tick_params(labelsize=12)

        if self.is_compared:
            data_pandas, data_sdc = data_performance[self.func_id]
            rects_pandas = plt.bar(index + bar_width, data_pandas, bar_width,
            alpha=opacity,
            label='Pandas')

            plt.xticks(index + bar_width, self.x_labels)
        else:
            data_sdc = data_performance[self.func_id]
            plt.xticks(index, self.x_labels)

        rects_sdc = plt.bar(index, data_sdc, bar_width,
        alpha=opacity,
        label='SDC')

        if self.is_compared:
            self.autolabel(rects_pandas, plt)

        plt.legend(fontsize=16, loc=self.label_position)
        self.autolabel(rects_sdc, plt)

        plt.tight_layout()
        plt.show()
