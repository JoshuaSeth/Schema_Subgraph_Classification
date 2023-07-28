# Quick test to see correlations between topology and results, these are not statistically significant of course rather they are anecdotal evidence and cuase of further inquiry
from collections import defaultdict
from scipy.stats.stats import pearsonr

global_top_str = '''
$\mu$ entities & 9.22 & 9.20 & 4.26 & 4.25 & 4.57 & 4.55 & 6.26 & 6.22 & 3.32 & 3.30 & 9.10 & 8.02 \\ \hline
$\mu$ relations & 3.39 & 3.42 & 1.59 & 1.60 & 1.58 & 1.57 & 2.70 & 2.68 & 1.16 & 1.15 & 3.66 & 3.26 \\ \hline
$\sigma$ entities & 9.64 & 9.43 & 3.10 & 3.08 & 3.39 & 3.33 & 3.56 & 3.60 & 2.15 & 2.18 & 5.97 & 6.43 \\ \hline
$\sigma$ relations & 2.54 & 2.58 & 1.11 & 1.12 & 1.03 & 1.03 & 1.53 & 1.55 & 0.42 & 0.41 & 2.38 & 2.58 \\ \hline


$\mu$ degrees & \cellcolor{green}2.47 & \cellcolor{green}2.47 & 1.23 & 1.23 & 1.29 & 1.29 & 1.76 & 1.75 & \cellcolor{red}1.00 & 0.99 & 2.45 & 2.18 \\ \hline
$\sigma$  degrees  & \cellcolor{green}2.37 & 2.31 & 0.72 & 0.72 & 0.78 & 0.77 & 0.87 & 0.88 & \cellcolor{red}0.44 & \cellcolor{red}0.45 & 1.46 & 1.59 \\ \hline
$\mu$ degree centralities & 0.32 & 0.32 & 0.56 & 0.56 & 0.51 & 0.51 & 0.35 & 0.36 & \cellcolor{green}0.67 & \cellcolor{green}0.68 & \cellcolor{red}0.24 & 0.30 \\ \hline
$\sigma$  degree centralities & \cellcolor{red}0.37 & \cellcolor{red}0.37 & \cellcolor{green}0.42 & \cellcolor{green}0.42 & \cellcolor{green}0.42 & \cellcolor{green}0.42 & 0.37 & 0.37 & \cellcolor{green}0.41 & \cellcolor{green}0.41 & 0.30 & 0.34 \\ \hline

$\mu$ closeness centralities & 0.32 & 0.32 & 0.56 & 0.56 & 0.51 & 0.51 & 0.36 & 0.36 & \cellcolor{green}0.67 & \cellcolor{green}0.68 & \cellcolor{red}0.25 & 0.31 \\ \hline
$\sigma$  closeness centralities & \cellcolor{red}0.37 & \cellcolor{red}0.37 & 0.42 & 0.42 & 0.42 & 0.42 & 0.37 & 0.37 & \cellcolor{green}0.41 & \cellcolor{green}0.41 & 0.30 & 0.34 \\ \hline
$\mu$ clusterings & 0.18 & 0.18 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & \cellcolor{green}0.24 & \cellcolor{green}0.24 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & 0.06 & 0.05 \\ \hline
$\sigma$ clusterings & 0.19 & 0.19 & 0.04 & \cellcolor{red}0.03 & \cellcolor{red}0.03 & \cellcolor{red}0.03 & \cellcolor{green}0.21 & \cellcolor{green}0.22 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & 0.12 & 0.11 \\ \hline



$\mu$ modularities & 0.04 & 0.04 & 0.16 & 0.16 & 0.18 & 0.17 & \cellcolor{red}0.01 & \cellcolor{red}0.01 & 0.08 & 0.07 & \cellcolor{green}0.33 & \cellcolor{green}0.30 \\ \hline
$\sigma$  modularities  & \cellcolor{red}0.09 & \cellcolor{red}0.09 & \cellcolor{green}0.24 & \cellcolor{green}0.24 & \cellcolor{green}0.25 & \cellcolor{green}0.25 & 0.05 & 0.05 & 0.18 & 0.18 & 0.24 & 0.24 \\ \hline

$\mu$ densities & 0.32 & 0.32 & 0.56 & 0.56 & 0.51 & 0.51 & 0.35 & 0.36 & \cellcolor{green}0.67 & \cellcolor{green}0.68 & \cellcolor{red}0.24 & 0.30 \\ \hline
$\sigma$  densities  & \cellcolor{red}0.37 & \cellcolor{red}0.37 & \cellcolor{green}0.42 & \cellcolor{green}0.42 & \cellcolor{green}0.42 & \cellcolor{green}0.42 & 0.37 & 0.37 & \cellcolor{green}0.41 & \cellcolor{green}0.41 & 0.30 & 0.34 \\ \hline'''.replace('\cellcolor{green}', '').replace('\cellcolor{red}', '').replace(',', '')

topology = {}
topology['AND'] = {}
topology['OR'] = {}
for line in global_top_str.split(' \\ \hline'):
    metric_name = line.split('&')[0]
    topology['AND'][metric_name] = []
    topology['OR'][metric_name] = []

    for i, value in enumerate(line.strip().split('&')[1:]):
        mode = 'AND'
        if i % 2 == 0:
            mode = 'OR'
        topology[mode][metric_name].append(float(value.strip()))


graph_class_table = '''False & False & 0.78 & 0.52 & 0.75 & 0.53 & \textbf{0.81} & 0.52 & 0.80 & 0.52 & 0.76 & \textbf{0.54} & 0.56 & 0.53 \\ \hline
False & True & \textbf{0.99} & \textbf{0.97} & 0.95 & 0.89 & 0.95 & 0.93 & 0.99 & 0.94 & 0.75 & 0.83 & 0.85 & 0.92 \\ \hline
True & False & 0.78 & 0.55 & 0.76 & 0.53 & 0.82 & 0.52 & 0.81 & 0.55 & 0.76 & 0.55 & \textbf{0.99} & \textbf{0.70} \\ \hline
True & True & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 & 0.99 '''.replace('\textbf{', '').replace('}', '')


def res(): return defaultdict(res)


results = res()
for line in graph_class_table.split(' \\ \hline'):
    embed = eval(line.split('&')[0].strip())
    types = eval(line.split('&')[1].strip())

    for i, value in enumerate(line.split('&')[2:]):
        mode = 'AND'
        if i % 2 == 0:
            mode = 'OR'
        if not isinstance(results[embed][types][mode], list):
            results[embed][types][mode] = []
        results[embed][types][mode].append(float(value.strip()))


def notmode(mode):
    if mode == 'OR':
        return 'AND'
    return 'OR'


for mode, metrics in topology.items():
    for metric, values in metrics.items():
        if len(values) == len(results[False][True][mode]):
            vals = values + topology[notmode(mode)][metric]
            res = results[False][True][mode] + \
                results[False][True][notmode(mode)]
            corr = pearsonr(vals, res)
            print(metric, corr)

# print(topology)
