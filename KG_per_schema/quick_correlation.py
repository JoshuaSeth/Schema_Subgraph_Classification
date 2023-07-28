# Quick test to see correlations between topology and results, these are not statistically significant of course rather they are anecdotal evidence and cuase of further inquiry
from collections import defaultdict
from scipy.stats.stats import pearsonr

global_top_str = '''
Entities & 7,374 & 9,484 & 3,690 & 4,438 & 4,559 & 5,379 & 4,954 & 5,592 & 1,294 & 1,426 & 9,090 & 10,782 \\ \hline
Relations & 6,375 & 8,147 & 216 & 262 & 218 & 248 & 1,793 & 1,993 & 0 & 0 & 2922 & 3,092 \\ \hline

$\mu$ degree & \cellcolor{green}3.06 & \cellcolor{green}3.07 & 2.36 & 2.37 & 2.44 & 2.44 & 2.65 & 2.65 & \cellcolor{red}2.01 & \cellcolor{red}2.01 & 2.61 & 2.68 \\ \hline
$\sigma$ degree & \cellcolor{green}72.68 & \cellcolor{green}72.79 & 20.86 & 20.76 & 21.72 & 21.67 & 62.15 & 62.23 & 19.62 & 19.69 & 47.01 & 47.12 \\ \hline

$\mu$ degree centrality & 0.00043 & 0.00043 & 0.00156 & 0.00158 & 0.00154 & 0.00154 & 0.00064 & 0.00063 & 0.00202 & 0.00201 & 0.00260 & 0.00039 \\ \hline
$\sigma$ degree centrality & 0.01 & 0.01 & 0.01 & 0.01 & 0.01 & 0.01 & 0.01 & 0.01 & \cellcolor{green}0.02 & \cellcolor{green}0.02 & \cellcolor{green}0.02 & 0.01 \\ \hline


$\mu$ closeness centrality & 0.44 & 0.44 & 0.30 & 0.30 & 0.30 & 0.30 & \cellcolor{green}0.48 & \cellcolor{green}0.48 & \cellcolor{red}0.26 & \cellcolor{red}0.26 & 0.31 & 0.31 \\ \hline
$\sigma$ closeness centrality & \cellcolor{green}0.06 & \cellcolor{green}0.06 & \cellcolor{red}0.03 & \cellcolor{red}0.03 & \cellcolor{red}0.03 & \cellcolor{red}0.03 & 0.04 & 0.04 & \cellcolor{green}0.07 & \cellcolor{green}0.07 & 0.04 & 0.04 \\ \hline


$\mu$ clustering coefficient & 0.51 & 0.51 & 0.03 & 0.03 & 0.03 & 0.03 & \cellcolor{green}0.56 & \cellcolor{green}0.56 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & 0.16 & 0.17 \\ \hline
$\sigma$ clustering coefficient & \cellcolor{green}0.45 & \cellcolor{green}0.45 & 0.15 & 0.15 & 0.16 & 0.16 & 0.49 & 0.49 & \cellcolor{red}0.00 & \cellcolor{red}0.00 & 0.34 & 0.35 \\ \hline


modularity & 0.47 & 0.48 & \cellcolor{green}0.66 & \cellcolor{green}0.66 & \cellcolor{green}0.65 & \cellcolor{green}0.65 & \cellcolor{red}0.29 & \cellcolor{red}0.29 & 0.61 & 0.61 & 0.62 & 0.64 \\ \hline'''.replace('\cellcolor{green}', '').replace('\cellcolor{red}', '').replace(',', '')

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
