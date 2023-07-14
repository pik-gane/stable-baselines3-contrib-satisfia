from os import path

from experiments.cluster.experiment import LEARNING_STEPS, LOG_PATH, ASPIRATIONS, ENV
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar

models = list(
    map(
        lambda a: ARDQN.load(path.join(LOG_PATH, "ARDQN", str(round(a, 2)), "models", f"final_model_{LEARNING_STEPS}")),
        ASPIRATIONS,
    )
)
fig = plot_ar(ENV, models)
# Save plotly figure as html
fig.write_html(path.join(LOG_PATH, "ARDQN", "results.html"))
