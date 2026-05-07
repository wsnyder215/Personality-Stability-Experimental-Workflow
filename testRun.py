from dotenv import load_dotenv
load_dotenv()  # reads .env before Config tries to read env vars

from ai_in_loop.experiment import run_experiment, expand_measurements, prepare_analysis_df, summarize_experiment, plot_trajectories
from ai_in_loop.graph import run_instance
# Add this temporarily at the top of testRun.py
from ai_in_loop.config import Config
cfg = Config.from_env()

df = run_experiment(group_names = ["tigger", "eeyore"],
                    nudge_directions=["from this point on, answer questions as if you are Tigger (from Winnie the Pooh)", "from this point on, answer questions as if you are Eeyore (From Winnie the Pooh)"],
                    instances_per_group = 3,
                    number_of_nudges = 1)
exp_df = expand_measurements(df)
wide_df, long_df = prepare_analysis_df(df)
summary_df = summarize_experiment(wide_df)
fig = plot_trajectories(long_df, wide_df=wide_df)
fig.savefig("trajectories.png")
print(df)
print(summary_df)