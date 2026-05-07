import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from .config import Config
from .graph import run_instance
from dotenv import load_dotenv
load_dotenv()  # reads .env before Config tries to read env vars

# mention pesonality test options
def run_experiment(group_names: list[str], 
                   nudge_directions: str | list[str] | list[list[str]] | list[float] | list[list[float]] | list[list[list[float]]], 
                   # group-major nesting:
                   # str                    -> shared prompt
                   # list[str]              -> per-group prompts
                   # list[list[str]]        -> per-group per-nudge prompts
                   #
                   # vector analogue:
                   # list[float]
                   # list[list[float]]
                   # list[list[list[float]]]
                   instances_per_group: int | list[int], 
                   group_initial_personalities = None, 
                   initialization_attempt_limit: int = 3, 
                   number_of_nudges = 1,
                   nudge_destination = None,
                   personality_test = "BigFive"):
    """Main function to run personality-nudging experiment.

    This runs the experiment on a number of model instances
    in each condition group, returning a series of personality
    measurements for each model instance.

    Args:
        group_names: Names of each group.
        nudge_directions: direction each group is pushed during experimentation.
        instances_per_group: number of instances in each treatment group.
        group_initial_personalities: personalities instances of each group are \nmoved to before experimentation.
        initialization_attempt_limit: limit of times personality initialization \nis attempted.
        number_of_nudges: nudges carried out, 
        nudge_destination: target that nudges are aiming for.
        number_of_nudges_or_nudge_destination: nudges carried out, or target nudges are aiming for.
        personality_test: which personality test is being used ("BigFive" or "Enneagram")


    Returns:
        Pandas Dataframe with results from experimentation.
    
    """
    ### initialize variables ###
    results = []

    # group_names and num_groups
    num_groups = len(group_names)
    cfg = Config.from_env()

    # Interpret args according to type. For more more compact argument types, values are assumed 
    # to be constant across groups.
    
    # Helper to fix isinstance() with parameterized generics
    def _check_list_type(obj, types, depth=1):
        """
        Recursively verify obj is a nested list of specified depth
        whose leaves are instances of `types`.

        Examples:
            _check_list_type([1,2], int, 1) -> True
            _check_list_type([[1],[2]], int, 2) -> True
            _check_list_type([[1],["x"]], int, 2) -> False
        """
        if depth == 0:
            return isinstance(obj, types)

        if not isinstance(obj, list):
            return False

        # allow empty lists at any nesting level
        if len(obj) == 0:
            return True

        return all(
            _check_list_type(item, types, depth-1)
            for item in obj
        )

    # nudge_directions
    n_directions = None
    if isinstance(nudge_directions, str):
        n_directions = [nudge_directions]*num_groups
    elif _check_list_type(nudge_directions, str, 1): # all other list[str] objects are assumed to correspond to unique individual strings for each group
        n_directions = nudge_directions # per group
    elif _check_list_type(nudge_directions, str, 2):
        n_directions = nudge_directions # group -> nudge
    elif _check_list_type(nudge_directions, (int, float), 1):
        n_directions = [nudge_directions]*num_groups
    elif _check_list_type(nudge_directions, (int, float), 2):
        n_directions = nudge_directions
    elif _check_list_type(nudge_directions, (int, float), 3):
        n_directions = nudge_directions
    else:
        raise ValueError("Error: invalid type for nudge_directions.")
    # instances_per_group
    if isinstance(instances_per_group, int):
        instances_per_group = [instances_per_group]*num_groups

    # group_initial_personalities
    if _check_list_type(group_initial_personalities, (int, float), 1):
        group_initial_personalities = [list(group_initial_personalities) for _ in range(num_groups)]
    if group_initial_personalities is None:
        group_initial_personalities = [None] * num_groups
    nudge_count = None
    nudge_dest = None

    # number_of_nudges
    if isinstance(number_of_nudges, int):
        nudge_count = [number_of_nudges]*num_groups
    elif _check_list_type(number_of_nudges, int, 1):
        nudge_count = number_of_nudges
    else:
        raise ValueError("Error: invalid type for number_of_nudges")
    
    # nudge_destination
    if _check_list_type(nudge_destination, (int, float), 1) or (nudge_destination is None):
        nudge_dest = [nudge_destination] * num_groups
    elif _check_list_type(nudge_destination, (int, float), 2):
        nudge_dest = nudge_destination
    else:
        raise ValueError("Error: invalid type for nudge_destination")
    
    if len(instances_per_group)!=num_groups:
        raise ValueError("Error: wrong number of groups in instances_per_group")
    if len(nudge_count)!=num_groups:
        raise ValueError("Error: wrong number of groups in nudge_count")
    if len(nudge_dest)!=num_groups:
        raise ValueError("Error: wrong number of groups in nudge_dest")
    if len(n_directions)!=num_groups:
        raise ValueError("Error: wrong number of groups in n_directions")
    if _check_list_type(n_directions, str, 2):
        for sched in n_directions:
            if len(sched) != number_of_nudges:
                raise ValueError(
                    "Error: Each group must have one nudge direction per nudge"
                )
    if len(group_initial_personalities)!=num_groups:
        raise ValueError("Error: wrong number of groups in group_initial_personalities")

    # RUN EXPERIMENT
    # iterate through each group
    instance_id = 0
    for i in range(num_groups):
    
        # for each group, run the requested number of model instances
        for j in range(instances_per_group[i]):
            instance_res = run_instance(cfg,
                                        llm_ID=instance_id,
                                        nudge_direction=n_directions[i], 
                                        initial_personality=group_initial_personalities[i], 
                                        initialization_attempt_limit=initialization_attempt_limit,
                                        nudge_num_limit=nudge_count[i],
                                        nudge_destination=nudge_dest[i],
                                        personality_test=personality_test)
        # TODO open router
            # stick each observation in a list
            results.append({"ID": instance_res["llm_ID"], # WARNING: access using stateObj.attribute might be problematic- not sure yet
                            "group": group_names[i],
                            "initial personality": instance_res["initial_personality"],
                            "initialization attempts": instance_res["initialization_attempts"],
                            "successful initiation completed": instance_res["successfully_initialized"],
                            "total nudges": instance_res["nudges_performed"],
                            "nudge destination": instance_res["nudge_destination"],
                            "nudge destination reached": instance_res["nudge_destination_reached"],
                            "nudge direction": instance_res["nudge_direction"],
                            "measurements": instance_res["measurements"],
                            "personality test": personality_test, # Is this much necessary? YES IT IS!!!
                            })
            instance_id+=1
            
    return pd.DataFrame(results)

def expand_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Explode measurements column into long format.
    
    Each row becomes one measurement per instance per timestep.
    Personality vector dimensions are expanded into separate columns (dim_0, dim_1, ...).

    Args:
        df: Wide-format results dataframe from run_experiment()

    Returns:
        Long-format dataframe with one row per (instance, timestep)
    """
    records = []
    for _, row in df.iterrows(): # could vectorize for large experiments (probably not necessary)
        for t, measurement in enumerate(row["measurements"]):
            # measurement is assumed to be a list/array of floats (personality dims)
            record = {
                "ID": row["ID"],
                "group": row["group"],
                "timestep": t,
                "nudge_direction": str(row["nudge direction"]),  # stringify for groupby safety #possible TODO: does this make sense?
            }
            if isinstance(measurement, (list, np.ndarray)):
                for d, val in enumerate(measurement):
                    record[f"dim_{d}"] = val
            else:
                record["dim_0"] = measurement  # scalar fallback
            records.append(record)

    return pd.DataFrame(records)


def prepare_analysis_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape wide experiment results into analysis-ready formats, with anomaly flags.

    Args:
        df: Wide-format results dataframe from run_experiment()

    Returns:
        Tuple of (wide_flagged_df, long_df) where:
            - wide_flagged_df: original df with anomaly flag columns appended
            - long_df: long-format measurements from expand_measurements()
    """
    wide = df.copy()

    # --- Anomaly flags ---
    wide["flag_init_failed"] = (
        wide["successful initiation completed"] == False  # distinguishes False from None
    )
    wide["flag_dest_not_reached"] = (
        wide["nudge destination"].notna() & (wide["nudge destination reached"] == False)
    )
    wide["flag_no_measurements"] = wide["measurements"].apply(
        lambda m: not isinstance(m, list) or len(m) == 0
    )
    wide["any_flag"] = wide[["flag_init_failed", "flag_dest_not_reached", "flag_no_measurements"]].any(axis=1)

    long = expand_measurements(wide)

    return wide, long

def summarize_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level summary statistics from wide-format results.

    Args:
        df: Wide-format results dataframe (flagged or unflagged)

    Returns:
        Summary dataframe indexed by group
    """
    def safe_mean(series):
        return series.dropna().mean()

    agg_funcs = {
        "n_instances": ("ID", "count"),
        "init_success_rate": ("successful initiation completed", lambda x: x.eq(True).mean()),
        "mean_init_attempts": ("initialization attempts", safe_mean),
        "mean_nudges_performed": ("total nudges", safe_mean),
        "dest_reached_rate": ("nudge destination reached", lambda x: x.eq(True).mean()),
    }
    
    if "any_flag" in df.columns:
        agg_funcs["flagged_instances"] = ("any_flag", "sum")

    summary = df.groupby("group").agg(**agg_funcs).round(3)

    return summary


def plot_trajectories(
    long_df: pd.DataFrame,
    dims: list[int] | None = None,
    show_group_mean: bool = True,
    alpha_individual: float = 0.2,
    exclude_flagged: bool = True,
    wide_df: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot personality trajectories over nudge timesteps, by group.

    Args:
        long_df: Long-format dataframe from expand_measurements()
        dims: Which personality dimensions to plot. Defaults to all found.
        show_group_mean: Whether to overlay group mean trajectory.
        alpha_individual: Transparency for individual instance lines.
        exclude_flagged: If True and wide_df provided, drops flagged instances.
        wide_df: Wide-format df with anomaly flags, used if exclude_flagged=True.

    Returns:
        Matplotlib Figure
    """
    plot_df = long_df.copy()

    if exclude_flagged and wide_df is not None:
        flagged_ids = wide_df.loc[wide_df["any_flag"], "ID"]
        plot_df = plot_df[~plot_df["ID"].isin(flagged_ids)]

    dim_cols = [c for c in plot_df.columns if c.startswith("dim_")]
    if dims is not None:
        dim_cols = [f"dim_{d}" for d in dims if f"dim_{d}" in plot_df.columns]
    if plot_df.empty:
        raise(ValueError("Error: plot_df is empty and cannot be plotted."))
    groups = plot_df["group"].unique()
    colors = cm.tab10(np.linspace(0, 1, len(groups)))
    group_color = dict(zip(groups, colors))

    n_dims = len(dim_cols)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4), squeeze=False)

    for col_idx, dim in enumerate(dim_cols):
        ax = axes[0][col_idx]
        for group in groups:
            gdf = plot_df[plot_df["group"] == group]
            color = group_color[group]

            # Individual trajectories
            for instance_id, idf in gdf.groupby("ID"):
                ax.plot(idf["timestep"], idf[dim], color=color, alpha=alpha_individual, linewidth=1)

            # Group mean
            if show_group_mean:
                mean_traj = gdf.groupby("timestep")[dim].mean()
                ax.plot(mean_traj.index, mean_traj.values, color=color, linewidth=2.5, label=group)

        ax.set_title(f"Personality {dim}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()

    fig.suptitle("Personality Trajectories by Group", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig

def modeling_prep():
    pass
"""
a reasonable workflow might look like:

wide_df = run_experiment(...)
wide_flagged_df, long_df = prepare_analysis_df(wide_df)

print(summarize_experiment(wide_flagged_df))
fig = plot_trajectories(long_df, wide_df=wide_flagged_df)
fig.savefig("trajectories.png")

"""

