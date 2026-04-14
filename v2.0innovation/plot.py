import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse



def plot_time_series(rh: pd.DataFrame, select:list=[], all:bool=True, content:str="Value") -> None:
    if all:
        select = [i for i in range(len(rh.columns))]  # Adjust based on your DataFrame structure
    
    match content:
        case "revenue":
            ylabel = "Revenue"
            title = "Revenue History of Agents"
        case "action":
            ylabel = "Action"
            title = "Action History of Agents"
        case "innovation input":
            ylabel = "Innovation Input"
            title = "Innovation Input History of Agents"
        case "capital":
            ylabel = "Capital"
            title = "Capital History of Agents"
        case "tech":
            ylabel = "Tech Level"
            title = "Tech Level History of Agents"
        case _:
            ylabel = "Value"
            title = "Value History of Agents"

    for i in select:
        plt.plot(rh[f"firm_{i}"], label=f"Agent {i} {content}")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
    file_path = Path(f"./test_results/{args.experiment_idx}_{content.lower().replace(' ', '_')}_history.pdf")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path, format="pdf")
    print(f"Saved plot for to {file_path}")
    plt.close()

def get_average(data: pd.DataFrame, window_size: int, episode: int) -> pd.DataFrame:
    averaged_data = data[0: window_size]
    for i in range(episode):
        j = i+1
        averaged_data = data[0 + j* window_size: window_size + j* window_size].reset_index(drop=True) / (j+1) + averaged_data * (j / (j+1))
    return averaged_data


def get_data(args):
    # string = f"{args.experiment_idx:03d}"
    # os.getcwd return terminal current path not the file's path
    output_dir = os.path.join(os.getcwd(), "v2.0innovation", "test_data", f"{args.experiment_idx}_test_results")
    print(f"Data source directory: {output_dir}")
    revenue_csv = os.path.join(output_dir, "revenue_history.csv")
    action_csv = os.path.join(output_dir, "action_history.csv")
    innovation_csv = os.path.join(output_dir, "balance_log_innovation_input.csv")
    capital_csv = os.path.join(output_dir, "balance_log_capital.csv")
    tech_csv = os.path.join(output_dir, "balance_log_tech.csv")

    rh = pd.read_csv(revenue_csv)
    ah = pd.read_csv(action_csv)
    iv = pd.read_csv(innovation_csv)
    cp = pd.read_csv(capital_csv)
    th = pd.read_csv(tech_csv)

    avg_rh = get_average(rh, args.num_steps, args.episode)
    avg_iv = get_average(iv, args.num_steps, args.episode)
    avg_cp = get_average(cp, args.num_steps, args.episode)
    avg_th = get_average(th, args.num_steps, args.episode)
    avg_ah = get_average(ah, args.num_steps, args.episode)

    plot_time_series(avg_rh, all=True, content="revenue")
    plot_time_series(avg_iv, all=True, content="innovation input")
    plot_time_series(avg_cp, all=True, content="capital")
    plot_time_series(avg_th, all=True, content="tech")
    plot_time_series(avg_ah, all=True, content="action")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_idx", type=str, default="e001")
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=100)
    args = parser.parse_args()
    get_data(args)