import os
import yaml
import matplotlib.pyplot as plt

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def collect_data(base_dir):
    data = {}
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        
        data[dataset] = {}

        for val in os.listdir(dataset_path):
            run_path = os.path.join(dataset_path, val) # results/electricity/hfr1

            if not os.path.isdir(run_path):
                print(f"Run path not found: {run_path}")
                continue

            
            config_path = os.path.join(run_path, "config.yaml")
            result_path = os.path.join(run_path, "result.yaml")
            if not os.path.isfile(result_path):
                print(f"Result file not found: {result_path}")
                continue

            config = load_yaml(config_path)
            result = load_yaml(result_path)

            hfr = config["hfr"]
            
            data[dataset][hfr] = {
                **result
            }

    return data

def plot_data(data, output_dir):

    for metric in ["Normalized MAE", "Normalized RMSE"]:
        for dataset, hfr_data in data.items():
            plt.style.use('Solarize_Light2')
            plt.figure(figsize=(10, 6))

            x = []
            y = []

            for hfr_val, result_data in sorted(hfr_data.items(), key=lambda item: item[0]):
                x.append(hfr_val)
                y.append(result_data[metric])

            plt.plot(x, y, marker='o')

            plt.title(f"{metric} vs History-to-Feature Ratio for {dataset[0].upper() + dataset[1:]} Dataset")
            plt.xlabel("History-to-Feature Ratio")
            plt.ylabel(metric)
            plt.grid(True)

            output_path = os.path.join(output_dir, f"{dataset}_{metric.replace(' ', '_')}_plot.png")
            plt.savefig(output_path)
            plt.close()


def main(iterations):
    run_data = []
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        base_dir = f"experiments/hfr_expr/results/run{i+1}"
        if not os.path.isdir(base_dir):
            print(f"Run dir not found: {base_dir}")
            continue
        data = collect_data(base_dir)
        run_data.append(data)
    
    agg_data = {}
    for run in run_data:
        for dataset, hfr_data in run.items():
            if dataset not in agg_data:
                agg_data[dataset] = {}
            for hfr_val, result_data in hfr_data.items():
                if hfr_val not in agg_data[dataset]:
                    agg_data[dataset][hfr_val] = {}
                for metric, value in result_data.items():
                    if metric not in agg_data[dataset][hfr_val]:
                        agg_data[dataset][hfr_val][metric] = []
                    agg_data[dataset][hfr_val][metric].append(value)


    # Average the data across runs
    av_data = {}
    for dataset, hfr_data in agg_data.items():
        for hfr_val, result_data in hfr_data.items():
            for metric, values in result_data.items():
                if dataset not in av_data:
                    av_data[dataset] = {}
                if hfr_val not in av_data[dataset]:
                    av_data[dataset][hfr_val] = {}
                if metric not in av_data[dataset][hfr_val]:
                    av_data[dataset][hfr_val][metric] = 0
                av_data[dataset][hfr_val][metric] = sum(values) / len(values)

    output_dir = "experiments/hfr_expr/plots"

    plot_data(av_data, output_dir)

if __name__ == "__main__":
    iterations = 3
    main(iterations)