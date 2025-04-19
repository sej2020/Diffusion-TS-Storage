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
            run_path = os.path.join(dataset_path, val) # results/electricity/hbs1

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

            hbs = config["hbs"]
            
            data[dataset][hbs] = {
                **result
            }

    return data

def plot_data(data, output_dir):

    for metric in ["Normalized MAE", "Normalized RMSE"]:
        for dataset, hbs_data in data.items():
            plt.style.use('Solarize_Light2')
            plt.figure(figsize=(10, 6))

            x = []
            y = []

            for hbs_val, result_data in sorted(hbs_data.items(), key=lambda item: item[0]):
                x.append(hbs_val)
                y.append(result_data[metric])

            plt.plot(x, y, marker='o')

            plt.title(f"{metric} vs History Block Size for {dataset[0].upper() + dataset[1:]} Dataset")
            plt.xlabel("History Block size")
            plt.ylabel(metric)
            plt.grid(True)

            output_path = os.path.join(output_dir, f"{dataset}_{metric.replace(' ', '_')}_plot.png")
            plt.savefig(output_path)
            plt.close()

def main():
    base_dir = "experiments/hbs_expr/results"
    output_dir = "experiments/hbs_expr/plots"

    data = collect_data(base_dir)
    plot_data(data, output_dir)

if __name__ == "__main__":
    main()