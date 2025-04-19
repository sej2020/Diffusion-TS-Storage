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
        
        data[dataset] = {"channel": {}, "layer": {}}

        for category in ["channel", "layer"]:
            category_path = os.path.join(dataset_path, category)

            if not os.path.isdir(category_path):
                print(f"Category path not found: {category_path}")
                continue

            for subdir in os.listdir(category_path):
                subdir_path = os.path.join(category_path, subdir)
                
                config_path = os.path.join(subdir_path, "config.yaml")
                result_path = os.path.join(subdir_path, "result.yaml")
                if not os.path.isfile(result_path):
                    print(f"Result file not found: {result_path}")
                    continue

                config = load_yaml(config_path)
                result = load_yaml(result_path)

                total_params = config["total_params"]
                
                data[dataset][category][subdir] = {
                    "total_params": total_params,
                    **result
                }

    return data

def plot_data(data, output_dir):

    for metric in ["Normalized MAE", "Normalized RMSE"]:
        for dataset, categories in data.items():
            plt.style.use('Solarize_Light2')
            plt.figure(figsize=(10, 6))

            for category, subdirs in categories.items():
                x = []
                y = []

                for subdir, values in sorted(subdirs.items(), key=lambda item: item[1]["total_params"], reverse=True):
                    x.append(values["total_params"])
                    y.append(values[metric])

                plt.plot(x, y, marker='o', label=category)

            plt.title(f"{metric} vs Total Params for {dataset}")
            plt.xlabel("Total Params")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.gca().invert_xaxis()

            output_path = os.path.join(output_dir, f"{dataset}_{metric.replace(' ', '_')}_plot.png")
            plt.savefig(output_path)
            plt.close()

def main():
    base_dir = "experiments/cut_params_expr/results"
    output_dir = "experiments/cut_params_expr/plots"

    data = collect_data(base_dir)
    plot_data(data, output_dir)

if __name__ == "__main__":
    main()