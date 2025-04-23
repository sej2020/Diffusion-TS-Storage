    

for i in {1..3}
do    
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/electricity/hfr0_25" --config 'electricity_config' --hfr 0.25 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/electricity/hfr0_5" --config 'electricity_config' --hfr 0.5 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/electricity/hfr1_0" --config 'electricity_config' --hfr 1.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/electricity/hfr2_0" --config 'electricity_config' --hfr 2.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/electricity/hfr4_0" --config 'electricity_config' --hfr 4.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/electricity/hfr8_0" --config 'electricity_config' --hfr 8.0 --device 'cuda:1'

    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/weather/hfr0_25" --config 'weather_config' --hfr 0.25 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/weather/hfr0_5" --config 'weather_config' --hfr 0.5 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/weather/hfr1_0" --config 'weather_config' --hfr 1.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/weather/hfr2_0" --config 'weather_config' --hfr 2.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/weather/hfr4_0" --config 'weather_config' --hfr 4.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/weather/hfr8_0" --config 'weather_config' --hfr 8.0 --device 'cuda:1'

    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/traffic/hfr0_25" --config 'traffic_config' --hfr 0.25 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/traffic/hfr0_5" --config 'traffic_config' --hfr 0.5 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/traffic/hfr1_0" --config 'traffic_config' --hfr 1.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/traffic/hfr2_0" --config 'traffic_config' --hfr 2.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/traffic/hfr4_0" --config 'traffic_config' --hfr 4.0 --device 'cuda:1'
    python -m experiments.hfr_expr.run_expr --save_folder "experiments/hfr_expr/results/run${i}/traffic/hfr8_0" --config 'traffic_config' --hfr 8.0 --device 'cuda:1'
done
