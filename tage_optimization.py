import optuna, subprocess, re

def objective(trial):
    params = [
        trial.suggest_int("LOGLB", 4, 8),
        trial.suggest_int("NUMG", 4, 16),
        trial.suggest_int("LOGG", 9, 14),
        trial.suggest_int("LOGB", 8, 14),
        trial.suggest_int("TAGW", 7, 15),
        trial.suggest_int("GHIST", 40, 200),
        trial.suggest_int("LOGP1", 10, 16),
        trial.suggest_int("GHIST1", 4, 16),
    ]

    p = ",".join(map(str, params))
    outdir = f"out/out_{trial.number}"

    subprocess.run(f'./compile_custom {trial.number} -DPREDICTOR="tage<{p}>"', shell=True)
    subprocess.run(f'./run_all ./cbp_{trial.number} ./cbp-ng_training_traces {outdir} > /dev/null 2>&1', shell=True)
    subprocess.run(f'rm ./cbp_{trial.number}', shell=True)

    r = subprocess.run(f'./predictor_metrics.py {outdir} | ./vfs.py', shell=True, capture_output=True, text=True)

    subprocess.run(f'rm -r ./out/out_{trial.number}', shell=True)

    m = re.search(r"([0-9.]+)", r.stdout)
    return -float(m.group(1)) if m else 1e9

study = optuna.create_study(study_name="tage_optimization", storage="sqlite:///tage.db", load_if_exists=True)
study.optimize(objective, n_trials=300, n_jobs=4)

print(study.best_params, -study.best_value)