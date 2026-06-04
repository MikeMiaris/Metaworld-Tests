# Σύνοψη Αποτελεσμάτων PPO σε Meta-World

Το παρόν αρχείο συγκεντρώνει τα βασικά αποτελέσματα αξιολόγησης των PPO πειραμάτων που περιλαμβάνονται στο repository και παραπέμπει στα αντίστοιχα notebooks και σχήματα.

> Αν το GitHub δεν εμφανίζει σωστά κάποιο notebook, μπορεί να ανοιχτεί μέσω των συνδέσμων nbviewer που δίνονται παρακάτω.

---

## Δομή repository

### Single-task PPO πειράματα

| Φάκελος | Περιβάλλον | Κύρια αρχεία αποτελεσμάτων |
|---|---|---|
| `button_press_v3/` | `button-press-v3` | notebook + σχήμα |
| `basketball/` | `basketball-v3` | `ppo_basketball_results.csv` + σχήματα |
| `push_v3/` | `push-v3` | aggregate/per-episode CSVs + σχήματα |
| `pick-place/` | `pick-place-v3` | aggregate/per-episode CSVs + σχήματα |

### Custom multi-task PPO πειράματα

| Φάκελος | Tasks | Κύρια αρχεία αποτελεσμάτων |
|---|---|---|
| `custom_button_push/` | `button-press-v3` + `push-v3` | summary CSVs + σχήματα |
| `custom_basketball_pick_place/` | `basketball-v3` + `pick-place-v3` | summary CSVs + σχήματα |
| `custom_push_pickplace/` | `push-v3` + `pick-place-v3` | summary CSVs + σχήματα |

---

## Σύνδεσμοι notebooks

---

# 1. Single-task πειράματα

## 1.1 `button-press-v3`

Στον φάκελο `button_press_v3/` υπάρχουν το notebook `button_press_results.ipynb`, το evaluation script, το training script και το σχήμα `fig1_success_rate_heatmap.png`.

Οι PPO configurations που εμφανίζονται στο notebook είναι:

- `base_button`
- `careful_button`
- `light_entropy_button`
- `short_rollout_button`

Το checkpoint evaluation χρησιμοποιεί:

- checkpoints: `100000`, `200000`, `300000`, `400000`, `500000`
- groups: `train`, `test`
- metrics: success rate, return, first-success step

### Σχήμα

![Button-press success heatmap](button_press_v3/fig1_success_rate_heatmap.png)

---

## 1.2 `basketball-v3`

Source CSV:

```text
basketball/ppo_basketball_results.csv
```

Πειραματική διάταξη:

- environment: `basketball-v3`
- total timesteps: `6,000,000`
- parallel envs: `4`
- train tasks: `45`
- test tasks: `5`
- split seed: `67`
- train seeds: `11`, `22`, `33`
- configs: `A_basketball_main`, `B_basketball_entropy`

### Αποτελέσματα ανά run

| Config | Train seed | Train success | Test success | Train return | Test return | Success gap |
|---|---:|---:|---:|---:|---:|---:|
| `A_basketball_main` | 11 | 0.956 | 0.600 | 4446.56 | 3811.52 | 0.356 |
| `A_basketball_main` | 22 | 1.000 | 0.800 | 4364.21 | 4012.98 | 0.200 |
| `A_basketball_main` | 33 | 0.978 | 1.000 | 4438.33 | 4456.54 | -0.022 |
| `B_basketball_entropy` | 11 | 0.822 | 0.400 | 4158.06 | 3679.44 | 0.422 |
| `B_basketball_entropy` | 22 | 0.978 | 0.800 | 4438.20 | 3900.01 | 0.178 |
| `B_basketball_entropy` | 33 | 1.000 | 1.000 | 4494.50 | 4501.43 | 0.000 |

### Μέσοι όροι ανά config

| Config | Mean train success | Mean test success | Mean train return | Mean test return |
|---|---:|---:|---:|---:|
| `A_basketball_main` | 0.978 | 0.800 | 4416.37 | 4093.68 |
| `B_basketball_entropy` | 0.933 | 0.733 | 4363.59 | 4026.96 |

### Σχήματα

![Basketball mean success across seeds](basketball/basketball_figures/basketball_mean_success_across_seeds.png)

![Basketball success by config and seed](basketball/basketball_figures/basketball_success_by_config_seed.png)

![Basketball success gap by run](basketball/basketball_figures/basketball_success_gap_by_run.png)

---

## 1.3 `push-v3`

Source CSV:

```text
push_v3/push_v3_ppo_split_runs/results/push_v3_aggregate_results.csv
```

Πειραματική διάταξη:

- environment: `push-v3`
- total timesteps: `6,000,000`
- parallel envs: `4`
- train/test split: `45/5`
- split seeds: `67`, `68`, `75`
- train seed: `11`
- configs: `base_push`, `careful_push`, `short_rollout_push`
- `VecNormalize`: `True`

### Μέσοι όροι ανά config

| Config | Mean train success | Mean test success | Mean train return | Mean test return |
|---|---:|---:|---:|---:|
| `base_push` | 0.837 | 0.933 | 262.14 | 293.73 |
| `careful_push` | 0.985 | 1.000 | 174.92 | 158.05 |
| `short_rollout_push` | 0.474 | 0.533 | 470.77 | 172.29 |

### Test success ανά split

| Config | Split 0 | Split 1 | Split 2 |
|---|---:|---:|---:|
| `base_push` | 1.000 | 0.800 | 1.000 |
| `careful_push` | 1.000 | 1.000 | 1.000 |
| `short_rollout_push` | 0.800 | 0.400 | 0.400 |

### Σχήματα

![Push-v3 mean success across splits](push_v3/push_v3_ppo_split_runs/figures/push_v3_mean_success_across_splits.png)

![Push-v3 success gap by run](push_v3/push_v3_ppo_split_runs/figures/push_v3_success_gap_by_run.png)

---

## 1.4 `pick-place-v3`

Source CSV:

```text
pick-place/pick-place_v3_ppo_split_runs/results/pick-place_v3_aggregate_results.csv
```

Πειραματική διάταξη:

- environment: `pick-place-v3`
- total timesteps: `6,000,000`
- parallel envs: `4`
- train/test split: `45/5`
- split seeds: `67`, `68`, `75`
- train seed: `11`
- configs: `base_pick`, `careful_pick`, `short_rollout_pick`, `light_entropy_pick`
- `VecNormalize`: `True`

### Μέσοι όροι ανά config

| Config | Mean train success | Mean test success | Mean train return | Mean test return |
|---|---:|---:|---:|---:|
| `base_pick` | 0.837 | 0.733 | 72.08 | 51.57 |
| `careful_pick` | 0.993 | 0.933 | 58.69 | 59.14 |
| `short_rollout_pick` | 0.089 | 0.000 | 273.56 | 196.14 |
| `light_entropy_pick` | 0.956 | 1.000 | 47.33 | 66.08 |

### Test success ανά split

| Config | Split 0 | Split 1 | Split 2 |
|---|---:|---:|---:|
| `base_pick` | 0.800 | 0.400 | 1.000 |
| `careful_pick` | 1.000 | 0.800 | 1.000 |
| `short_rollout_pick` | 0.000 | 0.000 | 0.000 |
| `light_entropy_pick` | 1.000 | 1.000 | 1.000 |

### Σχήμα

![Pick-place mean success by config](pick-place/pick-place_v3_ppo_split_runs/figures/pick_place_mean_success_by_config.png)

---

# 2. Custom multi-task πειράματα

## 2.1 Custom MT: `button-press-v3` + `push-v3`

Source CSVs:

```text
custom_button_push/button_push_eval_results_100ep_3seeds/button_push_eval_summary.csv
custom_button_push/button_push_eval_results_100ep_3seeds/button_push_success_rate_pivot.csv
```

Πειραματική διάταξη:

- tasks: `button-press-v3`, `push-v3`
- configs: `base`, `careful`, `explore`
- evaluation episodes: `300` ανά task/config
- metrics: success rate, return, episode length, first success step

### Success rate pivot

| Config | `button-press-v3` | `push-v3` |
|---|---:|---:|
| `base` | 1.000 | 0.973 |
| `careful` | 1.000 | 0.980 |
| `explore` | 1.000 | 0.560 |

### Αναλυτική σύνοψη

| Config | Task | Success rate | Avg return | Avg episode length | Avg first success step | Episodes |
|---|---|---:|---:|---:|---:|---:|
| `base` | `button-press-v3` | 1.000 | 64.17 | 37.39 | 37.39 | 300 |
| `base` | `push-v3` | 0.973 | 159.32 | 50.66 | 38.35 | 300 |
| `careful` | `button-press-v3` | 1.000 | 58.88 | 37.39 | 37.39 | 300 |
| `careful` | `push-v3` | 0.980 | 207.75 | 47.53 | 38.30 | 300 |
| `explore` | `button-press-v3` | 1.000 | 75.03 | 37.92 | 37.92 | 300 |
| `explore` | `push-v3` | 0.560 | 1148.26 | 270.43 | 90.06 | 300 |

### Σχήματα

![Custom Button-Push success by config](custom_button_push/button_push_custom_mt_figures/custom_wrapper_success_by_config.png)

![Individual env success by config](custom_button_push/button_push_custom_mt_figures/individual_env_success_by_config.png)

---

## 2.2 Custom MT: `basketball-v3` + `pick-place-v3`

Source CSVs:

```text
custom_basketball_pick_place/basketball_pickplace_eval_results/basketball_pickplace_eval_summary.csv
custom_basketball_pick_place/basketball_pickplace_eval_results/basketball_pickplace_success_rate_pivot.csv
```

Πειραματική διάταξη:

- tasks: `basketball-v3`, `pick-place-v3`
- configs: `base`, `careful`, `explore`
- evaluation episodes: `50` ανά task/config
- metrics: success rate, return, episode length, first success step

### Success rate pivot

| Config | `basketball-v3` | `pick-place-v3` |
|---|---:|---:|
| `base` | 0.960 | 1.000 |
| `careful` | 1.000 | 1.000 |
| `explore` | 0.900 | 1.000 |

### Αναλυτική σύνοψη

| Config | Task | Success rate | Avg return | Avg episode length | Avg first success step | Episodes |
|---|---|---:|---:|---:|---:|---:|
| `base` | `basketball-v3` | 0.960 | 3632.40 | 500.00 | 55.31 | 50 |
| `base` | `pick-place-v3` | 1.000 | 4554.93 | 500.00 | 49.88 | 50 |
| `careful` | `basketball-v3` | 1.000 | 4573.85 | 500.00 | 54.98 | 50 |
| `careful` | `pick-place-v3` | 1.000 | 4617.36 | 500.00 | 42.22 | 50 |
| `explore` | `basketball-v3` | 0.900 | 1787.81 | 500.00 | 68.11 | 50 |
| `explore` | `pick-place-v3` | 1.000 | 4125.38 | 500.00 | 42.10 | 50 |

### Σχήματα

![Basketball-PickPlace success by config](custom_basketball_pick_place/basketball_pickplace_custom_mt_figures/basketball_pickplace_success_by_config.png)

![Basketball-PickPlace return by config](custom_basketball_pick_place/basketball_pickplace_custom_mt_figures/basketball_pickplace_return_by_config.png)

---

## 2.3 Custom MT: `push-v3` + `pick-place-v3`

Source CSVs:

```text
custom_push_pickplace/push_pickplace_eval_results/push_pickplace_summary.csv
custom_push_pickplace/push_pickplace_eval_results/push_pickplace_success_rate_pivot.csv
```

Πειραματική διάταξη:

- tasks: `push-v3`, `pick-place-v3`
- configs: `base`, `careful`, `explore`
- evaluation episodes: `300` ανά task/config
- metrics: success rate, return, episode length, first success step

### Success rate pivot

| Config | `pick-place-v3` | `push-v3` |
|---|---:|---:|
| `base` | 1.000 | 0.973 |
| `careful` | 1.000 | 1.000 |
| `explore` | 0.947 | 0.960 |

### Αναλυτική σύνοψη

| Config | Task | Success rate | Avg return | Avg episode length | Avg first success step | Episodes |
|---|---|---:|---:|---:|---:|---:|
| `base` | `pick-place-v3` | 1.000 | 61.43 | 40.11 | 40.11 | 300 |
| `base` | `push-v3` | 0.973 | 153.97 | 55.88 | 43.71 | 300 |
| `careful` | `pick-place-v3` | 1.000 | 57.83 | 42.44 | 42.44 | 300 |
| `careful` | `push-v3` | 1.000 | 128.82 | 35.91 | 35.91 | 300 |
| `explore` | `pick-place-v3` | 0.947 | 53.50 | 66.61 | 42.20 | 300 |
| `explore` | `push-v3` | 0.960 | 138.52 | 59.47 | 41.12 | 300 |

### Σχήματα

![Push-PickPlace mean success](custom_push_pickplace/push_pickplace_figures/push_pickplace_mean_success.png)

![Push-PickPlace success by config](custom_push_pickplace/push_pickplace_figures/push_pickplace_success_by_config.png)

![Push-PickPlace return by config](custom_push_pickplace/push_pickplace_figures/push_pickplace_return_by_config.png)

---

# 3. Συγκεντρωτικοί πίνακες

## 3.1 Single-task αποτελέσματα

| Environment | Config | Test success |
|---|---|---:|
| `basketball-v3` | `A_basketball_main` | 0.800 |
| `basketball-v3` | `B_basketball_entropy` | 0.733 |
| `push-v3` | `base_push` | 0.933 |
| `push-v3` | `careful_push` | 1.000 |
| `push-v3` | `short_rollout_push` | 0.533 |
| `pick-place-v3` | `base_pick` | 0.733 |
| `pick-place-v3` | `careful_pick` | 0.933 |
| `pick-place-v3` | `short_rollout_pick` | 0.000 |
| `pick-place-v3` | `light_entropy_pick` | 1.000 |

## 3.2 Custom multi-task αποτελέσματα

| Custom MT | Config | Task 1 success | Task 2 success |
|---|---|---:|---:|
| `button-press-v3` + `push-v3` | `base` | 1.000 | 0.973 |
| `button-press-v3` + `push-v3` | `careful` | 1.000 | 0.980 |
| `button-press-v3` + `push-v3` | `explore` | 1.000 | 0.560 |
| `basketball-v3` + `pick-place-v3` | `base` | 0.960 | 1.000 |
| `basketball-v3` + `pick-place-v3` | `careful` | 1.000 | 1.000 |
| `basketball-v3` + `pick-place-v3` | `explore` | 0.900 | 1.000 |
| `push-v3` + `pick-place-v3` | `base` | 0.973 | 1.000 |
| `push-v3` + `pick-place-v3` | `careful` | 1.000 | 1.000 |
| `push-v3` + `pick-place-v3` | `explore` | 0.960 | 0.947 |


### Γενικές ρυθμίσεις εκπαίδευσης

| Παράμετρος | Τιμή / Περιγραφή |
|---|---|
| Αλγόριθμος | PPO |
| Policy | `MlpPolicy` |
| Περιβάλλοντα | Meta-World MT1 tasks |
| Reward function | `v2` |
| Μέγιστο μήκος επεισοδίου | `500` steps |
| Vectorized environments | `SubprocVecEnv` |
| Καταγραφή επεισοδίων | `VecMonitor` |
| Κανονικοποίηση | `VecNormalize`, όπου χρησιμοποιείται |
| TensorBoard | Χρησιμοποιείται για logging των training runs |
| Checkpoints | Αποθηκεύονται ανά συγκεκριμένο αριθμό timesteps |

### Single-task template configs

Οι single-task πειραματικές ομάδες `button-press-v3`, `push-v3` και `pick-place-v3` χρησιμοποιούν παρόμοια δομή PPO configurations. Τα ονόματα των configs διαφέρουν ανά task, π.χ. `base_push`, `base_pick`, `base_button`, αλλά οι βασικές ρυθμίσεις ακολουθούν την ίδια λογική.

| Config family | Learning rate | `n_steps` | Rollout size με 4 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_*` | `3e-4` | `1024` | `4096` | `256` | `10` | `0.99` | `0.95` | `0.20` | `0.0` | `0.5` | `0.5` |
| `careful_*` | `1e-4` | `1024` | `4096` | `512` | `15` | `0.995` | `0.95` | `0.15` | `0.0` | `0.7` | `0.5` |
| `short_rollout_*` | `3e-4` | `512` | `2048` | `256` | `10` | `0.99` | `0.95` | `0.20` | `0.0` | `0.5` | `0.5` |
| `light_entropy_*` | `2.5e-4` | `1024` | `4096` | `256` | `10` | `0.99` | `0.95` | `0.20` | `0.002` | `0.5` | `0.5` |

Στα single-task scripts χρησιμοποιούνται 4 parallel workers (`n_envs=4`) και, για τα split-based πειράματα, τρία 45/5 train/test splits με 50 συνολικά task variations ανά περιβάλλον.

### Basketball single-task configs

Το `basketball-v3` χρησιμοποιεί δύο ειδικές PPO configurations, οι οποίες εκπαιδεύονται για `6,000,000` timesteps με `4` parallel environments και τρία train seeds (`11`, `22`, `33`).

| Config | Learning rate | `n_steps` | Rollout size με 4 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_basketball_main` | `3e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.0` | `0.5` | `1.0` |
| `B_basketball_entropy` | `3e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.001` | `0.5` | `1.0` |

### Custom multi-task configs

Τα custom multi-task πειράματα χρησιμοποιούν τρεις κοινές PPO configurations: `base`, `careful` και `explore`. Κάθε custom environment επιλέγει ένα από τα δύο tasks στην αρχή κάθε επεισοδίου και προσθέτει one-hot task ID στην παρατήρηση, ώστε η πολιτική να γνωρίζει ποιο task είναι ενεργό.

| Config | Learning rate | `n_steps` | Rollout size με 8 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` | Network architecture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | `1e-4` | `2048` | `16384` | `1024` | `10` | `0.99` | `0.95` | `0.15` | `0.005` | `0.7` | `0.5` | `(256, 256)` |
| `careful` | `3e-5` | `2048` | `16384` | `1024` | `15` | `0.99` | `0.95` | `0.10` | `0.002` | `0.8` | `0.3` | `(256, 256)` |
| `explore` | `2e-4` | `2048` | `16384` | `1024` | `10` | `0.99` | `0.95` | `0.20` | `0.01` | `0.5` | `0.5` | `(256, 256)` |

Οι custom multi-task εκπαιδεύσεις χρησιμοποιούν `8` parallel environments και default training budget `10,000,000` timesteps. Στα training scripts αποθηκεύονται τόσο το τελικό PPO μοντέλο όσο και τα στατιστικά του `VecNormalize`, ώστε η αξιολόγηση να μπορεί να γίνει με τα ίδια normalization statistics.

### Config names ανά πείραμα

| Πείραμα | Config names |
|---|---|
| `button-press-v3` | `base_button`, `careful_button`, `short_rollout_button`, `light_entropy_button` |
| `basketball-v3` | `A_basketball_main`, `B_basketball_entropy` |
| `push-v3` | `base_push`, `careful_push`, `short_rollout_push`, `light_entropy_push` |
| `pick-place-v3` | `base_pick`, `careful_pick`, `short_rollout_pick`, `light_entropy_pick` |
| Custom `button-press-v3` + `push-v3` | `base`, `careful`, `explore` |
| Custom `basketball-v3` + `pick-place-v3` | `base`, `careful`, `explore` |
| Custom `push-v3` + `pick-place-v3` | `base`, `careful`, `explore` |
