# PPO Configurations

Το παρόν κεφάλαιο περιγράφει τα βασικά PPO configurations που χρησιμοποιήθηκαν στα πειράματα του repository. Στόχος είναι να γίνει ξεκάθαρο τι αντιπροσωπεύει κάθε config και γιατί χρησιμοποιήθηκαν διαφορετικές παραλλαγές ανάλογα με το task ή το multi-task setting.

---

## 1. Γενική PPO διάταξη

Όλα τα πειράματα βασίζονται στον αλγόριθμο **Proximal Policy Optimization (PPO)** με `MlpPolicy`.

Στα πειράματα χρησιμοποιούνται κυρίως:

| Παράμετρος | Περιγραφή |
|---|---|
| `learning_rate` | Ρυθμός μάθησης του optimizer |
| `n_steps` | Πόσα βήματα συλλέγει κάθε environment πριν γίνει update |
| `batch_size` | Μέγεθος mini-batch για τα PPO updates |
| `n_epochs` | Πόσες φορές περνά το PPO πάνω από το ίδιο rollout data |
| `gamma` | Discount factor για μελλοντικά rewards |
| `gae_lambda` | Παράμετρος Generalized Advantage Estimation |
| `clip_range` | Όριο clipping του PPO objective |
| `ent_coef` | Βάρος entropy bonus για exploration |
| `vf_coef` | Βάρος value function loss |
| `max_grad_norm` | Gradient clipping |
| `net_arch` | Μέγεθος hidden layers του policy/value network |

---

## 2. Single-task PPO configs

Τα single-task πειράματα χρησιμοποιούν διαφορετικές οικογένειες configs ανά task. Για τα περισσότερα environments (`button-press-v3`, `push-v3`, `pick-place-v3`) χρησιμοποιήθηκαν παρόμοια config families:

```text
base_*
careful_*
short_rollout_*
light_entropy_*
```

Το `*` αντικαθίσταται ανά task, π.χ.:

```text
base_push
careful_push
short_rollout_push
light_entropy_push
```

ή:

```text
base_pick
careful_pick
short_rollout_pick
light_entropy_pick
```

---

## 3. Single-task template configs

| Config family | Learning rate | `n_steps` | Rollout size με 4 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_*` | `3e-4` | `1024` | `4096` | `256` | `10` | `0.99` | `0.95` | `0.20` | `0.0` | `0.5` | `0.5` |
| `careful_*` | `1e-4` | `1024` | `4096` | `512` | `15` | `0.995` | `0.95` | `0.15` | `0.0` | `0.7` | `0.5` |
| `short_rollout_*` | `3e-4` | `512` | `2048` | `256` | `10` | `0.99` | `0.95` | `0.20` | `0.0` | `0.5` | `0.5` |
| `light_entropy_*` | `2.5e-4` | `1024` | `4096` | `256` | `10` | `0.99` | `0.95` | `0.20` | `0.002` | `0.5` | `0.5` |

---

## 4. Ερμηνεία των single-task configs

### `base_*`

Το `base_*` είναι το βασικό PPO configuration. Χρησιμοποιεί σχετικά τυπικό learning rate (`3e-4`), κανονικό rollout length και μέτριο clipping. Λειτουργεί ως reference config για να συγκριθούν οι πιο προσεκτικές ή πιο exploratory παραλλαγές.

Χρησιμοποιείται για να απαντήσει:

```text
Πόσο καλά μαθαίνεται το task με μια βασική PPO ρύθμιση;
```

---

### `careful_*`

Το `careful_*` είναι πιο συντηρητικό configuration. Έχει μικρότερο learning rate, μεγαλύτερο batch size, περισσότερα epochs, υψηλότερο `gamma` και μικρότερο `clip_range`.

Στόχος του είναι να κάνει πιο σταθερή τη μάθηση και να μειώσει απότομα ή ασταθή policy updates.

Χρησιμοποιείται όταν:

```text
Το task χρειάζεται σταθερότερη βελτιστοποίηση
ή όταν το base config έχει ασταθή συμπεριφορά.
```

Στα αποτελέσματα, το `careful` family ήταν συχνά το πιο σταθερό, ειδικά σε tasks όπως `push-v3` και στο single-task `basketball-v3`.

---

### `short_rollout_*`

Το `short_rollout_*` μειώνει το `n_steps`, άρα το PPO κάνει updates πιο συχνά με μικρότερα rollouts.

Αυτό μπορεί να είναι χρήσιμο όταν θέλουμε πιο συχνές ενημερώσεις, αλλά έχει και μειονέκτημα: το PPO βλέπει λιγότερο πλήρη trajectories πριν από κάθε update. Σε tasks που χρειάζονται πιο μακροπρόθεσμη συμπεριφορά, το short rollout μπορεί να είναι χειρότερο.

Στα αποτελέσματα, τα `short_rollout_*` configs συχνά ήταν πιο αδύναμα.

---

### `light_entropy_*`

Το `light_entropy_*` προσθέτει μικρό entropy bonus (`ent_coef`). Αυτό ενθαρρύνει το policy να εξερευνά περισσότερο αντί να γίνει γρήγορα deterministic.

Χρησιμοποιείται για να ελεγχθεί αν η επιπλέον exploration βοηθά σε tasks όπου το policy κολλάει σε κακή στρατηγική.

Στα single-task πειράματα, το entropy βοήθησε σε ορισμένα settings, όπως στο `pick-place-v3`, αλλά δεν ήταν πάντα καλύτερο.

---

## 5. Basketball single-task configs

Το `basketball-v3` χρησιμοποιεί ειδικά configs με μεγαλύτερα rollouts, επειδή το task είναι πιο σύνθετο και απαιτεί πιο μακροπρόθεσμη ακολουθία ενεργειών.

### Αρχικά basketball configs

| Config | Learning rate | `n_steps` | Rollout size με 4 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_basketball_main` | `3e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.0` | `0.5` | `1.0` |
| `B_basketball_entropy` | `3e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.001` | `0.5` | `1.0` |

### Νεότερα basketball split configs

| Config | Learning rate | `n_steps` | Rollout size με 4 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` | Network |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base_basketball` | `3e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.0` | `0.5` | `1.0` | `(256, 256)` |
| `careful_basketball` | `1e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.15` | `0.0` | `0.7` | `0.7` | `(256, 256)` |
| `short_rollout_basketball` | `3e-4` | `4096` | `16384` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.0` | `0.5` | `1.0` | `(256, 256)` |
| `light_entropy_basketball` | `3e-4` | `8192` | `32768` | `512` | `20` | `0.995` | `0.95` | `0.20` | `0.001` | `0.5` | `1.0` | `(256, 256)` |

Το `careful_basketball` ήταν το πιο αξιόπιστο basketball single-task configuration, φτάνοντας 100% train και test success στα διαθέσιμα split results.

---

## 6. Custom multi-task PPO configs

Τα custom multi-task πειράματα χρησιμοποιούν κοινές PPO configurations:

```text
base
careful
explore
```

Αυτές οι configurations χρησιμοποιούνται στα:

```text
custom-mt-pairs/
custom-mt-3-envs/
custom-mt-4-envs/
custom-mt-pairs-no-id/
```

Στα standard custom MT settings, το observation περιλαμβάνει:

```text
Meta-World observation + one-hot task ID
```

Στα no-task-ID custom mt's, το one-hot task ID αφαιρείται και το policy λαμβάνει μόνο το αρχικό Meta-World observation.

---

## 7. Custom MT config table

| Config | Learning rate | `n_steps` | Rollout size με 8 envs | Batch size | Epochs | `gamma` | `gae_lambda` | `clip_range` | `ent_coef` | `vf_coef` | `max_grad_norm` | Network architecture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | `1e-4` | `2048` | `16384` | `1024` | `10` | `0.99` | `0.95` | `0.15` | `0.005` | `0.7` | `0.5` | `(256, 256)` |
| `careful` | `3e-5` | `2048` | `16384` | `1024` | `15` | `0.99` | `0.95` | `0.10` | `0.002` | `0.8` | `0.3` | `(256, 256)` |
| `explore` | `2e-4` | `2048` | `16384` | `1024` | `10` | `0.99` | `0.95` | `0.20` | `0.01` | `0.5` | `0.5` | `(256, 256)` |

---

## 8. Ερμηνεία των custom MT configs

### `base`

Το `base` είναι το βασικό multi-task configuration. Έχει μέτριο learning rate, μέτριο entropy coefficient και σχετικά ισορροπημένο clipping.

Χρησιμοποιείται ως baseline για όλα τα custom MT experiments.

Σε ορισμένα pairs, όπως το `basketball-v3 + push-v3`, το `base` ήταν το καλύτερο configuration, επειδή κατάφερε να μάθει και τα δύο tasks.

---

### `careful`

Το `careful` είναι πιο συντηρητικό configuration. Έχει:

```text
χαμηλότερο learning rate
μικρότερο clip_range
περισσότερα epochs
μικρότερο max_grad_norm
μεγαλύτερο vf_coef
```

Αυτό το κάνει πιο σταθερό, αλλά ενδέχεται να το κάνει λιγότερο ικανό να εξερευνήσει δύσκολες στρατηγικές.

Στα all3 και all4 experiments, το `careful` ήταν συνήθως το πιο σταθερό για τα εύκολα/contact-based tasks, ειδικά για το `push-v3`.

---

### `explore`

Το `explore` έχει μεγαλύτερο entropy coefficient και μεγαλύτερο `clip_range`. Στόχος του είναι να ενισχύσει την exploration.

Ωστόσο, στα αποτελέσματα, το `explore` δεν έλυσε τα hard tasks στα all3/all4 settings και σε ορισμένες περιπτώσεις έκανε το `push-v3` χειρότερο.

Άρα, η βασική παρατήρηση είναι:

```text
Περισσότερo exploration δεν σημαίνει απαραίτητα καλύτερη απόδοση.
```

---

## 9. Config names ανά πείραμα

| Πείραμα | Config names |
|---|---|
| `button-press-v3` | `base_button`, `careful_button`, `short_rollout_button`, `light_entropy_button` |
| `basketball-v3` αρχικό | `A_basketball_main`, `B_basketball_entropy` |
| `basketball-v3` split/checkpoint | `base_basketball`, `careful_basketball`, `short_rollout_basketball`, `light_entropy_basketball` |
| `push-v3` | `base_push`, `careful_push`, `short_rollout_push`, `light_entropy_push` |
| `pick-place-v3` | `base_pick`, `careful_pick`, `short_rollout_pick`, `light_entropy_pick` |
| Custom two-task MT | `base`, `careful`, `explore` |
| Custom three-task MT | `base`, `careful`, `explore` |
| Custom four-task MT | `base`, `careful`, `explore` |
| No-task-ID ΜΤ's | `base`, `careful`, ανάλογα με το pair |

---

## 10. Σχέση configs με τα αποτελέσματα

Τα αποτελέσματα δείχνουν ότι η επιλογή PPO configuration παίζει σημαντικό ρόλο:

| Παρατήρηση | Ερμηνεία |
|---|---|
| Το `careful` συχνά σταθεροποιεί το `push-v3` | Η πιο συντηρητική ενημέρωση του policy βοηθά σταθερότερη μάθηση |
| Το `base` ήταν καλύτερο στο `basketball-v3 + push-v3` | Η πιο ισορροπημένη ρύθμιση βοήθησε το basketball σε pair setting |
| Το `explore` δεν έλυσε τα hard all3/all4 tasks | Επιπλέον exploration δεν αρκεί από μόνη του |
| Τα `short_rollout_*` configs ήταν συχνά πιο αδύναμα | Τα μικρότερα rollouts δεν δίνουν αρκετό trajectory context |
| Το `light_entropy_pick` πέτυχε πολύ καλά στο pick-place single-task | Μικρό exploration μπορεί να βοηθήσει σε συγκεκριμένα object-placement tasks |

---

## 11. Συμπεράσματα

Οι PPO configs σχεδιάστηκαν με γνώμονα τα παρακάτω:

```text
base      -> ισορροπημένη βασική ρύθμιση
careful   -> πιο σταθερή και συντηρητική μάθηση
explore   -> περισσότερο exploration
short     -> συχνότερα updates με μικρότερα rollouts
entropy   -> ελαφριά ενίσχυση εξερεύνησης
```

Συνολικά, τα αποτελέσματα δείχνουν ότι δεν υπάρχει ένα configuration που να είναι καλύτερο παντού. Το κατάλληλο config εξαρτάται από το task, τον αριθμό των tasks και το αν το policy πρέπει να μάθει μία ή περισσότερες συμπεριφορές ταυτόχρονα.
