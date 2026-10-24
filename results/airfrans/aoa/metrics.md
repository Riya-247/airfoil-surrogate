# Metrics (airfrans, task=aoa, features=physics)

High angle of attack means alpha >= 10 deg. 1 drag count = 1e-4 in Cd.
cv = nested 5-fold CV on the training set (grouped by geometry); test = held-out test set.

| model | split | band | n | MAE_Cl | MAE_Cd_counts | MAPE_Cd_pct | spearman_Cl | spearman_Cd | spearman_LD |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | cv | all | 804 | 0.01809 | 4.539 | 3.519 | 0.9987 | 0.9865 | 0.984 |
| random_forest | cv | alpha<10 | 687 | 0.01732 | 3.648 | 3.252 | 0.9984 | 0.9805 | 0.9918 |
| random_forest | cv | alpha>=10 | 117 | 0.02265 | 9.774 | 5.088 | 0.9881 | 0.9065 | 0.8195 |
| random_forest | test | all | 196 | 0.08043 | 21.54 | 9.624 | 0.9728 | 0.9276 | 0.931 |
| random_forest | test | alpha<10 | 111 | 0.1076 | 10.69 | 6.521 | 0.8604 | 0.8778 | 0.888 |
| random_forest | test | alpha>=10 | 85 | 0.04491 | 35.71 | 13.68 | 0.977 | 0.7976 | 0.4154 |
| svr | cv | all | 804 | 0.00503 | 0.9359 | 0.6484 | 0.9999 | 0.9994 | 0.9978 |
| svr | cv | alpha<10 | 687 | 0.004866 | 0.5445 | 0.4818 | 0.9999 | 0.9993 | 0.9995 |
| svr | cv | alpha>=10 | 117 | 0.005992 | 3.234 | 1.626 | 0.9986 | 0.9773 | 0.9675 |
| svr | test | all | 196 | 0.01711 | 8.948 | 3.392 | 0.9991 | 0.9538 | 0.9898 |
| svr | test | alpha<10 | 111 | 0.02138 | 6.979 | 2.763 | 0.9968 | 0.9367 | 0.9802 |
| svr | test | alpha>=10 | 85 | 0.01154 | 11.52 | 4.214 | 0.996 | 0.9671 | 0.9219 |
| mlp | cv | all | 804 | 0.009519 | 1.313 | 0.9628 | 0.9996 | 0.9988 | 0.9964 |
| mlp | cv | alpha<10 | 687 | 0.009121 | 0.8728 | 0.7819 | 0.9996 | 0.9985 | 0.9987 |
| mlp | cv | alpha>=10 | 117 | 0.01186 | 3.901 | 2.025 | 0.9941 | 0.9706 | 0.9426 |
| mlp | test | all | 196 | 0.02117 | 11.9 | 5.708 | 0.9981 | 0.9396 | 0.9827 |
| mlp | test | alpha<10 | 111 | 0.01609 | 9.975 | 5.91 | 0.9958 | 0.8863 | 0.9753 |
| mlp | test | alpha>=10 | 85 | 0.02779 | 14.41 | 5.445 | 0.9858 | 0.9525 | 0.8457 |
