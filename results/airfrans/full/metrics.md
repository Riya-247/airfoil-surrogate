# Metrics (airfrans, task=full, features=physics)

High angle of attack means alpha >= 10 deg. 1 drag count = 1e-4 in Cd.
cv = nested 5-fold CV on the training set (grouped by geometry); test = held-out test set.

| model | split | band | n | MAE_Cl | MAE_Cd_counts | MAPE_Cd_pct | spearman_Cl | spearman_Cd | spearman_LD |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | cv | all | 800 | 0.01901 | 6.847 | 4.592 | 0.9992 | 0.9678 | 0.9821 |
| random_forest | cv | alpha<10 | 636 | 0.01626 | 5.225 | 4.266 | 0.9991 | 0.9465 | 0.9937 |
| random_forest | cv | alpha>=10 | 164 | 0.02968 | 13.13 | 5.857 | 0.9821 | 0.9371 | 0.7936 |
| random_forest | test | all | 200 | 0.02086 | 6.194 | 4.493 | 0.999 | 0.9755 | 0.9829 |
| random_forest | test | alpha<10 | 162 | 0.01752 | 4.687 | 4.178 | 0.9989 | 0.9563 | 0.9919 |
| random_forest | test | alpha>=10 | 38 | 0.03507 | 12.62 | 5.839 | 0.9867 | 0.9501 | 0.8431 |
| svr | cv | all | 800 | 0.006557 | 2.367 | 1.129 | 0.9999 | 0.9895 | 0.9977 |
| svr | cv | alpha<10 | 636 | 0.006168 | 1.656 | 0.8711 | 0.9999 | 0.9879 | 0.9993 |
| svr | cv | alpha>=10 | 164 | 0.008067 | 5.127 | 2.127 | 0.9981 | 0.9886 | 0.9735 |
| svr | test | all | 200 | 0.006404 | 1.603 | 0.9794 | 0.9998 | 0.9992 | 0.9979 |
| svr | test | alpha<10 | 162 | 0.006283 | 0.764 | 0.6835 | 0.9997 | 0.9988 | 0.9994 |
| svr | test | alpha>=10 | 38 | 0.006919 | 5.182 | 2.241 | 0.9991 | 0.9788 | 0.9565 |
| mlp | cv | all | 800 | 0.0114 | 3.254 | 1.915 | 0.9996 | 0.9866 | 0.9958 |
| mlp | cv | alpha<10 | 636 | 0.01084 | 2.462 | 1.698 | 0.9995 | 0.9808 | 0.9985 |
| mlp | cv | alpha>=10 | 164 | 0.01356 | 6.326 | 2.757 | 0.9949 | 0.9839 | 0.952 |
| mlp | test | all | 200 | 0.018 | 2.97 | 2.135 | 0.999 | 0.9954 | 0.9932 |
| mlp | test | alpha<10 | 162 | 0.01689 | 1.986 | 1.858 | 0.9989 | 0.9917 | 0.9963 |
| mlp | test | alpha>=10 | 38 | 0.02271 | 7.168 | 3.317 | 0.9862 | 0.9807 | 0.9269 |
