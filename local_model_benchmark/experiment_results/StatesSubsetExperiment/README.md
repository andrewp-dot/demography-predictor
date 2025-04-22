
# StatesSubsetExperiment

**Description:** Trains the model from the simpliest to more complex ones by hidden size for each group and reveals which models performs best for each group.

## Overall metrics for high_income  - model comparision
```
        mae         mse      rmse            r2     model  rank
4  3.208628   48.993009  3.526688 -1.204184e+29  lstm-512   6.0
3  3.475072   51.444550  3.786619 -1.464280e+28  lstm-256   7.0
2  3.747586   54.826392  4.033260 -2.488191e+28  lstm-128  11.0
1  4.216498   66.620384  4.511270 -2.116507e+29   lstm-64  16.0
0  6.041048  115.544323  6.331558 -1.639553e+31   lstm-32  20.0
```


## Overall metrics for upper_middle_income  - model comparision
```
        mae         mse      rmse            r2     model  rank
4  2.948628   28.008357  3.222059 -6.836222e+28  lstm-512   4.0
3  3.377800   34.121168  3.645507 -2.307643e+29  lstm-256   9.0
2  3.978878   47.514080  4.241138 -9.066878e+29  lstm-128  13.0
1  4.731716   68.011368  4.975916 -1.352827e+29   lstm-64  14.0
0  6.335627  119.788492  6.549717 -5.206309e+30   lstm-32  20.0
```


## Overall metrics for lower_middle_income  - model comparision
```
        mae         mse      rmse            r2     model  rank
4  3.028267   26.471899  3.239002 -6.740903e+30  lstm-512   4.0
2  3.168595   27.706541  3.352137 -2.670589e+31  lstm-128   9.0
3  3.378095   35.357775  3.603636 -7.732187e+30  lstm-256  11.0
1  5.007392   70.218024  5.159106 -1.403800e+32   lstm-64  16.0
0  6.136428  112.657362  6.305171 -2.062176e+32   lstm-32  20.0
```


## Overall metrics for low_income  - model comparision
```
        mae        mse      rmse            r2     model  rank
2  3.784109  33.330007  4.023211 -2.232171e+28  lstm-128   4.0
3  4.323531  41.783312  4.641048 -8.282222e+28  lstm-256  10.0
4  4.483426  41.105210  4.836267 -9.748775e+28  lstm-512  12.0
1  5.011485  60.358993  5.199066 -2.611070e+28   lstm-64  14.0
0  6.384217  98.786974  6.583495 -1.494733e+29   lstm-32  20.0
```


## Overall metrics for europe  - model comparision
```
        mae        mse      rmse            r2     model  rank
3  3.389677  29.356343  3.583705 -7.349930e+27  lstm-256   4.0
4  3.425747  31.447943  3.641811 -1.672447e+28  lstm-512   8.0
2  4.188579  43.923712  4.399712 -2.384225e+28  lstm-128  12.0
1  5.196788  81.409364  5.394537 -5.805245e+29   lstm-64  17.0
0  5.259219  85.349380  5.454991 -5.641747e+29   lstm-32  19.0
```


## Overall metrics for asia  - model comparision
```
        mae         mse      rmse            r2     model  rank
4  4.319264   53.602938  4.648829 -2.952857e+30  lstm-512   6.0
3  4.332198   59.612898  4.656543 -1.578725e+28  lstm-256   7.0
2  4.970391   66.697394  5.295712 -3.503972e+30  lstm-128  13.0
1  5.774062   99.674045  6.116154 -1.195749e+30   lstm-64  14.0
0  7.844520  189.600834  8.177164 -3.769083e+31   lstm-32  20.0
```


## Overall metrics for africa  - model comparision
```
        mae         mse      rmse            r2     model  rank
3  3.790645   41.297618  4.042630 -1.080997e+31  lstm-256   6.0
2  3.830926   42.900900  4.067076 -1.862651e+31  lstm-128  10.0
4  3.893849   47.936660  4.197540 -4.187171e+30  lstm-512  10.0
1  5.057146   77.440261  5.322473 -8.543017e+30   lstm-64  14.0
0  7.293180  157.367200  7.554635 -1.395232e+32   lstm-32  20.0
```


## Overall metrics for north_america  - model comparision
```
        mae         mse      rmse            r2     model  rank
4  4.649938   56.153586  4.919844 -4.898905e+28  lstm-512   8.0
2  5.394687   91.823215  5.616213 -2.771177e+26  lstm-128  10.0
3  4.726234   65.692148  4.980298 -4.755413e+28  lstm-256  10.0
1  6.002542  112.354739  6.196405 -7.475160e+27   lstm-64  14.0
0  6.045709  114.210130  6.316529 -2.964109e+28   lstm-32  18.0
```


## Overall metrics for south_america  - model comparision
```
        mae         mse      rmse            r2     model  rank
3  4.458522   62.506561  4.732655 -1.267493e+28  lstm-256   5.0
4  4.417679   68.465692  4.740030 -9.741231e+28  lstm-512   7.0
2  5.243245  100.914788  5.487342 -1.153769e+29  lstm-128  14.0
1  5.218197  100.996685  5.474872 -1.964592e+29   lstm-64  15.0
0  5.268228  104.212413  5.507369 -1.299664e+29   lstm-32  19.0
```


## Overall metrics for oceania  - model comparision
```
        mae         mse      rmse            r2     model  rank
4  3.182807   32.210580  3.337407 -4.467562e+28  lstm-512   4.0
3  5.649949  105.249100  5.816634 -5.039368e+29  lstm-256   8.0
2  5.963166  120.113018  6.120375 -6.312321e+29  lstm-128  13.0
1  6.275697  134.951350  6.429549 -8.242628e+29   lstm-64  17.0
0  6.483714  144.002628  6.645801 -5.328080e+29   lstm-32  18.0
```


