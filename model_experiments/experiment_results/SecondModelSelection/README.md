
# SecondModelSelection

**Description:** test

## Per target metrics - model comparision
```
                            target        mae         mse       rmse            r2                                              state       model     rank
2845          population ages 0-14   0.035431    0.001740   0.041717      0.995274                                        Afghanistan          rf    132.0
1             population ages 0-14   0.073938    0.007167   0.084656      0.980538                                        Afghanistan  simple-rnn    477.0
1423          population ages 0-14   0.157606    0.026750   0.163555      0.927355                                        Afghanistan    base-gru   1380.0
712           population ages 0-14   0.172045    0.032433   0.180092      0.911922                                        Afghanistan   base-lstm   1582.0
2134          population ages 0-14   0.458946    0.337883   0.581277      0.082419                                        Afghanistan     xgboost   4883.0
2844         population ages 15-64   0.057429    0.004887   0.069908      0.986615                                        Afghanistan          rf    339.0
2133         population ages 15-64   0.114531    0.032415   0.180042      0.911218                                        Afghanistan     xgboost   1389.0
1422         population ages 15-64   0.510730    0.271945   0.521483      0.255164                                        Afghanistan    base-gru   4688.0
711          population ages 15-64   0.546956    0.309014   0.555891      0.153634                                        Afghanistan   base-lstm   4910.0
0            population ages 15-64   0.571261    0.332902   0.576977      0.088206                                        Afghanistan  simple-rnn   5072.0
2135  population ages 65 and above   0.007011    0.000080   0.008949     -0.588075                                        Afghanistan     xgboost   1427.0
2846  population ages 65 and above   0.008287    0.000196   0.013997     -2.884472                                        Afghanistan          rf   1997.0
2     population ages 65 and above   0.339821    0.121874   0.349104  -2415.580407                                        Afghanistan  simple-rnn   6104.0
713   population ages 65 and above   0.375785    0.146880   0.383249  -2911.423893                                        Afghanistan   base-lstm   6375.0
1424  population ages 65 and above   0.380754    0.152771   0.390859  -3028.225808                                        Afghanistan    base-gru   6425.0
2137          population ages 0-14   0.113319    0.016929   0.130113      0.960265                        Africa Eastern and Southern     xgboost    935.0
2848          population ages 0-14   0.244446    0.085362   0.292167      0.799647                        Africa Eastern and Southern          rf   2598.0
1426          population ages 0-14   0.886661    0.906436   0.952070     -1.127509                        Africa Eastern and Southern    base-gru   7162.0
715           population ages 0-14   0.966784    1.079906   1.039185     -1.534662                        Africa Eastern and Southern   base-lstm   7583.0
4             population ages 0-14   1.220427    1.699218   1.303541     -2.988257                        Africa Eastern and Southern  simple-rnn   8748.0
2136         population ages 15-64   0.146503    0.031967   0.178794      0.895617                        Africa Eastern and Southern     xgboost   1551.0
2847         population ages 15-64   0.155484    0.036199   0.190261      0.881799                        Africa Eastern and Southern          rf   1666.0
1425         population ages 15-64   0.366444    0.161434   0.401789      0.472869                        Africa Eastern and Southern    base-gru   3791.0
714          population ages 15-64   0.510424    0.316765   0.562819     -0.034333                        Africa Eastern and Southern   base-lstm   4968.0
3            population ages 15-64   0.723429    0.615749   0.784697     -1.010604                        Africa Eastern and Southern  simple-rnn   6430.0
2138  population ages 65 and above   0.040132    0.002005   0.044783      0.800831                        Africa Eastern and Southern     xgboost    697.0
2849  population ages 65 and above   0.055142    0.004515   0.067190      0.551651                        Africa Eastern and Southern          rf   1088.0
5     population ages 65 and above   0.393967    0.175793   0.419277    -16.458345                        Africa Eastern and Southern  simple-rnn   5724.0
716   population ages 65 and above   0.420036    0.194135   0.440607    -18.279909                        Africa Eastern and Southern   base-lstm   5903.0
1427  population ages 65 and above   0.504623    0.283352   0.532308    -27.140246                        Africa Eastern and Southern    base-gru   6558.0
2851          population ages 0-14   0.099465    0.014951   0.122275      0.896752                         Africa Western and Central          rf   1046.0
2140          population ages 0-14   0.105852    0.017755   0.133248      0.877388                         Africa Western and Central     xgboost   1168.0
1429          population ages 0-14   0.590512    0.378064   0.614869     -1.610816                         Africa Western and Central    base-gru   5847.0
718           population ages 0-14   0.764490    0.642115   0.801321     -3.434290                         Africa Western and Central   base-lstm   7074.0
7             population ages 0-14   1.051989    1.212572   1.101169     -7.373730                         Africa Western and Central  simple-rnn   8530.0
2139         population ages 15-64   0.064045    0.007391   0.085969      0.955097                         Africa Western and Central     xgboost    545.0
1428         population ages 15-64   0.102048    0.017239   0.131299      0.895260                         Africa Western and Central    base-gru   1115.0
2850         population ages 15-64   0.108134    0.031664   0.177945      0.807618                         Africa Western and Central          rf   1576.0
717          population ages 15-64   0.187093    0.047317   0.217526      0.712517                         Africa Western and Central   base-lstm   2169.0
6            population ages 15-64   0.430264    0.207457   0.455475     -0.260436                         Africa Western and Central  simple-rnn   4543.0
2852  population ages 65 and above   0.046537    0.003284   0.057302     -1.509033                         Africa Western and Central          rf   1901.0
2141  population ages 65 and above   0.069586    0.006381   0.079882     -3.875959                         Africa Western and Central     xgboost   2489.0
8     population ages 65 and above   0.494288    0.282143   0.531172   -214.593013                         Africa Western and Central  simple-rnn   6947.0
719   population ages 65 and above   0.540560    0.330824   0.575173   -251.791279                         Africa Western and Central   base-lstm   7209.0
1430  population ages 65 and above   0.584934    0.387810   0.622744   -295.335324                         Africa Western and Central    base-gru   7456.0
2143          population ages 0-14   0.237359    0.072458   0.269181      0.913750                                            Albania     xgboost   2212.0
2854          population ages 0-14   0.411190    0.225986   0.475380      0.730998                                            Albania          rf   3932.0
1432          population ages 0-14   0.903464    0.994146   0.997069     -0.183378                                            Albania    base-gru   6968.0
721           population ages 0-14   1.208619    1.756147   1.325197     -1.090423                                            Albania   base-lstm   8362.0
10            population ages 0-14   1.282791    1.939416   1.392629     -1.308577                                            Albania  simple-rnn   8630.0
2142         population ages 15-64   0.398998    0.243579   0.493538     -2.348098                                            Albania     xgboost   5214.0
2853         population ages 15-64   0.460523    0.313382   0.559805     -3.307565                                            Albania          rf   5743.0
1431         population ages 15-64   0.531827    0.472147   0.687129     -5.489851                                            Albania    base-gru   6504.0
9            population ages 15-64   0.724288    0.862950   0.928951    -10.861603                                            Albania  simple-rnn   7846.0
720          population ages 15-64   1.290117    2.265701   1.505224    -30.142976                                            Albania   base-lstm  10062.0
1433  population ages 65 and above   0.105446    0.020052   0.141604      0.984986                                            Albania    base-gru    872.0
11    population ages 65 and above   0.149950    0.038471   0.196141      0.971194                                            Albania  simple-rnn   1423.0
722   population ages 65 and above   0.326581    0.137822   0.371243      0.896804                                            Albania   base-lstm   3037.0
2144  population ages 65 and above   0.634533    0.681701   0.825652      0.489566                                            Albania     xgboost   5761.0
2855  population ages 65 and above   0.641572    0.690073   0.830707      0.483297                                            Albania          rf   5797.0
2146          population ages 0-14   1.448028    3.486015   1.867087     -1.496667                                            Algeria     xgboost   9463.0
2857          population ages 0-14   2.372773    8.789788   2.964758     -5.295202                                            Algeria          rf  11438.0
724           population ages 0-14   4.051850   21.092686   4.592677    -14.106477                                            Algeria   base-lstm  12978.0
1435          population ages 0-14   4.162704   21.813238   4.670464    -14.622533                                            Algeria    base-gru  13018.0
13            population ages 0-14   4.359689   24.026653   4.901699    -16.207770                                            Algeria  simple-rnn  13101.0
2145         population ages 15-64   0.992098    1.511897   1.229592      0.434337                                            Algeria     xgboost   7289.0
2856         population ages 15-64   2.230213    7.404736   2.721164     -1.770418                                            Algeria          rf  10766.0
1434         population ages 15-64   2.816323   10.461643   3.234446     -2.914133                                            Algeria    base-gru  11477.0
723          population ages 15-64   3.164705   13.423107   3.663756     -4.022139                                            Algeria   base-lstm  12017.0
12           population ages 15-64   3.204177   13.503136   3.674661     -4.052082                                            Algeria  simple-rnn  12042.0
2858  population ages 65 and above   0.077218    0.007656   0.087499      0.964024                                            Algeria          rf    565.0
2147  population ages 65 and above   0.161138    0.032527   0.180353      0.847155                                            Algeria     xgboost   1691.0
725   population ages 65 and above   1.409043    2.080647   1.442445     -8.776949                                            Algeria   base-lstm   9622.0
14    population ages 65 and above   1.622708    2.799190   1.673078    -12.153377                                            Algeria  simple-rnn  10225.0
1436  population ages 65 and above   1.665805    2.938772   1.714285    -12.809273                                            Algeria    base-gru  10340.0
2860          population ages 0-14   0.164488    0.051391   0.226696     -0.220407                                             Angola          rf   2755.0
2149          population ages 0-14   0.210049    0.083050   0.288184     -0.972213                                             Angola     xgboost   3467.0
1438          population ages 0-14   0.980316    1.099197   1.048426    -25.103035                                             Angola    base-gru   8771.0
727           population ages 0-14   1.142051    1.474639   1.214347    -34.018784                                             Angola   base-lstm   9437.0
16            population ages 0-14   1.343660    2.028174   1.424140    -47.163767                                             Angola  simple-rnn  10102.0
2859         population ages 15-64   0.067236    0.005920   0.076939      0.837475                                             Angola          rf    800.0
2148         population ages 15-64   0.275044    0.124395   0.352697     -2.415285                                             Angola     xgboost   4339.0
1437         population ages 15-64   0.715758    0.587683   0.766605    -15.134914                                             Angola    base-gru   7476.0
726          population ages 15-64   0.902837    0.919127   0.958711    -24.234748                                             Angola   base-lstm   8456.0
15           population ages 15-64   1.066616    1.273178   1.128352    -33.955289                                             Angola  simple-rnn   9154.0
2861  population ages 65 and above   0.027715    0.001151   0.033931     -2.794348                                             Angola          rf   2046.0
2150  population ages 65 and above   0.035120    0.001548   0.039348     -4.102638                                             Angola     xgboost   2260.0
17    population ages 65 and above   0.266687    0.077709   0.278763   -255.108394                                             Angola  simple-rnn   5364.0
728   population ages 65 and above   0.321927    0.109965   0.331610   -361.417174                                             Angola   base-lstm   5825.0
1439  population ages 65 and above   0.331439    0.118011   0.343527   -387.931511                                             Angola    base-gru   5909.0
2152          population ages 0-14   0.094486    0.013891   0.117859      0.988427                                Antigua and Barbuda     xgboost    688.0
2863          population ages 0-14   0.276418    0.101002   0.317807      0.915855                                Antigua and Barbuda          rf   2566.0
1441          population ages 0-14   1.332597    1.996983   1.413147     -0.663703                                Antigua and Barbuda    base-gru   8487.0
19            population ages 0-14   1.444203    2.370618   1.539681     -0.974981                                Antigua and Barbuda  simple-rnn   8874.0
730           population ages 0-14   1.480465    2.518654   1.587027     -1.098311                                Antigua and Barbuda   base-lstm   9030.0
2862         population ages 15-64   0.299460    0.177388   0.421175     -1.189235                                Antigua and Barbuda          rf   4439.0
2151         population ages 15-64   0.460708    0.310401   0.557136     -2.830814                                Antigua and Barbuda     xgboost   5647.0
1440         population ages 15-64   0.659699    0.640548   0.800342     -6.905325                                Antigua and Barbuda    base-gru   7221.0
18           population ages 15-64   0.666968    0.661939   0.813596     -7.169322                                Antigua and Barbuda  simple-rnn   7296.0
729          population ages 15-64   1.142938    1.821018   1.349451    -21.474100                                Antigua and Barbuda   base-lstm   9554.0
2864  population ages 65 and above   0.621263    0.652252   0.807621      0.111856                                Antigua and Barbuda          rf   5911.0
2153  population ages 65 and above   0.627270    0.730296   0.854574      0.005586                                Antigua and Barbuda     xgboost   6139.0
731   population ages 65 and above   1.098984    1.211668   1.100758     -0.649878                                Antigua and Barbuda   base-lstm   7606.0
20    population ages 65 and above   1.246270    1.563883   1.250553     -1.129474                                Antigua and Barbuda  simple-rnn   8290.0
1442  population ages 65 and above   1.262167    1.598536   1.264332     -1.176660                                Antigua and Barbuda    base-gru   8341.0
2155          population ages 0-14   0.157271    0.038319   0.195752      0.392972                                         Arab World     xgboost   2262.0
2866          population ages 0-14   0.764498    0.819777   0.905415    -11.986462                                         Arab World          rf   7869.0
733           population ages 0-14   2.581262    7.618727   2.760204   -119.691791                                         Arab World   base-lstm  12377.0
1444          population ages 0-14   2.640459    7.916869   2.813693   -124.414805                                         Arab World    base-gru  12429.0
22            population ages 0-14   2.826669    9.101129   3.016808   -143.175212                                         Arab World  simple-rnn  12621.0
2154         population ages 15-64   0.290197    0.131485   0.362608    -16.075530                                         Arab World     xgboost   5204.0
2865         population ages 15-64   1.240800    2.134239   1.460904   -276.167035                                         Arab World          rf  10373.0
1443         population ages 15-64   1.684809    3.363450   1.833971   -435.800821                                         Arab World    base-gru  11316.0
732          population ages 15-64   1.861874    4.188963   2.046696   -543.007601                                         Arab World   base-lstm  11707.0
21           population ages 15-64   2.022619    4.835732   2.199030   -627.001406                                         Arab World  simple-rnn  11978.0
2867  population ages 65 and above   0.193821    0.051142   0.226145      0.223860                                         Arab World          rf   2605.0
2156  population ages 65 and above   0.242419    0.089485   0.299140     -0.358048                                         Arab World     xgboost   3414.0
734   population ages 65 and above   0.893545    0.831795   0.912028    -11.623574                                         Arab World   base-lstm   8058.0
23    population ages 65 and above   0.986916    1.027363   1.013589    -14.591575                                         Arab World  simple-rnn   8496.0
1445  population ages 65 and above   1.089397    1.246252   1.116356    -17.913493                                         Arab World    base-gru   8920.0
2869          population ages 0-14   0.225880    0.063923   0.252829      0.883023                                          Argentina          rf   2200.0
2158          population ages 0-14   0.208995    0.085666   0.292688      0.843233                                          Argentina     xgboost   2397.0
1447          population ages 0-14   3.152286   10.836540   3.291890    -18.830566                                          Argentina    base-gru  12401.0
25            population ages 0-14   3.549444   13.713511   3.703176    -24.095342                                          Argentina  simple-rnn  12859.0
736           population ages 0-14   3.609776   14.268495   3.777366    -25.110947                                          Argentina   base-lstm  12913.0
2157         population ages 15-64   0.094798    0.012152   0.110236      0.862455                                          Argentina     xgboost   1013.0
1446         population ages 15-64   0.213071    0.053617   0.231554      0.393124                                          Argentina    base-gru   2615.0
2868         population ages 15-64   0.600495    0.521657   0.722258     -4.904463                                          Argentina          rf   6704.0
24           population ages 15-64   0.809470    0.789947   0.888790     -7.941149                                          Argentina  simple-rnn   7750.0
735          population ages 15-64   1.067991    1.373195   1.171834    -14.542731                                          Argentina   base-lstm   8966.0
2870  population ages 65 and above   0.048747    0.004871   0.069792      0.975601                                          Argentina          rf    364.0
2159  population ages 65 and above   0.136750    0.049657   0.222839      0.751264                                          Argentina     xgboost   2004.0
26    population ages 65 and above   1.735142    3.463351   1.861008    -16.348169                                          Argentina  simple-rnn  10658.0
737   population ages 65 and above   1.756151    3.544319   1.882636    -16.753742                                          Argentina   base-lstm  10713.0
1448  population ages 65 and above   2.033244    4.708019   2.169797    -22.582799                                          Argentina    base-gru  11316.0
2161          population ages 0-14   0.232020    0.075791   0.275301     -1.123913                                            Armenia     xgboost   3513.0
2872          population ages 0-14   0.390629    0.185878   0.431135     -4.208914                                            Armenia          rf   5243.0
1450          population ages 0-14   4.268103   18.708775   4.325364   -523.282372                                            Armenia    base-gru  13742.0
739           population ages 0-14   4.428318   20.229063   4.497673   -565.885934                                            Armenia   base-lstm  13787.0
28            population ages 0-14   4.521139   21.035230   4.586418   -588.477421                                            Armenia  simple-rnn  13826.0
2160         population ages 15-64   0.343549    0.177709   0.421555      0.734588                                            Armenia     xgboost   3582.0
2871         population ages 15-64   0.507049    0.344493   0.586935      0.485494                                            Armenia          rf   4747.0
1449         population ages 15-64   2.648915    7.817674   2.796010    -10.675849                                            Armenia    base-gru  11732.0
27           population ages 15-64   2.898934    9.398144   3.065639    -13.036311                                            Armenia  simple-rnn  12040.0
738          population ages 15-64   3.319952   12.277264   3.503893    -17.336333                                            Armenia   base-lstm  12552.0
2162  population ages 65 and above   0.069273    0.008600   0.092736      0.978730                                            Armenia     xgboost    512.0
2873  population ages 65 and above   0.369147    0.209534   0.457748      0.481773                                            Armenia          rf   4014.0
740   population ages 65 and above   1.260057    1.627810   1.275857     -3.025968                                            Armenia   base-lstm   8746.0
29    population ages 65 and above   1.344402    1.839073   1.356124     -3.548472                                            Armenia  simple-rnn   9056.0
1451  population ages 65 and above   1.507259    2.307335   1.518991     -4.706596                                            Armenia    base-gru   9558.0
1453          population ages 0-14   0.139064    0.028252   0.168083      0.933782                                              Aruba    base-gru   1333.0
2875          population ages 0-14   0.218191    0.095926   0.309719      0.775166                                              Aruba          rf   2658.0
31            population ages 0-14   0.408444    0.206136   0.454022      0.516851                                              Aruba  simple-rnn   4050.0
742           population ages 0-14   0.437274    0.249940   0.499940      0.414182                                              Aruba   base-lstm   4353.0
2164          population ages 0-14   0.498003    0.424018   0.651167      0.006173                                              Aruba     xgboost   5205.0
2874         population ages 15-64   0.218024    0.063201   0.251398      0.829973                                              Aruba          rf   2271.0
2163         population ages 15-64   0.352097    0.194715   0.441265      0.476166                                              Aruba     xgboost   3916.0
1452         population ages 15-64   1.174165    1.635778   1.278975     -3.400669                                              Aruba    base-gru   8734.0
30           population ages 15-64   1.471437    2.476410   1.573661     -5.662188                                              Aruba  simple-rnn   9700.0
741          population ages 15-64   1.932393    4.213053   2.052572    -10.334209                                              Aruba   base-lstm  10859.0
2165  population ages 65 and above   0.545782    0.506293   0.711542      0.674243                                              Aruba     xgboost   5062.0
1454  population ages 65 and above   1.044243    1.344089   1.159348      0.135192                                              Aruba    base-gru   7334.0
32    population ages 65 and above   1.104043    1.470229   1.212530      0.054031                                              Aruba  simple-rnn   7581.0
743   population ages 65 and above   1.181641    1.662321   1.289310     -0.069564                                              Aruba   base-lstm   7890.0
2876  population ages 65 and above   1.146366    1.935031   1.391054     -0.245030                                              Aruba          rf   8115.0
2167          population ages 0-14   0.235797    0.071129   0.266701     -0.719119                                          Australia     xgboost   3352.0
2878          population ages 0-14   0.561490    0.476590   0.690355    -10.518682                                          Australia          rf   6838.0
1456          population ages 0-14   2.938269    9.503842   3.082830   -228.698124                                          Australia    base-gru  12766.0
745           population ages 0-14   3.438659   13.039486   3.611023   -314.150994                                          Australia   base-lstm  13277.0
34            population ages 0-14   3.527344   13.684819   3.699300   -329.748049                                          Australia  simple-rnn  13362.0
1455         population ages 15-64   0.582623    0.454167   0.673919      0.299467                                          Australia    base-gru   5286.0
2166         population ages 15-64   0.825669    0.992458   0.996222     -0.530826                                          Australia     xgboost   6995.0
2877         population ages 15-64   1.149974    1.889511   1.374595     -1.914493                                          Australia          rf   8619.0
33           population ages 15-64   1.242730    2.202808   1.484186     -2.397740                                          Australia  simple-rnn   8979.0
744          population ages 15-64   1.635246    3.602664   1.898068     -4.556960                                          Australia   base-lstm  10137.0
2168  population ages 65 and above   0.185246    0.053948   0.232267      0.945130                                          Australia     xgboost   1793.0
2879  population ages 65 and above   0.439236    0.257018   0.506969      0.738590                                          Australia          rf   4120.0
746   population ages 65 and above   1.381738    1.955749   1.398481     -0.989172                                          Australia   base-lstm   8600.0
35    population ages 65 and above   1.388573    1.980989   1.407476     -1.014843                                          Australia  simple-rnn   8633.0
1457  population ages 65 and above   1.751950    3.169741   1.780377     -2.223911                                          Australia    base-gru   9759.0
2170          population ages 0-14   0.091403    0.011176   0.105716      0.733786                                            Austria     xgboost   1207.0
2881          population ages 0-14   0.526357    0.350952   0.592412     -7.359791                                            Austria          rf   6329.0
1459          population ages 0-14   2.333794    6.033004   2.456217   -142.708167                                            Austria    base-gru  12069.0
748           population ages 0-14   2.875474    9.091236   3.015168   -215.556271                                            Austria   base-lstm  12680.0
37            population ages 0-14   3.207297   11.269927   3.357071   -267.453426                                            Austria  simple-rnn  13045.0
2880         population ages 15-64   0.294205    0.105745   0.325184      0.475436                                            Austria          rf   3245.0
1458         population ages 15-64   0.612431    0.482538   0.694650     -1.393707                                            Austria    base-gru   6097.0
2169         population ages 15-64   1.043721    1.521499   1.233491     -6.547635                                            Austria     xgboost   8792.0
36           population ages 15-64   1.419318    2.353840   1.534223    -10.676587                                            Austria  simple-rnn   9848.0
747          population ages 15-64   2.067017    4.752729   2.180075    -22.576650                                            Austria   base-lstm  11340.0
2882  population ages 65 and above   0.532193    0.416629   0.645468     -0.284641                                            Austria          rf   5387.0
749   population ages 65 and above   0.724655    0.637785   0.798614     -0.966559                                            Austria   base-lstm   6475.0
38    population ages 65 and above   0.790347    0.775755   0.880769     -1.391979                                            Austria  simple-rnn   6946.0
2171  population ages 65 and above   0.831796    0.901188   0.949309     -1.778743                                            Austria     xgboost   7258.0
1460  population ages 65 and above   1.011851    1.192271   1.091912     -2.676274                                            Austria    base-gru   7999.0
2884          population ages 0-14   0.248379    0.064501   0.253971     -0.081595                                         Azerbaijan          rf   3092.0
2173          population ages 0-14   0.429867    0.309070   0.555941     -4.182661                                         Azerbaijan     xgboost   5768.0
751           population ages 0-14   4.568551   21.534382   4.640515   -360.100848                                         Azerbaijan   base-lstm  13770.0
1462          population ages 0-14   4.659198   22.326338   4.725075   -373.380817                                         Azerbaijan    base-gru  13799.0
40            population ages 0-14   4.695291   22.732415   4.767852   -380.190153                                         Azerbaijan  simple-rnn  13806.0
2883         population ages 15-64   0.068023    0.011548   0.107460      0.623186                                         Azerbaijan          rf   1260.0
2172         population ages 15-64   0.436092    0.300812   0.548463     -8.815855                                         Azerbaijan     xgboost   6069.0
1461         population ages 15-64   3.825623   15.431103   3.928244   -502.535990                                         Azerbaijan    base-gru  13565.0
39           population ages 15-64   4.103002   17.630915   4.198918   -574.318579                                         Azerbaijan  simple-rnn  13702.0
750          population ages 15-64   4.258809   19.262046   4.388855   -627.544399                                         Azerbaijan   base-lstm  13784.0
2174  population ages 65 and above   0.047094    0.007696   0.087727      0.947203                                         Azerbaijan     xgboost    541.0
2885  population ages 65 and above   0.241488    0.086839   0.294684      0.404251                                         Azerbaijan          rf   2996.0
752   population ages 65 and above   1.361103    1.862399   1.364697    -11.776803                                         Azerbaijan   base-lstm   9560.0
41    population ages 65 and above   1.544889    2.394620   1.547456    -15.428053                                         Azerbaijan  simple-rnn  10093.0
1463  population ages 65 and above   1.660679    2.766078   1.663153    -17.976405                                         Azerbaijan    base-gru  10385.0
2176          population ages 0-14   0.571623    0.442357   0.665099      0.916540                                       Bahamas, The     xgboost   4534.0
2887          population ages 0-14   0.918477    1.246291   1.116374      0.764862                                       Bahamas, The          rf   6617.0
1465          population ages 0-14   1.955122    3.844225   1.960670      0.274710                                       Bahamas, The    base-gru   9240.0
754           population ages 0-14   2.181321    4.834564   2.198764      0.087862                                       Bahamas, The   base-lstm   9697.0
43            population ages 0-14   2.337917    5.535185   2.352697     -0.044324                                       Bahamas, The  simple-rnn   9948.0
1464         population ages 15-64   1.329129    1.809021   1.344998      0.232646                                       Bahamas, The    base-gru   7969.0
2175         population ages 15-64   1.349938    2.435836   1.560716     -0.033237                                       Bahamas, The     xgboost   8482.0
2886         population ages 15-64   1.493097    3.242107   1.800585     -0.375243                                       Bahamas, The          rf   9061.0
42           population ages 15-64   1.973564    3.905152   1.976146     -0.656494                                       Bahamas, The  simple-rnn   9699.0
753          population ages 15-64   2.059060    4.262231   2.064517     -0.807961                                       Bahamas, The   base-lstm   9901.0
2177  population ages 65 and above   0.143830    0.035743   0.189059      0.940998                                       Bahamas, The     xgboost   1461.0
2888  population ages 65 and above   0.162072    0.064015   0.253012      0.894329                                       Bahamas, The          rf   1992.0
755   population ages 65 and above   1.042066    1.189551   1.090666     -0.963619                                       Bahamas, The   base-lstm   7612.0
44    population ages 65 and above   1.080401    1.296948   1.138836     -1.140901                                       Bahamas, The  simple-rnn   7849.0
1466  population ages 65 and above   1.231590    1.639202   1.280313     -1.705867                                       Bahamas, The    base-gru   8484.0
2890          population ages 0-14   0.563303    0.423217   0.650551    -10.067953                                            Bahrain          rf   6681.0
2179          population ages 0-14   0.700273    0.681420   0.825482    -16.820489                                            Bahrain     xgboost   7686.0
757           population ages 0-14   5.680003   38.811289   6.229871  -1013.992373                                            Bahrain   base-lstm  14071.0
46            population ages 0-14   5.892185   41.409073   6.434988  -1081.929566                                            Bahrain  simple-rnn  14092.0
1468          population ages 0-14   6.233618   46.372782   6.809756  -1211.740446                                            Bahrain    base-gru  14113.0
2178         population ages 15-64   2.473265    9.333372   3.055057    -48.244259                                            Bahrain     xgboost  12327.0
2889         population ages 15-64   5.160944   34.198120   5.847916   -179.434366                                            Bahrain          rf  13817.0
45           population ages 15-64   7.912071   70.165127   8.376463   -369.201643                                            Bahrain  simple-rnn  14011.0
756          population ages 15-64   7.984950   72.237368   8.499257   -380.135092                                            Bahrain   base-lstm  14017.0
1467         population ages 15-64   8.112219   74.747753   8.645678   -393.380260                                            Bahrain    base-gru  14028.0
2180  population ages 65 and above   0.274719    0.108867   0.329950      0.209120                                            Bahrain     xgboost   3367.0
2891  population ages 65 and above   0.535726    0.413218   0.642820     -2.001877                                            Bahrain          rf   5903.0
1469  population ages 65 and above   1.677570    2.923829   1.709921    -20.240536                                            Bahrain    base-gru  10507.0
758   population ages 65 and above   1.711168    3.099035   1.760408    -21.513340                                            Bahrain   base-lstm  10632.0
47    population ages 65 and above   1.918640    3.955449   1.988831    -27.734873                                            Bahrain  simple-rnn  11141.0
2893          population ages 0-14   0.642456    0.590532   0.768461      0.897042                                         Bangladesh          rf   5069.0
2182          population ages 0-14   0.725163    0.929345   0.964025      0.837971                                         Bangladesh     xgboost   5838.0
1471          population ages 0-14   4.429974   21.327055   4.618122     -2.718331                                         Bangladesh    base-gru  12326.0
760           population ages 0-14   4.485229   21.961888   4.686351     -2.829013                                         Bangladesh   base-lstm  12370.0
49            population ages 0-14   4.527781   22.182782   4.709860     -2.867526                                         Bangladesh  simple-rnn  12381.0
2892         population ages 15-64   1.089338    1.823496   1.350369      0.525763                                         Bangladesh          rf   7534.0
2181         population ages 15-64   1.605943    3.818092   1.953994      0.007028                                         Bangladesh     xgboost   9151.0
1470         population ages 15-64   2.467386    6.476456   2.544888     -0.684334                                         Bangladesh    base-gru  10394.0
48           population ages 15-64   2.647867    7.466544   2.732498     -0.941827                                         Bangladesh  simple-rnn  10679.0
759          population ages 15-64   3.128680   10.799240   3.286220     -1.808562                                         Bangladesh   base-lstm  11424.0
2894  population ages 65 and above   0.076184    0.015271   0.123578      0.924295                                         Bangladesh          rf    874.0
2183  population ages 65 and above   0.159420    0.033973   0.184319      0.831584                                         Bangladesh     xgboost   1750.0
761   population ages 65 and above   1.447108    2.395825   1.547845    -10.876759                                         Bangladesh   base-lstm   9911.0
50    population ages 65 and above   1.742280    3.511880   1.874001    -16.409346                                         Bangladesh  simple-rnn  10678.0
1472  population ages 65 and above   1.960793    4.367919   2.089957    -20.652963                                         Bangladesh    base-gru  11183.0
2896          population ages 0-14   0.650749    0.726715   0.852476      0.307950                                           Barbados          rf   6003.0
2185          population ages 0-14   1.087831    1.999199   1.413930     -0.903836                                           Barbados     xgboost   8311.0
1474          population ages 0-14   3.390388   11.693951   3.419642    -10.136142                                           Barbados    base-gru  12325.0
763           population ages 0-14   3.847351   15.170293   3.894906    -13.446660                                           Barbados   base-lstm  12764.0
52            population ages 0-14   3.940235   15.922284   3.990274    -14.162781                                           Barbados  simple-rnn  12829.0
2184         population ages 15-64   0.330239    0.159589   0.399486     -0.534481                                           Barbados     xgboost   4194.0
2895         population ages 15-64   0.668639    0.619881   0.787325     -4.960289                                           Barbados          rf   7043.0
1473         population ages 15-64   1.154236    1.527679   1.235993    -13.688975                                           Barbados    base-gru   9214.0
51           population ages 15-64   1.546706    2.768733   1.663951    -25.621978                                           Barbados  simple-rnn  10445.0
762          population ages 15-64   2.218141    5.555345   2.356978    -52.415872                                           Barbados   base-lstm  11762.0
2186  population ages 65 and above   0.467609    0.348523   0.590359      0.790551                                           Barbados     xgboost   4387.0
2897  population ages 65 and above   0.624318    0.794179   0.891167      0.522729                                           Barbados          rf   5888.0
764   population ages 65 and above   1.600665    2.592834   1.610228     -0.558194                                           Barbados   base-lstm   8969.0
53    population ages 65 and above   1.757285    3.121633   1.766814     -0.875982                                           Barbados  simple-rnn   9378.0
1475  population ages 65 and above   1.864741    3.506534   1.872574     -1.107292                                           Barbados    base-gru   9646.0
2188          population ages 0-14   0.145818    0.028645   0.169248     -1.281154                                            Belarus     xgboost   2762.0
2899          population ages 0-14   0.319954    0.148252   0.385035    -10.806100                                            Belarus          rf   5269.0
1477          population ages 0-14   3.169475   10.158694   3.187271   -807.992577                                            Belarus    base-gru  13073.0
766           population ages 0-14   3.442383   12.048539   3.471101   -958.491319                                            Belarus   base-lstm  13346.0
55            population ages 0-14   3.667362   13.643562   3.693719  -1085.511699                                            Belarus  simple-rnn  13532.0
2187         population ages 15-64   0.261996    0.122863   0.350518      0.721637                                            Belarus     xgboost   3044.0
2898         population ages 15-64   0.332198    0.150596   0.388066      0.658804                                            Belarus          rf   3484.0
1476         population ages 15-64   1.446658    2.325818   1.525063     -4.269475                                            Belarus    base-gru   9487.0
54           population ages 15-64   1.924676    4.079317   2.019732     -8.242279                                            Belarus  simple-rnn  10737.0
765          population ages 15-64   2.401101    6.279449   2.505883    -13.226995                                            Belarus   base-lstm  11501.0
2189  population ages 65 and above   0.110010    0.026708   0.163427      0.915828                                            Belarus     xgboost   1253.0
2900  population ages 65 and above   0.250406    0.087690   0.296125      0.723642                                            Belarus          rf   2751.0
767   population ages 65 and above   0.702393    0.534575   0.731147     -0.684730                                            Belarus   base-lstm   6135.0
56    population ages 65 and above   0.762473    0.613728   0.783408     -0.934183                                            Belarus  simple-rnn   6456.0
1478  population ages 65 and above   0.950631    0.939009   0.969025     -1.959314                                            Belarus    base-gru   7498.0
2191          population ages 0-14   0.048418    0.004640   0.068120      0.495301                                            Belgium     xgboost   1124.0
2902          population ages 0-14   0.092982    0.017361   0.131761     -0.888245                                            Belgium          rf   2228.0
1480          population ages 0-14   2.664728    7.194798   2.682312   -781.531628                                            Belgium    base-gru  12584.0
769           population ages 0-14   2.980690    9.042011   3.006994   -982.440997                                            Belgium   base-lstm  12912.0
58            population ages 0-14   3.240989   10.675201   3.267293  -1160.072541                                            Belgium  simple-rnn  13162.0
2901         population ages 15-64   0.049402    0.004527   0.067285      0.893353                                            Belgium          rf    626.0
2190         population ages 15-64   0.115510    0.018531   0.136127      0.563490                                            Belgium     xgboost   1633.0
1479         population ages 15-64   0.917656    0.936782   0.967875    -21.066993                                            Belgium    base-gru   8450.0
57           population ages 15-64   1.592883    2.701204   1.643534    -62.629998                                            Belgium  simple-rnn  10682.0
768          population ages 15-64   1.991147    4.179344   2.044344    -97.449310                                            Belgium   base-lstm  11531.0
2903  population ages 65 and above   0.085484    0.014874   0.121959      0.831785                                            Belgium          rf   1107.0
770   population ages 65 and above   0.123635    0.016141   0.127048      0.817454                                            Belgium   base-lstm   1295.0
59    population ages 65 and above   0.155591    0.024641   0.156976      0.721322                                            Belgium  simple-rnn   1732.0
2192  population ages 65 and above   0.194070    0.070072   0.264712      0.207531                                            Belgium     xgboost   2805.0
1481  population ages 65 and above   0.350133    0.123055   0.350792     -0.391671                                            Belgium    base-gru   3946.0
2194          population ages 0-14   0.304124    0.178005   0.421907      0.971147                                             Belize     xgboost   2967.0
1483          population ages 0-14   1.195317    1.491970   1.221462      0.758168                                             Belize    base-gru   7196.0
772           population ages 0-14   1.296928    1.761246   1.327119      0.714521                                             Belize   base-lstm   7547.0
2905          population ages 0-14   1.289562    2.159042   1.469368      0.650043                                             Belize          rf   7842.0
61            population ages 0-14   1.495620    2.336028   1.528407      0.621355                                             Belize  simple-rnn   8125.0
1482         population ages 15-64   0.157460    0.041290   0.203198      0.991302                                             Belize    base-gru   1364.0
771          population ages 15-64   0.360452    0.135735   0.368422      0.971406                                             Belize   base-lstm   2854.0
2193         population ages 15-64   0.331831    0.159757   0.399696      0.966346                                             Belize     xgboost   2946.0
60           population ages 15-64   0.501261    0.255673   0.505642      0.946140                                             Belize  simple-rnn   3774.0
2904         population ages 15-64   1.545575    3.187519   1.785363      0.328518                                             Belize          rf   8731.0
2195  population ages 65 and above   0.113749    0.031609   0.177790      0.685094                                             Belize     xgboost   1753.0
2906  population ages 65 and above   0.383700    0.245772   0.495754     -1.448494                                             Belize          rf   4985.0
773   population ages 65 and above   1.126263    1.407096   1.186211    -13.018131                                             Belize   base-lstm   9014.0
62    population ages 65 and above   1.130067    1.442995   1.201247    -13.375771                                             Belize  simple-rnn   9064.0
1484  population ages 65 and above   1.333805    1.966707   1.402393    -18.593228                                             Belize    base-gru   9778.0
1486          population ages 0-14   0.459504    0.267413   0.517120     -1.002762                                              Benin    base-gru   5089.0
2908          population ages 0-14   0.395019    0.299808   0.547547     -1.245380                                              Benin          rf   5136.0
2197          population ages 0-14   0.534714    0.489850   0.699893     -2.668676                                              Benin     xgboost   6222.0
775           population ages 0-14   0.616757    0.482577   0.694677     -2.614203                                              Benin   base-lstm   6367.0
64            population ages 0-14   0.885407    0.932014   0.965409     -5.980211                                              Benin  simple-rnn   7914.0
63           population ages 15-64   0.093948    0.010995   0.104859      0.932410                                              Benin  simple-rnn    809.0
774          population ages 15-64   0.162031    0.027792   0.166709      0.829161                                              Benin   base-lstm   1651.0
2907         population ages 15-64   0.176155    0.070760   0.266007      0.565036                                              Benin          rf   2518.0
1485         population ages 15-64   0.359846    0.137087   0.370253      0.157321                                              Benin    base-gru   3817.0
2196         population ages 15-64   0.440917    0.283978   0.532896     -0.745623                                              Benin     xgboost   5037.0
2198  population ages 65 and above   0.061000    0.005858   0.076535     -2.236132                                              Benin     xgboost   2184.0
2909  population ages 65 and above   0.076269    0.008874   0.094200     -3.902426                                              Benin          rf   2585.0
65    population ages 65 and above   0.684731    0.538532   0.733848   -296.524741                                              Benin  simple-rnn   8005.0
776   population ages 65 and above   0.720810    0.588737   0.767292   -324.261525                                              Benin   base-lstm   8172.0
1487  population ages 65 and above   0.780387    0.691267   0.831425   -380.906912                                              Benin    base-gru   8524.0
2200          population ages 0-14   0.088528    0.011359   0.106577      0.997087                                             Bhutan     xgboost    551.0
2911          population ages 0-14   0.203044    0.069006   0.262690      0.982304                                             Bhutan          rf   1850.0
778           population ages 0-14   1.097403    1.363977   1.167894      0.650225                                             Bhutan   base-lstm   7075.0
1489          population ages 0-14   1.132169    1.433379   1.197238      0.632427                                             Bhutan    base-gru   7183.0
67            population ages 0-14   1.220337    1.647084   1.283388      0.577625                                             Bhutan  simple-rnn   7510.0
777          population ages 15-64   0.276170    0.107961   0.328574      0.962794                                             Bhutan   base-lstm   2473.0
1488         population ages 15-64   0.296836    0.120576   0.347241      0.958447                                             Bhutan    base-gru   2627.0
66           population ages 15-64   0.301912    0.122749   0.350355      0.957698                                             Bhutan  simple-rnn   2658.0
2199         population ages 15-64   0.314816    0.180924   0.425351      0.937650                                             Bhutan     xgboost   3112.0
2910         population ages 15-64   0.481765    0.470944   0.686253      0.837702                                             Bhutan          rf   4601.0
2912  population ages 65 and above   0.152519    0.058912   0.242719      0.256995                                             Bhutan          rf   2570.0
2201  population ages 65 and above   0.333008    0.215928   0.464681     -1.723299                                             Bhutan     xgboost   4846.0
779   population ages 65 and above   1.320572    2.099360   1.448917    -25.477240                                             Bhutan   base-lstm   9957.0
68    population ages 65 and above   1.433974    2.479768   1.574728    -30.274965                                             Bhutan  simple-rnn  10285.0
1490  population ages 65 and above   1.446921    2.494072   1.579263    -30.455360                                             Bhutan    base-gru  10318.0
2203          population ages 0-14   0.306958    0.135446   0.368029      0.918356                                            Bolivia     xgboost   2884.0
2914          population ages 0-14   0.489782    0.353343   0.594427      0.787012                                            Bolivia          rf   4443.0
1492          population ages 0-14   1.782636    3.762683   1.939764     -1.268065                                            Bolivia    base-gru   9736.0
781           population ages 0-14   1.969263    4.606765   2.146338     -1.776860                                            Bolivia   base-lstm  10217.0
70            population ages 0-14   2.148418    5.397537   2.323260     -2.253521                                            Bolivia  simple-rnn  10544.0
2913         population ages 15-64   0.076541    0.011412   0.106825      0.992473                                            Bolivia          rf    536.0
1491         population ages 15-64   0.146899    0.029143   0.170714      0.980777                                            Bolivia    base-gru   1216.0
2202         population ages 15-64   0.494609    0.300805   0.548457      0.801583                                            Bolivia     xgboost   4254.0
780          population ages 15-64   0.504368    0.351346   0.592745      0.768245                                            Bolivia   base-lstm   4486.0
69           population ages 15-64   0.525920    0.373553   0.611190      0.753597                                            Bolivia  simple-rnn   4599.0
2204  population ages 65 and above   0.122711    0.017847   0.133594     -3.785797                                            Bolivia     xgboost   2951.0
2915  population ages 65 and above   0.121821    0.024975   0.158036     -5.697167                                            Bolivia          rf   3248.0
782   population ages 65 and above   1.336530    2.077751   1.441441   -556.150070                                            Bolivia   base-lstm  10533.0
71    population ages 65 and above   1.386561    2.252312   1.500771   -602.958635                                            Bolivia  simple-rnn  10681.0
1493  population ages 65 and above   1.540494    2.745080   1.656828   -735.094559                                            Bolivia    base-gru  11033.0
2206          population ages 0-14   0.067199    0.009462   0.097271     -6.461329                             Bosnia and Herzegovina     xgboost   2782.0
2917          population ages 0-14   0.087907    0.013360   0.115584     -9.535342                             Bosnia and Herzegovina          rf   3105.0
1495          population ages 0-14   2.328034    5.589094   2.364127  -4406.515047                             Bosnia and Herzegovina    base-gru  12308.0
784           population ages 0-14   2.623404    7.136971   2.671511  -5627.158553                             Bosnia and Herzegovina   base-lstm  12624.0
73            population ages 0-14   2.899471    8.675773   2.945466  -6840.644999                             Bosnia and Herzegovina  simple-rnn  12885.0
2205         population ages 15-64   0.267269    0.113321   0.336632      0.836218                             Bosnia and Herzegovina     xgboost   2798.0
2916         population ages 15-64   0.319115    0.159692   0.399615      0.769198                             Bosnia and Herzegovina          rf   3379.0
1494         population ages 15-64   1.650073    3.326034   1.823742     -3.807088                             Bosnia and Herzegovina    base-gru   9998.0
72           population ages 15-64   1.949967    4.596409   2.143924     -5.643150                             Bosnia and Herzegovina  simple-rnn  10711.0
783          population ages 15-64   2.556434    7.510237   2.740481     -9.854481                             Bosnia and Herzegovina   base-lstm  11644.0
2207  population ages 65 and above   0.249909    0.101085   0.317939      0.865240                             Bosnia and Herzegovina     xgboost   2615.0
785   population ages 65 and above   0.345759    0.156280   0.395323      0.791657                             Bosnia and Herzegovina   base-lstm   3387.0
74    population ages 65 and above   0.327616    0.169929   0.412224      0.773462                             Bosnia and Herzegovina  simple-rnn   3454.0
1496  population ages 65 and above   0.366978    0.226431   0.475848      0.698136                             Bosnia and Herzegovina    base-gru   3882.0
2918  population ages 65 and above   0.432342    0.256522   0.506480      0.658021                             Bosnia and Herzegovina          rf   4173.0
2209          population ages 0-14   0.776858    0.870994   0.933271     -1.098305                                           Botswana     xgboost   6969.0
2920          population ages 0-14   1.353537    2.534448   1.591995     -5.105717                                           Botswana          rf   9595.0
787           population ages 0-14   4.019031   18.648998   4.318449    -43.927147                                           Botswana   base-lstm  13295.0
1498          population ages 0-14   4.238291   20.745214   4.554691    -48.977125                                           Botswana    base-gru  13387.0
76            population ages 0-14   4.331639   21.605107   4.648129    -51.048687                                           Botswana  simple-rnn  13423.0
2208         population ages 15-64   0.627867    0.512242   0.715711     -1.068201                                           Botswana     xgboost   6093.0
2919         population ages 15-64   1.173148    1.994639   1.412317     -7.053451                                           Botswana          rf   9281.0
1497         population ages 15-64   3.032651   10.688902   3.269389    -42.156949                                           Botswana    base-gru  12584.0
786          population ages 15-64   3.044053   10.918330   3.304290    -43.083279                                           Botswana   base-lstm  12634.0
75           population ages 15-64   3.183391   11.726328   3.424373    -46.345609                                           Botswana  simple-rnn  12771.0
2210  population ages 65 and above   0.098499    0.015855   0.125917      0.352198                                           Botswana     xgboost   1673.0
2921  population ages 65 and above   0.096559    0.022623   0.150408      0.075692                                           Botswana          rf   1956.0
788   population ages 65 and above   1.302294    1.865072   1.365676    -75.202712                                           Botswana   base-lstm  10070.0
77    population ages 65 and above   1.449314    2.353781   1.534204    -95.170307                                           Botswana  simple-rnn  10505.0
1499  population ages 65 and above   1.511641    2.516005   1.586192   -101.798436                                           Botswana    base-gru  10652.0
2923          population ages 0-14   0.159766    0.036816   0.191876      0.983056                                             Brazil          rf   1374.0
2212          population ages 0-14   0.545471    0.393943   0.627649      0.818694                                             Brazil     xgboost   4575.0
1501          population ages 0-14   2.303353    5.754388   2.398831     -1.648370                                             Brazil    base-gru  10512.0
790           population ages 0-14   2.475732    6.768275   2.601591     -2.114996                                             Brazil   base-lstm  10840.0
79            population ages 0-14   2.602110    7.351018   2.711276     -2.383195                                             Brazil  simple-rnn  11025.0
2922         population ages 15-64   0.260902    0.087560   0.295905      0.770473                                             Brazil          rf   2710.0
2211         population ages 15-64   0.373636    0.174972   0.418297      0.541330                                             Brazil     xgboost   3811.0
1500         population ages 15-64   1.499180    2.361301   1.536653     -5.189883                                             Brazil    base-gru   9619.0
78           population ages 15-64   1.934772    3.947355   1.986795     -9.347545                                             Brazil  simple-rnn  10755.0
789          population ages 15-64   2.260040    5.557941   2.357529    -13.569513                                             Brazil   base-lstm  11360.0
2213  population ages 65 and above   0.068086    0.006673   0.081685      0.991753                                             Brazil     xgboost    387.0
2924  population ages 65 and above   0.310379    0.175110   0.418461      0.783570                                             Brazil          rf   3422.0
791   population ages 65 and above   0.925094    0.925403   0.961979     -0.143769                                             Brazil   base-lstm   6872.0
80    population ages 65 and above   1.091916    1.310363   1.144711     -0.619567                                             Brazil  simple-rnn   7701.0
1502  population ages 65 and above   1.165659    1.470885   1.212800     -0.817968                                             Brazil    base-gru   8012.0
2926          population ages 0-14   0.208548    0.061653   0.248301      0.924820                                  Brunei Darussalam          rf   2002.0
2215          population ages 0-14   0.618506    0.515828   0.718212      0.370997                                  Brunei Darussalam     xgboost   5479.0
793           population ages 0-14   3.135629   11.071126   3.327330    -12.500172                                  Brunei Darussalam   base-lstm  12271.0
1504          population ages 0-14   3.444951   13.273677   3.643306    -15.185970                                  Brunei Darussalam    base-gru  12612.0
82            population ages 0-14   3.474647   13.477440   3.671163    -15.434440                                  Brunei Darussalam  simple-rnn  12637.0
2925         population ages 15-64   0.126061    0.025776   0.160549      0.504840                                  Brunei Darussalam          rf   1842.0
2214         population ages 15-64   0.882409    1.033921   1.016819    -18.861819                                  Brunei Darussalam     xgboost   8471.0
1503         population ages 15-64   3.958306   17.468629   4.179549   -334.575737                                  Brunei Darussalam    base-gru  13592.0
792          population ages 15-64   3.998811   17.970528   4.239166   -344.217308                                  Brunei Darussalam   base-lstm  13619.0
81           population ages 15-64   4.136907   18.825195   4.338801   -360.635642                                  Brunei Darussalam  simple-rnn  13677.0
2216  population ages 65 and above   0.368717    0.239235   0.489117      0.532082                                  Brunei Darussalam     xgboost   4070.0
2927  population ages 65 and above   0.636598    0.657931   0.811129     -0.286842                                  Brunei Darussalam          rf   6134.0
794   population ages 65 and above   0.997484    0.995495   0.997745     -0.947082                                  Brunei Darussalam   base-lstm   7349.0
1505  population ages 65 and above   1.122566    1.260538   1.122737     -1.465476                                  Brunei Darussalam    base-gru   7942.0
83    population ages 65 and above   1.146076    1.319719   1.148790     -1.581228                                  Brunei Darussalam  simple-rnn   8063.0
2218          population ages 0-14   0.259740    0.071575   0.267536    -33.844975                                           Bulgaria     xgboost   4906.0
2929          population ages 0-14   0.451267    0.249313   0.499313   -120.373065                                           Bulgaria          rf   6665.0
1507          population ages 0-14   2.647181    7.174302   2.678489  -3491.662718                                           Bulgaria    base-gru  12639.0
796           population ages 0-14   3.002740    9.301832   3.049891  -4527.407784                                           Bulgaria   base-lstm  13001.0
85            population ages 0-14   3.477386   12.436383   3.526526  -6053.400111                                           Bulgaria  simple-rnn  13440.0
2928         population ages 15-64   0.114760    0.014475   0.120313      0.981419                                           Bulgaria          rf    804.0
2217         population ages 15-64   0.377187    0.200405   0.447666      0.742749                                           Bulgaria     xgboost   3752.0
1506         population ages 15-64   2.438633    6.570835   2.563364     -7.434711                                           Bulgaria    base-gru  11362.0
84           population ages 15-64   3.176458   11.032981   3.321593    -13.162586                                           Bulgaria  simple-rnn  12298.0
795          population ages 15-64   3.783335   15.437546   3.929064    -18.816546                                           Bulgaria   base-lstm  12883.0
2930  population ages 65 and above   0.194287    0.051724   0.227428      0.939163                                           Bulgaria          rf   1805.0
1508  population ages 65 and above   0.940044    1.081898   1.040143     -0.272521                                           Bulgaria    base-gru   7134.0
2219  population ages 65 and above   0.929382    1.284219   1.133234     -0.510490                                           Bulgaria     xgboost   7447.0
86    population ages 65 and above   1.114352    1.439737   1.199890     -0.693408                                           Bulgaria  simple-rnn   7875.0
797   population ages 65 and above   1.205976    1.692085   1.300802     -0.990219                                           Bulgaria   base-lstm   8270.0
2932          population ages 0-14   0.133407    0.030187   0.173744      0.883621                                       Burkina Faso          rf   1490.0
2221          population ages 0-14   0.159332    0.029463   0.171647      0.886414                                       Burkina Faso     xgboost   1557.0
1510          population ages 0-14   0.638563    0.435305   0.659777     -0.678216                                       Burkina Faso    base-gru   5787.0
799           population ages 0-14   0.803795    0.705543   0.839966     -1.720051                                       Burkina Faso   base-lstm   6961.0
88            population ages 0-14   1.006627    1.100965   1.049269     -3.244509                                       Burkina Faso  simple-rnn   7992.0
2931         population ages 15-64   0.131051    0.023300   0.152643      0.925687                                       Burkina Faso          rf   1230.0
798          population ages 15-64   0.136070    0.024052   0.155088      0.923287                                       Burkina Faso   base-lstm   1271.0
1509         population ages 15-64   0.135594    0.028733   0.169509      0.908358                                       Burkina Faso    base-gru   1406.0
87           population ages 15-64   0.228666    0.061984   0.248966      0.802307                                       Burkina Faso  simple-rnn   2350.0
2220         population ages 15-64   0.343550    0.150770   0.388291      0.519132                                       Burkina Faso     xgboost   3642.0
2933  population ages 65 and above   0.149640    0.037724   0.194228     -8.527453                                       Burkina Faso          rf   3749.0
2222  population ages 65 and above   0.209286    0.074791   0.273479    -17.888770                                       Burkina Faso     xgboost   4572.0
89    population ages 65 and above   0.650827    0.492592   0.701849   -123.406438                                       Burkina Faso  simple-rnn   7734.0
800   population ages 65 and above   0.681371    0.533174   0.730188   -133.655650                                       Burkina Faso   base-lstm   7883.0
1511  population ages 65 and above   0.710571    0.582656   0.763319   -146.152473                                       Burkina Faso    base-gru   8039.0
1513          population ages 0-14   1.340995    2.567211   1.602252     -1.702750                                            Burundi    base-gru   9130.0
802           population ages 0-14   1.410379    2.824058   1.680493     -1.973158                                            Burundi   base-lstm   9300.0
91            population ages 0-14   1.398535    2.834457   1.683585     -1.984107                                            Burundi  simple-rnn   9303.0
2224          population ages 0-14   1.520781    3.298292   1.816120     -2.472430                                            Burundi     xgboost   9683.0
2935          population ages 0-14   1.554260    3.327864   1.824244     -2.503564                                            Burundi          rf   9725.0
1512         population ages 15-64   0.948342    1.097913   1.047813     -0.095579                                            Burundi    base-gru   7084.0
90           population ages 15-64   1.015494    1.248639   1.117425     -0.245985                                            Burundi  simple-rnn   7408.0
801          population ages 15-64   1.084484    1.446050   1.202518     -0.442977                                            Burundi   base-lstm   7758.0
2934         population ages 15-64   1.581153    3.716015   1.927697     -2.708117                                            Burundi          rf   9890.0
2223         population ages 15-64   1.624851    3.819713   1.954408     -2.811594                                            Burundi     xgboost   9973.0
2225  population ages 65 and above   0.021622    0.000901   0.030012      0.818984                                            Burundi     xgboost    566.0
2936  population ages 65 and above   0.107172    0.015427   0.124205     -2.100250                                            Burundi          rf   2555.0
803   population ages 65 and above   0.713663    0.548312   0.740481   -109.191123                                            Burundi   base-lstm   7932.0
92    population ages 65 and above   0.740026    0.592997   0.770063   -118.171139                                            Burundi  simple-rnn   8090.0
1514  population ages 65 and above   0.796912    0.696024   0.834281   -138.875877                                            Burundi    base-gru   8407.0
2938          population ages 0-14   0.833363    1.065947   1.032447      0.136248                                         Cabo Verde          rf   6767.0
2227          population ages 0-14   1.034178    1.467484   1.211398     -0.189123                                         Cabo Verde     xgboost   7619.0
1516          population ages 0-14   1.771642    3.717086   1.927975     -2.012007                                         Cabo Verde    base-gru   9890.0
805           population ages 0-14   1.888799    4.230630   2.056849     -2.428139                                         Cabo Verde   base-lstm  10218.0
94            population ages 0-14   1.966819    4.523232   2.126789     -2.665238                                         Cabo Verde  simple-rnn  10358.0
2937         population ages 15-64   0.192576    0.057808   0.240433      0.956090                                         Cabo Verde          rf   1835.0
2226         population ages 15-64   0.222246    0.077505   0.278398      0.941128                                         Cabo Verde     xgboost   2136.0
804          population ages 15-64   0.545877    0.430901   0.656430      0.672695                                         Cabo Verde   base-lstm   4864.0
93           population ages 15-64   0.721788    0.610690   0.781466      0.536131                                         Cabo Verde  simple-rnn   5707.0
1515         population ages 15-64   0.874867    0.796152   0.892274      0.395257                                         Cabo Verde    base-gru   6354.0
2228  population ages 65 and above   0.210534    0.056227   0.237122     -3.839513                                         Cabo Verde     xgboost   3839.0
2939  population ages 65 and above   0.223484    0.057698   0.240203     -3.966087                                         Cabo Verde          rf   3893.0
806   population ages 65 and above   2.337397    5.846455   2.417944   -502.210324                                         Cabo Verde   base-lstm  12213.0
95    population ages 65 and above   2.411370    6.241887   2.498377   -536.245545                                         Cabo Verde  simple-rnn  12314.0
1517  population ages 65 and above   2.516089    6.777007   2.603269   -582.303945                                         Cabo Verde    base-gru  12458.0
2230          population ages 0-14   0.092278    0.012433   0.111505      0.975220                                           Cambodia     xgboost    694.0
2941          population ages 0-14   0.526777    0.408677   0.639278      0.185503                                           Cambodia          rf   5121.0
808           population ages 0-14   3.300127   12.572353   3.545751    -24.056819                                           Cambodia   base-lstm  12696.0
1519          population ages 0-14   3.357260   12.970040   3.601394    -24.849413                                           Cambodia    base-gru  12758.0
97            population ages 0-14   3.441742   13.614327   3.689760    -26.133484                                           Cambodia  simple-rnn  12843.0
2229         population ages 15-64   0.591858    0.395397   0.628806    -14.191270                                           Cambodia     xgboost   6803.0
2940         population ages 15-64   0.948709    1.216681   1.103033    -45.745274                                           Cambodia          rf   9026.0
1518         population ages 15-64   2.637732    8.280646   2.877611   -317.144975                                           Cambodia    base-gru  12573.0
96           population ages 15-64   2.665124    8.544076   2.923025   -327.266033                                           Cambodia  simple-rnn  12612.0
807          population ages 15-64   2.725566    9.056327   3.009373   -346.946899                                           Cambodia   base-lstm  12710.0
2231  population ages 65 and above   0.297627    0.159758   0.399698      0.478386                                           Cambodia     xgboost   3609.0
2942  population ages 65 and above   0.318870    0.180815   0.425224      0.409634                                           Cambodia          rf   3818.0
809   population ages 65 and above   0.970735    0.963353   0.981506     -2.145370                                           Cambodia   base-lstm   7606.0
98    population ages 65 and above   1.107814    1.271726   1.127708     -3.152215                                           Cambodia  simple-rnn   8279.0
1520  population ages 65 and above   1.137504    1.330657   1.153541     -3.344623                                           Cambodia    base-gru   8395.0
2944          population ages 0-14   0.481892    0.378159   0.614946     -0.804201                                           Cameroon          rf   5394.0
2233          population ages 0-14   0.732716    0.746288   0.863880     -2.560550                                           Cameroon     xgboost   7082.0
1522          population ages 0-14   1.572320    2.886602   1.699000    -12.772014                                           Cameroon    base-gru  10255.0
811           population ages 0-14   1.733179    3.506909   1.872674    -15.731508                                           Cameroon   base-lstm  10653.0
100           population ages 0-14   2.061181    4.900571   2.213723    -22.380690                                           Cameroon  simple-rnn  11370.0
2232         population ages 15-64   0.140203    0.031992   0.178864      0.900045                                           Cameroon     xgboost   1516.0
2943         population ages 15-64   0.269020    0.112086   0.334793      0.649804                                           Cameroon          rf   3065.0
1521         population ages 15-64   0.576951    0.399113   0.631754     -0.246968                                           Cameroon    base-gru   5426.0
810          population ages 15-64   0.760918    0.703297   0.838628     -1.197343                                           Cameroon   base-lstm   6738.0
99           population ages 15-64   1.044401    1.288898   1.135297     -3.026967                                           Cameroon  simple-rnn   8202.0
2234  population ages 65 and above   0.175266    0.044323   0.210531     -2.702169                                           Cameroon     xgboost   3362.0
2945  population ages 65 and above   0.173137    0.044918   0.211939     -2.751875                                           Cameroon          rf   3373.0
101   population ages 65 and above   0.869640    0.875017   0.935423    -72.087342                                           Cameroon  simple-rnn   8618.0
812   population ages 65 and above   0.918928    0.961680   0.980653    -79.326057                                           Cameroon   base-lstm   8828.0
1523  population ages 65 and above   0.978879    1.091572   1.044783    -90.175508                                           Cameroon    base-gru   9060.0
2947          population ages 0-14   0.236220    0.091508   0.302502     -0.380100                                             Canada          rf   3415.0
2236          population ages 0-14   0.412776    0.220759   0.469850     -2.329450                                             Canada     xgboost   5161.0
1525          population ages 0-14   1.194886    1.651512   1.285112    -23.907786                                             Canada    base-gru   9532.0
103           population ages 0-14   1.789245    3.649306   1.910316    -54.038133                                             Canada  simple-rnn  11116.0
814           population ages 0-14   1.789876    3.685514   1.919769    -54.584218                                             Canada   base-lstm  11132.0
1524         population ages 15-64   0.618131    0.517142   0.719126      0.658368                                             Canada    base-gru   5243.0
2946         population ages 15-64   0.554332    0.601020   0.775255      0.602957                                             Canada          rf   5361.0
102          population ages 15-64   1.075422    1.760062   1.326673     -0.162724                                             Canada  simple-rnn   7876.0
813          population ages 15-64   1.457440    3.190897   1.786308     -1.107955                                             Canada   base-lstm   9275.0
2235         population ages 15-64   1.459201    3.301296   1.816947     -1.180886                                             Canada     xgboost   9337.0
104   population ages 65 and above   0.400599    0.227224   0.476681      0.895724                                             Canada  simple-rnn   3621.0
815   population ages 65 and above   0.425011    0.251078   0.501076      0.884777                                             Canada   base-lstm   3783.0
1526  population ages 65 and above   0.522699    0.354662   0.595535      0.837241                                             Canada    base-gru   4387.0
2237  population ages 65 and above   0.644459    0.524016   0.723889      0.759522                                             Canada     xgboost   5188.0
2948  population ages 65 and above   0.937307    1.295157   1.138049      0.405635                                             Canada          rf   7022.0
2950          population ages 0-14   0.131446    0.024677   0.157090      0.988742                             Caribbean small states          rf   1033.0
2239          population ages 0-14   0.133076    0.026677   0.163331      0.987830                             Caribbean small states     xgboost   1089.0
1528          population ages 0-14   1.478629    2.424431   1.557058     -0.106033                             Caribbean small states    base-gru   8608.0
817           population ages 0-14   1.628460    2.988497   1.728727     -0.363361                             Caribbean small states   base-lstm   9051.0
106           population ages 0-14   1.664077    3.064566   1.750590     -0.398064                             Caribbean small states  simple-rnn   9119.0
1527         population ages 15-64   0.050403    0.004311   0.065658      0.995348                             Caribbean small states    base-gru    259.0
2949         population ages 15-64   0.254332    0.081868   0.286126      0.911665                             Caribbean small states          rf   2344.0
105          population ages 15-64   0.314269    0.111027   0.333207      0.880202                             Caribbean small states  simple-rnn   2831.0
2238         population ages 15-64   0.416579    0.212736   0.461233      0.770459                             Caribbean small states     xgboost   3847.0
816          population ages 15-64   0.456666    0.262120   0.511977      0.717173                             Caribbean small states   base-lstm   4195.0
2951  population ages 65 and above   0.156412    0.055187   0.234920      0.810360                             Caribbean small states          rf   2040.0
2240  population ages 65 and above   0.701639    0.778916   0.882562     -1.676588                             Caribbean small states     xgboost   6892.0
818   population ages 65 and above   1.273017    1.792775   1.338946     -5.160513                             Caribbean small states   base-lstm   9120.0
107   population ages 65 and above   1.324386    1.953839   1.397798     -5.713977                             Caribbean small states  simple-rnn   9316.0
1529  population ages 65 and above   1.464489    2.361255   1.536638     -7.113981                             Caribbean small states    base-gru   9745.0
2953          population ages 0-14   1.826277    4.483643   2.117461     -3.233698                           Central African Republic          rf  10381.0
2242          population ages 0-14   1.870333    4.641788   2.154481     -3.383027                           Central African Republic     xgboost  10455.0
820           population ages 0-14   3.042149   10.555695   3.248953     -8.967257                           Central African Republic   base-lstm  12054.0
1531          population ages 0-14   3.109827   11.080807   3.328785     -9.463097                           Central African Republic    base-gru  12165.0
109           population ages 0-14   3.413855   13.225079   3.636630    -11.487834                           Central African Republic  simple-rnn  12496.0
2952         population ages 15-64   1.933088    5.089633   2.256022     -2.764491                           Central African Republic          rf  10479.0
2241         population ages 15-64   2.153640    6.257698   2.501539     -3.628438                           Central African Republic     xgboost  10891.0
819          population ages 15-64   2.686285    8.412176   2.900375     -5.221974                           Central African Republic   base-lstm  11508.0
1530         population ages 15-64   2.788215    9.037724   3.006281     -5.684653                           Central African Republic    base-gru  11639.0
108          population ages 15-64   3.109601   11.112143   3.333488     -7.218974                           Central African Republic  simple-rnn  12068.0
2954  population ages 65 and above   0.185374    0.063346   0.251686     -1.019641                           Central African Republic          rf   3236.0
2243  population ages 65 and above   0.228420    0.085868   0.293032     -1.737704                           Central African Republic     xgboost   3766.0
110   population ages 65 and above   0.379727    0.144962   0.380739     -3.621816                           Central African Republic  simple-rnn   4942.0
821   population ages 65 and above   0.464442    0.216976   0.465807     -5.917812                           Central African Republic   base-lstm   5673.0
1532  population ages 65 and above   0.465401    0.217515   0.466385     -5.935009                           Central African Republic    base-gru   5678.0
2245          population ages 0-14   0.058039    0.005989   0.077387      0.631375                     Central Europe and the Baltics     xgboost   1072.0
2956          population ages 0-14   0.158227    0.036555   0.191194     -1.250092                     Central Europe and the Baltics          rf   2933.0
1534          population ages 0-14   2.842860    8.416796   2.901171   -517.084230                     Central Europe and the Baltics    base-gru  12720.0
823           population ages 0-14   3.215546   10.837566   3.292046   -666.091403                     Central Europe and the Baltics   base-lstm  13141.0
112           population ages 0-14   3.515741   12.885950   3.589701   -792.176841                     Central Europe and the Baltics  simple-rnn  13415.0
2244         population ages 15-64   0.345724    0.136226   0.369088      0.865823                     Central Europe and the Baltics     xgboost   3116.0
2955         population ages 15-64   0.707211    0.660829   0.812914      0.349112                     Central Europe and the Baltics          rf   5924.0
1533         population ages 15-64   2.381968    6.377351   2.525342     -5.281421                     Central Europe and the Baltics    base-gru  11148.0
111          population ages 15-64   2.919272    9.498691   3.081995     -8.355810                     Central Europe and the Baltics  simple-rnn  11898.0
822          population ages 15-64   3.532146   13.650855   3.694706    -12.445516                     Central Europe and the Baltics   base-lstm  12598.0
2246  population ages 65 and above   0.214623    0.056498   0.237693      0.927313                     Central Europe and the Baltics     xgboost   1982.0
2957  population ages 65 and above   0.312736    0.124624   0.353022      0.839667                     Central Europe and the Baltics          rf   2997.0
1535  population ages 65 and above   0.350654    0.169380   0.411558      0.782087                     Central Europe and the Baltics    base-gru   3492.0
113   population ages 65 and above   0.454555    0.281402   0.530473      0.637967                     Central Europe and the Baltics  simple-rnn   4323.0
824   population ages 65 and above   0.521813    0.373987   0.611545      0.518853                     Central Europe and the Baltics   base-lstm   4819.0
115           population ages 0-14   0.094702    0.012239   0.110632      0.857909                                               Chad  simple-rnn   1026.0
826           population ages 0-14   0.136295    0.026451   0.162639      0.692915                                               Chad   base-lstm   1740.0
1537          population ages 0-14   0.314071    0.118996   0.344958     -0.381470                                               Chad    base-gru   3829.0
2959          population ages 0-14   0.296798    0.168903   0.410978     -0.960854                                               Chad          rf   4303.0
2248          population ages 0-14   0.390866    0.229269   0.478820     -1.661667                                               Chad     xgboost   4998.0
114          population ages 15-64   0.297382    0.100648   0.317250      0.222855                                               Chad  simple-rnn   3365.0
825          population ages 15-64   0.466244    0.247868   0.497863     -0.913898                                               Chad   base-lstm   4983.0
2958         population ages 15-64   0.469141    0.340711   0.583705     -1.630783                                               Chad          rf   5535.0
1536         population ages 15-64   0.612107    0.418775   0.647128     -2.233550                                               Chad    base-gru   6123.0
2247         population ages 15-64   0.629274    0.540137   0.734940     -3.170639                                               Chad     xgboost   6620.0
2249  population ages 65 and above   0.096847    0.012173   0.110329     -0.677093                                               Chad     xgboost   2046.0
2960  population ages 65 and above   0.167502    0.035095   0.187337     -3.835255                                               Chad          rf   3427.0
116   population ages 65 and above   0.373770    0.147516   0.384078    -19.324196                                               Chad  simple-rnn   5591.0
1538  population ages 65 and above   0.422163    0.186936   0.432361    -24.755388                                               Chad    base-gru   6004.0
827   population ages 65 and above   0.470167    0.230621   0.480231    -30.774150                                               Chad   base-lstm   6334.0
2251          population ages 0-14   0.350395    0.135743   0.368434      0.880344                                              Chile     xgboost   3102.0
2962          population ages 0-14   0.567873    0.505133   0.710727      0.554731                                              Chile          rf   5204.0
1540          population ages 0-14   1.146992    1.482862   1.217728     -0.307127                                              Chile    base-gru   7827.0
118           population ages 0-14   1.584288    2.844316   1.686510     -1.507234                                              Chile  simple-rnn   9332.0
829           population ages 0-14   1.615163    3.024811   1.739198     -1.666338                                              Chile   base-lstm   9469.0
1539         population ages 15-64   0.255952    0.070811   0.266104     -2.155997                                              Chile    base-gru   3807.0
2250         population ages 15-64   0.250413    0.092455   0.304064     -3.120658                                              Chile     xgboost   4149.0
117          population ages 15-64   0.374811    0.197405   0.444303     -7.798180                                              Chile  simple-rnn   5526.0
2961         population ages 15-64   0.494296    0.346880   0.588966    -14.460194                                              Chile          rf   6501.0
828          population ages 15-64   0.739651    0.698662   0.835860    -30.138807                                              Chile   base-lstm   7996.0
2252  population ages 65 and above   0.385899    0.204836   0.452588      0.788574                                              Chile     xgboost   3727.0
2963  population ages 65 and above   0.433806    0.275750   0.525119      0.715379                                              Chile          rf   4198.0
119   population ages 65 and above   1.052898    1.195655   1.093460     -0.234119                                              Chile  simple-rnn   7379.0
830   population ages 65 and above   1.056349    1.204664   1.097572     -0.243417                                              Chile   base-lstm   7392.0
1541  population ages 65 and above   1.229238    1.629573   1.276547     -0.681996                                              Chile    base-gru   8157.0
2254          population ages 0-14   0.211919    0.093450   0.305695     -0.934069                                              China     xgboost   3547.0
2965          population ages 0-14   1.029155    1.423595   1.193145    -28.463246                                              China          rf   9202.0
1543          population ages 0-14   3.249613   12.035142   3.469170   -248.083777                                              China    base-gru  13117.0
832           population ages 0-14   3.310999   12.781377   3.575105   -263.528120                                              China   base-lstm  13195.0
121           population ages 0-14   3.430501   13.540728   3.679773   -279.243941                                              China  simple-rnn  13296.0
1542         population ages 15-64   2.186684    5.755840   2.399133     -2.156420                                              China    base-gru  10586.0
120          population ages 15-64   2.184175    5.905707   2.430166     -2.238605                                              China  simple-rnn  10628.0
831          population ages 15-64   2.963705   10.642566   3.262295     -4.836230                                              China   base-lstm  11783.0
2253         population ages 15-64   3.641441   19.295897   4.392710     -9.581593                                              China     xgboost  12747.0
2964         population ages 15-64   8.476897   96.533548   9.825149    -51.937611                                              China          rf  13694.0
2255  population ages 65 and above   0.262945    0.131316   0.362375      0.942472                                              China     xgboost   2668.0
2966  population ages 65 and above   0.837549    1.329047   1.152843      0.417761                                              China          rf   6910.0
833   population ages 65 and above   1.368450    1.915177   1.383899      0.160985                                              China   base-lstm   8106.0
1544  population ages 65 and above   1.644586    2.744226   1.656571     -0.202211                                              China    base-gru   8899.0
122   population ages 65 and above   1.730719    3.045974   1.745272     -0.334403                                              China  simple-rnn   9129.0
2257          population ages 0-14   0.097125    0.015689   0.125255      0.995839                                           Colombia     xgboost    699.0
2968          population ages 0-14   0.208539    0.065208   0.255359      0.982707                                           Colombia          rf   1839.0
1546          population ages 0-14   1.876849    3.943152   1.985737     -0.045703                                           Colombia    base-gru   9408.0
835           population ages 0-14   1.928634    4.231593   2.057084     -0.122197                                           Colombia   base-lstm   9563.0
124           population ages 0-14   2.094202    4.876549   2.208291     -0.293236                                           Colombia  simple-rnn   9855.0
2967         population ages 15-64   0.027653    0.001661   0.040749      0.998312                                           Colombia          rf     99.0
2256         population ages 15-64   0.308670    0.128689   0.358733      0.869203                                           Colombia     xgboost   2965.0
1545         population ages 15-64   1.725928    3.300970   1.816857     -2.355019                                           Colombia    base-gru   9804.0
123          population ages 15-64   2.010097    4.424292   2.103400     -3.496734                                           Colombia  simple-rnn  10517.0
834          population ages 15-64   2.248127    5.729052   2.393544     -4.822857                                           Colombia   base-lstm  10957.0
2258  population ages 65 and above   0.308778    0.141942   0.376752      0.845120                                           Colombia     xgboost   3101.0
836   population ages 65 and above   0.366616    0.143665   0.379032      0.843240                                           Colombia   base-lstm   3245.0
2969  population ages 65 and above   0.506244    0.368435   0.606989      0.597981                                           Colombia          rf   4702.0
125   population ages 65 and above   0.551759    0.339999   0.583095      0.629009                                           Colombia  simple-rnn   4710.0
1547  population ages 65 and above   0.592481    0.378440   0.615175      0.587064                                           Colombia    base-gru   4912.0
2260          population ages 0-14   0.101900    0.023714   0.153993      0.959433                                            Comoros     xgboost   1035.0
2971          population ages 0-14   0.113828    0.032426   0.180072      0.944530                                            Comoros          rf   1295.0
127           population ages 0-14   0.178861    0.046253   0.215066      0.920876                                            Comoros  simple-rnn   1731.0
838           population ages 0-14   0.216660    0.055540   0.235669      0.904990                                            Comoros   base-lstm   2037.0
1549          population ages 0-14   0.232914    0.061850   0.248697      0.894194                                            Comoros    base-gru   2175.0
2259         population ages 15-64   0.093556    0.010914   0.104469      0.976970                                            Comoros     xgboost    651.0
2970         population ages 15-64   0.353479    0.222544   0.471746      0.530385                                            Comoros          rf   3981.0
837          population ages 15-64   1.032237    1.077335   1.037948     -1.273403                                            Comoros   base-lstm   7585.0
126          population ages 15-64   1.097501    1.215383   1.102444     -1.564713                                            Comoros  simple-rnn   7895.0
1548         population ages 15-64   1.230232    1.524239   1.234601     -2.216464                                            Comoros    base-gru   8511.0
2261  population ages 65 and above   0.027941    0.002790   0.052822      0.604120                                            Comoros     xgboost    916.0
2972  population ages 65 and above   0.065215    0.007413   0.086097     -0.051726                                            Comoros          rf   1558.0
839   population ages 65 and above   0.592979    0.369855   0.608157    -51.475721                                            Comoros   base-lstm   7144.0
128   population ages 65 and above   0.613765    0.400099   0.632534    -55.766771                                            Comoros  simple-rnn   7279.0
1550  population ages 65 and above   0.742382    0.586007   0.765511    -82.143626                                            Comoros    base-gru   7994.0
841           population ages 0-14   0.208889    0.050647   0.225049     -2.029091                                   Congo, Dem. Rep.   base-lstm   3427.0
130           population ages 0-14   0.201043    0.088732   0.297880     -4.306886                                   Congo, Dem. Rep.  simple-rnn   4116.0
2263          population ages 0-14   0.194864    0.094927   0.308103     -4.677390                                   Congo, Dem. Rep.     xgboost   4185.0
2974          population ages 0-14   0.262228    0.139604   0.373636     -7.349382                                   Congo, Dem. Rep.          rf   4926.0
1552          population ages 0-14   0.360807    0.149391   0.386512     -7.934765                                   Congo, Dem. Rep.    base-gru   5256.0
2973         population ages 15-64   0.103507    0.018965   0.137714     -1.437377                                   Congo, Dem. Rep.          rf   2471.0
129          population ages 15-64   0.120177    0.032535   0.180374     -3.181282                                   Congo, Dem. Rep.  simple-rnn   3142.0
2262         population ages 15-64   0.154689    0.032925   0.181452     -3.231430                                   Congo, Dem. Rep.     xgboost   3277.0
840          population ages 15-64   0.178525    0.041725   0.204268     -4.362440                                   Congo, Dem. Rep.   base-lstm   3589.0
1551         population ages 15-64   0.371129    0.152196   0.390123    -18.559925                                   Congo, Dem. Rep.    base-gru   5593.0
131   population ages 65 and above   0.086657    0.013364   0.115603     -6.236773                                   Congo, Dem. Rep.  simple-rnn   2938.0
2264  population ages 65 and above   0.086643    0.014188   0.119114     -6.682981                                   Congo, Dem. Rep.     xgboost   2979.0
1553  population ages 65 and above   0.119423    0.024608   0.156869    -12.325366                                   Congo, Dem. Rep.    base-gru   3520.0
2975  population ages 65 and above   0.131177    0.033122   0.181994    -16.935787                                   Congo, Dem. Rep.          rf   3869.0
842   population ages 65 and above   0.142007    0.032355   0.179875    -16.520573                                   Congo, Dem. Rep.   base-lstm   3877.0
2266          population ages 0-14   0.773222    0.828523   0.910232     -2.037605                                        Congo, Rep.     xgboost   7142.0
2977          population ages 0-14   1.933034    5.120587   2.262872    -17.773552                                        Congo, Rep.          rf  11247.0
1555          population ages 0-14   3.048530   11.007846   3.317807    -39.357943                                        Congo, Rep.    base-gru  12617.0
844           population ages 0-14   3.108322   11.460627   3.385355    -41.017966                                        Congo, Rep.   base-lstm  12686.0
133           population ages 0-14   3.455932   14.051828   3.748577    -50.518057                                        Congo, Rep.  simple-rnn  13049.0
2265         population ages 15-64   1.653794    3.679740   1.918265    -13.789428                                        Congo, Rep.     xgboost  10615.0
2976         population ages 15-64   1.893172    4.959984   2.227102    -18.934921                                        Congo, Rep.          rf  11220.0
1554         population ages 15-64   2.015891    5.055220   2.248382    -19.317690                                        Congo, Rep.    base-gru  11320.0
843          population ages 15-64   2.207080    6.073030   2.464352    -23.408420                                        Congo, Rep.   base-lstm  11608.0
132          population ages 15-64   2.566330    8.003461   2.829039    -31.167114                                        Congo, Rep.  simple-rnn  12108.0
2978  population ages 65 and above   0.030614    0.001098   0.033131      0.444781                                        Congo, Rep.          rf   1001.0
2267  population ages 65 and above   0.080360    0.008032   0.089623     -3.062795                                        Congo, Rep.     xgboost   2448.0
134   population ages 65 and above   0.939607    0.957805   0.978675   -483.462205                                        Congo, Rep.  simple-rnn   9110.0
845   population ages 65 and above   0.948678    0.966019   0.982863   -487.616840                                        Congo, Rep.   base-lstm   9140.0
1556  population ages 65 and above   1.077058    1.248483   1.117355   -630.488584                                        Congo, Rep.    base-gru   9643.0
2980          population ages 0-14   0.111854    0.018420   0.135720      0.990989                                         Costa Rica          rf    834.0
2269          population ages 0-14   0.209697    0.059582   0.244094      0.970852                                         Costa Rica     xgboost   1872.0
1558          population ages 0-14   1.685052    3.233778   1.798271     -0.581993                                         Costa Rica    base-gru   9275.0
847           population ages 0-14   1.768536    3.666684   1.914859     -0.793774                                         Costa Rica   base-lstm   9536.0
136           population ages 0-14   1.914703    4.184698   2.045653     -1.047192                                         Costa Rica  simple-rnn   9863.0
2979         population ages 15-64   0.420832    0.290714   0.539179     -0.543916                                         Costa Rica          rf   4937.0
2268         population ages 15-64   0.532295    0.488996   0.699283     -1.596944                                         Costa Rica     xgboost   6008.0
1557         population ages 15-64   0.861109    0.861271   0.928047     -3.574006                                         Costa Rica    base-gru   7571.0
135          population ages 15-64   1.101706    1.415562   1.189774     -6.517716                                         Costa Rica  simple-rnn   8739.0
846          population ages 15-64   1.545456    2.850551   1.688358    -14.138607                                         Costa Rica   base-lstm  10244.0
2270  population ages 65 and above   0.380669    0.257602   0.507545      0.755735                                         Costa Rica     xgboost   3972.0
2981  population ages 65 and above   0.446430    0.321325   0.566855      0.695311                                         Costa Rica          rf   4391.0
848   population ages 65 and above   0.596511    0.376030   0.613213      0.643439                                         Costa Rica   base-lstm   4861.0
137   population ages 65 and above   0.893463    0.867012   0.931135      0.177877                                         Costa Rica  simple-rnn   6602.0
1559  population ages 65 and above   0.920488    0.911348   0.954646      0.135837                                         Costa Rica    base-gru   6704.0
2983          population ages 0-14   0.293755    0.126397   0.355524      0.797981                                      Cote d'Ivoire          rf   3060.0
2272          population ages 0-14   0.898027    1.004275   1.002135     -0.605118                                      Cote d'Ivoire     xgboost   7151.0
1561          population ages 0-14   1.271900    1.948428   1.395861     -2.114144                                      Cote d'Ivoire    base-gru   8815.0
850           population ages 0-14   1.318385    2.107096   1.451584     -2.367740                                      Cote d'Ivoire   base-lstm   8997.0
139           population ages 0-14   1.557787    2.878269   1.696546     -3.600296                                      Cote d'Ivoire  simple-rnn   9733.0
2271         population ages 15-64   0.338651    0.163469   0.404313      0.756501                                      Cote d'Ivoire     xgboost   3472.0
2982         population ages 15-64   0.428121    0.256821   0.506776      0.617446                                      Cote d'Ivoire          rf   4206.0
1560         population ages 15-64   0.564490    0.393290   0.627128      0.414166                                      Cote d'Ivoire    base-gru   5030.0
849          population ages 15-64   0.661800    0.549146   0.741044      0.182008                                      Cote d'Ivoire   base-lstm   5719.0
138          population ages 15-64   0.851416    0.893169   0.945076     -0.330440                                      Cote d'Ivoire  simple-rnn   6820.0
2273  population ages 65 and above   0.086507    0.009475   0.097337     -2.986717                                      Cote d'Ivoire     xgboost   2487.0
2984  population ages 65 and above   0.237126    0.081967   0.286299    -33.490280                                      Cote d'Ivoire          rf   4924.0
140   population ages 65 and above   0.613802    0.446293   0.668052   -186.791608                                      Cote d'Ivoire  simple-rnn   7609.0
851   population ages 65 and above   0.658593    0.505045   0.710665   -211.513037                                      Cote d'Ivoire   base-lstm   7839.0
1562  population ages 65 and above   0.719042    0.600084   0.774651   -251.503730                                      Cote d'Ivoire    base-gru   8167.0
2275          population ages 0-14   0.046718    0.003035   0.055092      0.216692                                            Croatia     xgboost   1233.0
2986          population ages 0-14   0.040304    0.003634   0.060281      0.062200                                            Croatia          rf   1327.0
1564          population ages 0-14   1.951360    3.938699   1.984616  -1015.496975                                            Croatia    base-gru  11758.0
853           population ages 0-14   2.298920    5.496819   2.344530  -1417.615742                                            Croatia   base-lstm  12244.0
142           population ages 0-14   2.667646    7.361679   2.713241  -1898.897611                                            Croatia  simple-rnn  12669.0
2985         population ages 15-64   0.310879    0.138897   0.372688      0.715578                                            Croatia          rf   3315.0
2274         population ages 15-64   0.380912    0.259701   0.509609      0.468204                                            Croatia     xgboost   4260.0
1563         population ages 15-64   2.001423    4.618681   2.149112     -8.457784                                            Croatia    base-gru  10926.0
141          population ages 15-64   2.587618    7.527468   2.743623    -14.414175                                            Croatia  simple-rnn  11794.0
852          population ages 15-64   3.137675   10.776667   3.282783    -21.067636                                            Croatia   base-lstm  12442.0
2987  population ages 65 and above   0.535611    0.473929   0.688425      0.179083                                            Croatia          rf   5309.0
2276  population ages 65 and above   0.794477    1.120292   1.058438     -0.940513                                            Croatia     xgboost   7231.0
1565  population ages 65 and above   0.924451    1.037039   1.018351     -0.796307                                            Croatia    base-gru   7263.0
143   population ages 65 and above   1.081377    1.350447   1.162088     -1.339177                                            Croatia  simple-rnn   7957.0
854   population ages 65 and above   1.117295    1.447090   1.202950     -1.506576                                            Croatia   base-lstm   8132.0
2989          population ages 0-14   0.463530    0.383037   0.618900     -0.111488                                               Cuba          rf   5106.0
2278          population ages 0-14   0.911735    1.188653   1.090254     -2.449202                                               Cuba     xgboost   7848.0
1567          population ages 0-14   2.167648    4.888631   2.211025    -13.185696                                               Cuba    base-gru  11203.0
856           population ages 0-14   2.509034    6.650146   2.578788    -18.297214                                               Cuba   base-lstm  11722.0
145           population ages 0-14   2.668966    7.486393   2.736127    -20.723813                                               Cuba  simple-rnn  11966.0
2277         population ages 15-64   0.474721    0.335047   0.578832     -2.927113                                               Cuba     xgboost   5769.0
1566         population ages 15-64   0.577288    0.339371   0.582556     -2.977801                                               Cuba    base-gru   5988.0
2988         population ages 15-64   0.596081    0.534096   0.730819     -5.260192                                               Cuba          rf   6762.0
144          population ages 15-64   1.163087    1.367456   1.169383    -15.028084                                               Cuba  simple-rnn   9089.0
855          population ages 15-64   1.732955    3.086408   1.756818    -35.176089                                               Cuba   base-lstm  10786.0
2279  population ages 65 and above   0.057769    0.008428   0.091802      0.988855                                               Cuba     xgboost    419.0
2990  population ages 65 and above   0.176853    0.058745   0.242373      0.922313                                               Cuba          rf   1901.0
857   population ages 65 and above   0.728260    0.620828   0.787926      0.178984                                               Cuba   base-lstm   5965.0
146   population ages 65 and above   0.837456    0.823069   0.907231     -0.088469                                               Cuba  simple-rnn   6606.0
1568  population ages 65 and above   1.061676    1.278565   1.130737     -0.690842                                               Cuba    base-gru   7661.0
2281          population ages 0-14   0.115616    0.016868   0.129878      0.134802                                             Cyprus     xgboost   1869.0
2992          population ages 0-14   0.383726    0.208980   0.457143     -9.718861                                             Cyprus          rf   5673.0
1570          population ages 0-14   1.520705    2.816135   1.678134   -143.443359                                             Cyprus    base-gru  10820.0
148           population ages 0-14   1.956367    4.584462   2.141136   -234.143250                                             Cyprus  simple-rnn  11735.0
859           population ages 0-14   1.987431    4.819301   2.195291   -246.188439                                             Cyprus   base-lstm  11810.0
2280         population ages 15-64   0.236148    0.111850   0.334439      0.785797                                             Cyprus     xgboost   2809.0
1569         population ages 15-64   0.610607    0.498186   0.705823      0.045924                                             Cyprus    base-gru   5586.0
147          population ages 15-64   1.009677    1.360012   1.166196     -1.604558                                             Cyprus  simple-rnn   7967.0
2991         population ages 15-64   0.964439    1.429756   1.195724     -1.738125                                             Cyprus          rf   8004.0
858          population ages 15-64   1.624673    3.264779   1.806870     -5.252375                                             Cyprus   base-lstm  10081.0
2282  population ages 65 and above   0.132778    0.031909   0.178630      0.955640                                             Cyprus     xgboost   1314.0
2993  population ages 65 and above   0.407536    0.258893   0.508815      0.640079                                             Cyprus          rf   4154.0
860   population ages 65 and above   1.026673    1.097616   1.047672     -0.525938                                             Cyprus   base-lstm   7353.0
149   population ages 65 and above   1.037474    1.127431   1.061806     -0.567387                                             Cyprus  simple-rnn   7422.0
1571  population ages 65 and above   1.147735    1.371777   1.171229     -0.907084                                             Cyprus    base-gru   7918.0
2995          population ages 0-14   0.215761    0.075336   0.274475     -0.782179                                            Czechia          rf   3352.0
2284          population ages 0-14   0.447607    0.245266   0.495243     -4.802080                                            Czechia     xgboost   5646.0
1573          population ages 0-14   2.927907    8.881996   2.980268   -209.114823                                            Czechia    base-gru  12656.0
862           population ages 0-14   3.297837   11.335233   3.366784   -267.149244                                            Czechia   base-lstm  13077.0
151           population ages 0-14   3.585490   13.347399   3.653409   -314.749567                                            Czechia  simple-rnn  13329.0
2283         population ages 15-64   0.039895    0.002393   0.048920      0.996719                                            Czechia     xgboost    165.0
2994         population ages 15-64   0.541756    0.392453   0.626461      0.461887                                            Czechia          rf   4955.0
1572         population ages 15-64   2.996450    9.285842   3.047268    -11.732291                                            Czechia    base-gru  12018.0
150          population ages 15-64   3.678013   13.977112   3.738598    -18.164729                                            Czechia  simple-rnn  12781.0
861          population ages 15-64   4.209928   18.259349   4.273096    -24.036322                                            Czechia   base-lstm  13131.0
2996  population ages 65 and above   0.177719    0.037245   0.192991      0.911604                                            Czechia          rf   1669.0
2285  population ages 65 and above   0.231509    0.098557   0.313939      0.766090                                            Czechia     xgboost   2720.0
1574  population ages 65 and above   0.949145    0.917825   0.958032     -1.178309                                            Czechia    base-gru   7265.0
152   population ages 65 and above   1.142251    1.322960   1.150200     -2.139828                                            Czechia  simple-rnn   8184.0
863   population ages 65 and above   1.192643    1.447267   1.203024     -2.434851                                            Czechia   base-lstm   8434.0
2287          population ages 0-14   0.519234    0.432708   0.657805     -0.012242                                            Denmark     xgboost   5269.0
2998          population ages 0-14   0.737624    0.785156   0.886090     -0.836731                                            Denmark          rf   6697.0
1576          population ages 0-14   3.367238   11.707228   3.421583    -26.386933                                            Denmark    base-gru  12673.0
865           population ages 0-14   3.818969   15.130148   3.889749    -34.394232                                            Denmark   base-lstm  13071.0
154           population ages 0-14   3.990547   16.567615   4.070333    -37.756925                                            Denmark  simple-rnn  13192.0
2997         population ages 15-64   0.358907    0.158392   0.397986      0.613129                                            Denmark          rf   3624.0
2286         population ages 15-64   0.565445    0.513137   0.716336     -0.253330                                            Denmark     xgboost   5689.0
1575         population ages 15-64   2.121923    5.235822   2.288192    -11.788415                                            Denmark    base-gru  11199.0
153          population ages 15-64   2.786105    8.892999   2.982113    -20.721015                                            Denmark  simple-rnn  12119.0
864          population ages 15-64   3.517134   13.937312   3.733271    -33.041673                                            Denmark   base-lstm  12944.0
1577  population ages 65 and above   0.426402    0.204947   0.452711      0.877151                                            Denmark    base-gru   3632.0
155   population ages 65 and above   0.590095    0.395516   0.628901      0.762921                                            Denmark  simple-rnn   4767.0
866   population ages 65 and above   0.712739    0.595946   0.771975      0.642780                                            Denmark   base-lstm   5580.0
2999  population ages 65 and above   1.057019    1.388840   1.178491      0.167505                                            Denmark          rf   7389.0
2288  population ages 65 and above   1.048797    1.756545   1.325347     -0.052903                                            Denmark     xgboost   7799.0
3001          population ages 0-14   0.118330    0.028388   0.168489      0.956299                                           Dominica          rf   1189.0
2290          population ages 0-14   0.229538    0.072504   0.269265      0.888388                                           Dominica     xgboost   2263.0
1579          population ages 0-14   2.010219    4.363649   2.088935     -5.717397                                           Dominica    base-gru  10722.0
157           population ages 0-14   2.251881    5.483985   2.341791     -7.442041                                           Dominica  simple-rnn  11127.0
868           population ages 0-14   2.350637    5.995491   2.448569     -8.229451                                           Dominica   base-lstm  11274.0
2289         population ages 15-64   0.731308    0.690156   0.830756      0.552978                                           Dominica     xgboost   5881.0
3000         population ages 15-64   0.834816    0.896135   0.946644      0.419564                                           Dominica          rf   6409.0
867          population ages 15-64   1.919698    3.875133   1.968536     -1.509965                                           Dominica   base-lstm   9923.0
156          population ages 15-64   2.383806    5.989883   2.447424     -2.879710                                           Dominica  simple-rnn  10815.0
1578         population ages 15-64   2.612574    7.271773   2.696623     -3.710004                                           Dominica    base-gru  11225.0
2291  population ages 65 and above   0.872072    1.073663   1.036177     -3.502828                                           Dominica     xgboost   7833.0
3002  population ages 65 and above   1.077421    1.522504   1.233898     -5.385222                                           Dominica          rf   8737.0
158   population ages 65 and above   3.800279   15.975288   3.996910    -65.998678                                           Dominica  simple-rnn  13263.0
869   population ages 65 and above   3.806409   16.016292   4.002036    -66.170645                                           Dominica   base-lstm  13272.0
1580  population ages 65 and above   3.983482   17.482333   4.181188    -72.319069                                           Dominica    base-gru  13357.0
2293          population ages 0-14   0.111527    0.020923   0.144648      0.983788                                 Dominican Republic     xgboost    912.0
3004          population ages 0-14   0.551833    0.447713   0.669114      0.653083                                 Dominican Republic          rf   4944.0
1582          population ages 0-14   3.038582   10.538469   3.246301     -7.165886                                 Dominican Republic    base-gru  11968.0
871           population ages 0-14   3.054207   10.754196   3.279359     -7.333045                                 Dominican Republic   base-lstm  12015.0
160           population ages 0-14   3.370657   12.905694   3.592450     -9.000165                                 Dominican Republic  simple-rnn  12382.0
2292         population ages 15-64   0.162626    0.054736   0.233957      0.861325                                 Dominican Republic     xgboost   1957.0
3003         population ages 15-64   0.830831    1.066750   1.032836     -1.702642                                 Dominican Republic          rf   7443.0
1581         population ages 15-64   1.382494    2.295615   1.515129     -4.816009                                 Dominican Republic    base-gru   9466.0
159          population ages 15-64   1.830280    3.949042   1.987220     -9.005016                                 Dominican Republic  simple-rnn  10680.0
870          population ages 15-64   1.933550    4.572675   2.138381    -10.585007                                 Dominican Republic   base-lstm  10956.0
3005  population ages 65 and above   0.078815    0.008186   0.090477      0.971115                                 Dominican Republic          rf    562.0
2294  population ages 65 and above   0.139585    0.042227   0.205493      0.851000                                 Dominican Republic     xgboost   1732.0
872   population ages 65 and above   1.117397    1.363865   1.167846     -3.812441                                 Dominican Republic   base-lstm   8488.0
161   population ages 65 and above   1.377460    2.109042   1.452254     -6.441821                                 Dominican Republic  simple-rnn   9494.0
1583  population ages 65 and above   1.490248    2.441336   1.562478     -7.614331                                 Dominican Republic    base-gru   9830.0
3007          population ages 0-14   0.098061    0.011480   0.107144      0.993056                         Early-demographic dividend          rf    616.0
2296          population ages 0-14   0.115642    0.016602   0.128849      0.989957                         Early-demographic dividend     xgboost    808.0
1585          population ages 0-14   2.359669    6.400010   2.529824     -2.871321                         Early-demographic dividend    base-gru  10865.0
874           population ages 0-14   2.428428    6.825549   2.612575     -3.128727                         Early-demographic dividend   base-lstm  11011.0
163           population ages 0-14   2.576420    7.589237   2.754857     -3.590677                         Early-demographic dividend  simple-rnn  11251.0
2295         population ages 15-64   0.081336    0.007665   0.087550      0.989226                         Early-demographic dividend     xgboost    469.0
3006         population ages 15-64   0.175376    0.039006   0.197499      0.945172                         Early-demographic dividend          rf   1590.0
1584         population ages 15-64   0.961992    1.122260   1.059368     -0.577494                         Early-demographic dividend    base-gru   7332.0
162          population ages 15-64   1.262433    1.896559   1.377156     -1.665878                         Early-demographic dividend  simple-rnn   8685.0
873          population ages 15-64   1.359693    2.286593   1.512149     -2.214126                         Early-demographic dividend   base-lstm   9096.0
2297  population ages 65 and above   0.081015    0.009329   0.096586      0.953827                         Early-demographic dividend     xgboost    657.0
3008  population ages 65 and above   0.097036    0.020980   0.144844      0.896160                         Early-demographic dividend          rf   1163.0
875   population ages 65 and above   1.117597    1.382958   1.175992     -5.844917                         Early-demographic dividend   base-lstm   8683.0
164   population ages 65 and above   1.261225    1.782982   1.335283     -7.824823                         Early-demographic dividend  simple-rnn   9293.0
1586  population ages 65 and above   1.382744    2.118971   1.455669     -9.487793                         Early-demographic dividend    base-gru   9660.0
2299          population ages 0-14   0.208440    0.068324   0.261388      0.583868                                East Asia & Pacific     xgboost   2575.0
3010          population ages 0-14   0.440904    0.269673   0.519300     -0.642464                                East Asia & Pacific          rf   4956.0
1588          population ages 0-14   2.884337    9.431921   3.071143    -56.445892                                East Asia & Pacific    base-gru  12490.0
877           population ages 0-14   3.032457   10.642440   3.262275    -63.818657                                East Asia & Pacific   base-lstm  12669.0
166           population ages 0-14   3.111437   11.066599   3.326650    -66.402032                                East Asia & Pacific  simple-rnn  12765.0
3009         population ages 15-64   0.131635    0.024446   0.156351      0.965586                                East Asia & Pacific          rf   1126.0
2298         population ages 15-64   0.343926    0.176726   0.420388      0.751214                                East Asia & Pacific     xgboost   3557.0
1587         population ages 15-64   1.464824    2.615762   1.617332     -2.682338                                East Asia & Pacific    base-gru   9425.0
165          population ages 15-64   1.638204    3.339392   1.827400     -3.701028                                East Asia & Pacific  simple-rnn   9984.0
876          population ages 15-64   2.273785    6.284287   2.506848     -7.846702                                East Asia & Pacific   base-lstm  11262.0
2300  population ages 65 and above   0.292184    0.147090   0.383523      0.902922                                East Asia & Pacific     xgboost   2981.0
3011  population ages 65 and above   0.525797    0.557704   0.746796      0.631920                                East Asia & Pacific          rf   5185.0
878   population ages 65 and above   1.259354    1.619630   1.272647     -0.068942                                East Asia & Pacific   base-lstm   7937.0
167   population ages 65 and above   1.545689    2.461107   1.568791     -0.624309                                East Asia & Pacific  simple-rnn   8890.0
1589  population ages 65 and above   1.570898    2.538852   1.593377     -0.675620                                East Asia & Pacific    base-gru   8960.0
3013          population ages 0-14   0.530470    0.397768   0.630688     -1.797637         East Asia & Pacific (IDA & IBRD countries)          rf   5825.0
2302          population ages 0-14   0.584709    0.553959   0.744285     -2.896186         East Asia & Pacific (IDA & IBRD countries)     xgboost   6515.0
1591          population ages 0-14   3.356436   12.790985   3.576449    -88.963371         East Asia & Pacific (IDA & IBRD countries)    base-gru  13043.0
880           population ages 0-14   3.429363   13.601659   3.688043    -94.665119         East Asia & Pacific (IDA & IBRD countries)   base-lstm  13138.0
169           population ages 0-14   3.531200   14.241316   3.773767    -99.164046         East Asia & Pacific (IDA & IBRD countries)  simple-rnn  13222.0
3012         population ages 15-64   0.365478    0.177160   0.420904      0.742329         East Asia & Pacific (IDA & IBRD countries)          rf   3613.0
2301         population ages 15-64   0.588222    0.511207   0.714987      0.256473         East Asia & Pacific (IDA & IBRD countries)     xgboost   5460.0
1590         population ages 15-64   2.006032    4.770419   2.184129     -5.938358         East Asia & Pacific (IDA & IBRD countries)    base-gru  10810.0
168          population ages 15-64   2.104718    5.311656   2.304703     -6.725563         East Asia & Pacific (IDA & IBRD countries)  simple-rnn  11000.0
879          population ages 15-64   2.709388    8.808871   2.967974    -11.812105         East Asia & Pacific (IDA & IBRD countries)   base-lstm  11878.0
2303  population ages 65 and above   0.311727    0.193116   0.439449      0.862795         East Asia & Pacific (IDA & IBRD countries)     xgboost   3342.0
3014  population ages 65 and above   0.527500    0.606713   0.778918      0.568942         East Asia & Pacific (IDA & IBRD countries)          rf   5346.0
881   population ages 65 and above   1.452145    2.156036   1.468345     -0.531821         East Asia & Pacific (IDA & IBRD countries)   base-lstm   8630.0
1592  population ages 65 and above   1.731306    3.083368   1.755952     -1.190672         East Asia & Pacific (IDA & IBRD countries)    base-gru   9445.0
170   population ages 65 and above   1.761347    3.203905   1.789946     -1.276311         East Asia & Pacific (IDA & IBRD countries)  simple-rnn   9542.0
3016          population ages 0-14   0.515837    0.380735   0.617038     -1.581849        East Asia & Pacific (excluding high income)          rf   5699.0
2305          population ages 0-14   0.591582    0.560596   0.748730     -2.801526        East Asia & Pacific (excluding high income)     xgboost   6529.0
1594          population ages 0-14   3.334388   12.626613   3.553395    -84.623816        East Asia & Pacific (excluding high income)    base-gru  13020.0
883           population ages 0-14   3.410970   13.459690   3.668745    -90.273096        East Asia & Pacific (excluding high income)   base-lstm  13104.0
172           population ages 0-14   3.512574   14.094538   3.754269    -94.578134        East Asia & Pacific (excluding high income)  simple-rnn  13198.0
3015         population ages 15-64   0.361497    0.171592   0.414237      0.740982        East Asia & Pacific (excluding high income)          rf   3581.0
2304         population ages 15-64   0.586044    0.504600   0.710352      0.238307        East Asia & Pacific (excluding high income)     xgboost   5448.0
1593         population ages 15-64   1.984551    4.659442   2.158574     -6.033416        East Asia & Pacific (excluding high income)    base-gru  10782.0
171          population ages 15-64   2.089232    5.221298   2.285016     -6.881536        East Asia & Pacific (excluding high income)  simple-rnn  10984.0
882          population ages 15-64   2.691420    8.676673   2.945619    -12.097417        East Asia & Pacific (excluding high income)   base-lstm  11867.0
2306  population ages 65 and above   0.291960    0.191872   0.438032      0.861800        East Asia & Pacific (excluding high income)     xgboost   3289.0
3017  population ages 65 and above   0.544196    0.622368   0.788903      0.551725        East Asia & Pacific (excluding high income)          rf   5418.0
884   population ages 65 and above   1.446383    2.141457   1.463372     -0.542434        East Asia & Pacific (excluding high income)   base-lstm   8617.0
1595  population ages 65 and above   1.725255    3.066164   1.751047     -1.208476        East Asia & Pacific (excluding high income)    base-gru   9439.0
173   population ages 65 and above   1.753445    3.179785   1.783195     -1.290314        East Asia & Pacific (excluding high income)  simple-rnn   9535.0
3019          population ages 0-14   0.100786    0.017127   0.130869      0.994421                                            Ecuador          rf    758.0
2308          population ages 0-14   0.253770    0.088533   0.297544      0.971162                                            Ecuador     xgboost   2232.0
1597          population ages 0-14   1.822504    3.614311   1.901134     -0.177303                                            Ecuador    base-gru   9317.0
886           population ages 0-14   2.027425    4.542810   2.131387     -0.479747                                            Ecuador   base-lstm   9838.0
175           population ages 0-14   2.190291    5.250581   2.291415     -0.710292                                            Ecuador  simple-rnn  10118.0
2307         population ages 15-64   0.090734    0.011443   0.106971      0.990983                                            Ecuador     xgboost    591.0
3018         population ages 15-64   0.397920    0.195858   0.442559      0.845664                                            Ecuador          rf   3585.0
1596         population ages 15-64   0.988252    1.048052   1.023744      0.174135                                            Ecuador    base-gru   6934.0
174          population ages 15-64   1.506527    2.460429   1.568576     -0.938819                                            Ecuador  simple-rnn   8952.0
885          population ages 15-64   1.543228    2.643093   1.625759     -1.082758                                            Ecuador   base-lstm   9118.0
3020  population ages 65 and above   0.219559    0.074365   0.272699      0.810438                                            Ecuador          rf   2397.0
2309  population ages 65 and above   0.331533    0.174809   0.418101      0.554397                                            Ecuador     xgboost   3702.0
887   population ages 65 and above   0.568207    0.380905   0.617175      0.029038                                            Ecuador   base-lstm   5227.0
176   population ages 65 and above   0.649244    0.502592   0.708937     -0.281155                                            Ecuador  simple-rnn   5817.0
1598  population ages 65 and above   0.808216    0.750380   0.866245     -0.912790                                            Ecuador    base-gru   6785.0
2311          population ages 0-14   0.588432    0.433881   0.658696    -23.989710                                   Egypt, Arab Rep.     xgboost   7084.0
3022          population ages 0-14   1.392786    2.545025   1.595313   -145.582802                                   Egypt, Arab Rep.          rf  10644.0
1600          population ages 0-14   2.956627    9.860872   3.140203   -566.945129                                   Egypt, Arab Rep.    base-gru  12938.0
889           population ages 0-14   2.991944   10.158643   3.187263   -584.095489                                   Egypt, Arab Rep.   base-lstm  12978.0
178           population ages 0-14   3.250494   11.951156   3.457044   -687.336767                                   Egypt, Arab Rep.  simple-rnn  13255.0
2310         population ages 15-64   0.290218    0.112518   0.335437     -1.320338                                   Egypt, Arab Rep.     xgboost   4036.0
1599         population ages 15-64   1.307340    2.062582   1.436169    -41.534347                                   Egypt, Arab Rep.    base-gru  10049.0
888          population ages 15-64   1.554825    3.000679   1.732247    -60.879685                                   Egypt, Arab Rep.   base-lstm  10751.0
3021         population ages 15-64   1.574640    3.392275   1.841813    -68.955137                                   Egypt, Arab Rep.          rf  10945.0
177          population ages 15-64   1.786739    3.842386   1.960200    -78.237278                                   Egypt, Arab Rep.  simple-rnn  11260.0
2312  population ages 65 and above   0.137933    0.026807   0.163729      0.362454                                   Egypt, Arab Rep.     xgboost   2034.0
3023  population ages 65 and above   0.228670    0.085138   0.291784     -1.024794                                   Egypt, Arab Rep.          rf   3541.0
890   population ages 65 and above   1.519589    2.444589   1.563518    -57.138665                                   Egypt, Arab Rep.   base-lstm  10499.0
179   population ages 65 and above   1.560393    2.598469   1.611977    -60.798324                                   Egypt, Arab Rep.  simple-rnn  10617.0
1601  population ages 65 and above   1.715905    3.131660   1.769650    -73.478980                                   Egypt, Arab Rep.    base-gru  10966.0
2314          population ages 0-14   0.234062    0.110657   0.332652      0.971082                                        El Salvador     xgboost   2335.0
3025          population ages 0-14   0.313885    0.145946   0.382029      0.961860                                        El Salvador          rf   2843.0
1603          population ages 0-14   2.259059    6.122298   2.474328     -0.599935                                        El Salvador    base-gru  10231.0
892           population ages 0-14   2.429857    7.050436   2.655266     -0.842485                                        El Salvador   base-lstm  10504.0
181           population ages 0-14   2.505110    7.377406   2.716138     -0.927931                                        El Salvador  simple-rnn  10617.0
3024         population ages 15-64   0.249674    0.093919   0.306461      0.960440                                        El Salvador          rf   2298.0
1602         population ages 15-64   0.488176    0.360936   0.600779      0.847969                                        El Salvador    base-gru   4319.0
180          population ages 15-64   0.784891    0.803452   0.896355      0.661576                                        El Salvador  simple-rnn   6022.0
2313         population ages 15-64   0.810997    1.084584   1.041434      0.543160                                        El Salvador     xgboost   6520.0
891          population ages 15-64   1.089301    1.544459   1.242763      0.349454                                        El Salvador   base-lstm   7479.0
3026  population ages 65 and above   0.312992    0.120049   0.346481      0.354553                                        El Salvador          rf   3468.0
2315  population ages 65 and above   0.373346    0.220474   0.469547     -0.185388                                        El Salvador     xgboost   4440.0
893   population ages 65 and above   0.942770    1.178165   1.085433     -5.334441                                        El Salvador   base-lstm   8211.0
182   population ages 65 and above   1.099483    1.589670   1.260821     -7.546912                                        El Salvador  simple-rnn   8966.0
1604  population ages 65 and above   1.286748    2.084393   1.443743    -10.206808                                        El Salvador    base-gru   9580.0
2317          population ages 0-14   0.138337    0.041955   0.204829     -3.225799                                  Equatorial Guinea     xgboost   3333.0
3028          population ages 0-14   0.394370    0.192601   0.438863    -18.399187                                  Equatorial Guinea          rf   5846.0
1606          population ages 0-14   0.895038    1.084611   1.041447   -108.244350                                  Equatorial Guinea    base-gru   9007.0
895           population ages 0-14   0.924929    1.157786   1.076005   -115.614667                                  Equatorial Guinea   base-lstm   9127.0
184           population ages 0-14   1.194710    1.832671   1.353762   -183.590600                                  Equatorial Guinea  simple-rnn  10094.0
1605         population ages 15-64   0.362860    0.159190   0.398987    -11.071106                                  Equatorial Guinea    base-gru   5422.0
894          population ages 15-64   0.454530    0.283217   0.532182    -20.475834                                  Equatorial Guinea   base-lstm   6363.0
183          population ages 15-64   0.580932    0.498373   0.705955    -36.790670                                  Equatorial Guinea  simple-rnn   7355.0
2316         population ages 15-64   0.674278    0.707872   0.841351    -52.676628                                  Equatorial Guinea     xgboost   8067.0
3027         population ages 15-64   0.926435    1.358764   1.165660   -102.032581                                  Equatorial Guinea          rf   9331.0
3029  population ages 65 and above   0.070107    0.007454   0.086337     -4.847808                                  Equatorial Guinea          rf   2610.0
2318  population ages 65 and above   0.120499    0.021621   0.147042    -15.962281                                  Equatorial Guinea     xgboost   3578.0
896   population ages 65 and above   0.662369    0.453957   0.673763   -355.136034                                  Equatorial Guinea   base-lstm   7791.0
185   population ages 65 and above   0.713757    0.528790   0.727180   -413.844279                                  Equatorial Guinea  simple-rnn   8080.0
1607  population ages 65 and above   0.799531    0.667520   0.817019   -522.679988                                  Equatorial Guinea    base-gru   8524.0
3031          population ages 0-14   0.027113    0.000992   0.031489      0.980743                                            Eritrea          rf    157.0
2320          population ages 0-14   0.144180    0.024685   0.157114      0.520590                                            Eritrea     xgboost   1889.0
898           population ages 0-14   0.564144    0.320391   0.566031     -5.222392                                            Eritrea   base-lstm   6149.0
1609          population ages 0-14   0.572378    0.329438   0.573967     -5.398101                                            Eritrea    base-gru   6220.0
187           population ages 0-14   0.773403    0.601778   0.775743    -10.687290                                            Eritrea  simple-rnn   7489.0
3030         population ages 15-64   0.070639    0.008831   0.093975      0.664827                                            Eritrea          rf   1146.0
2319         population ages 15-64   0.236426    0.100752   0.317415     -2.823807                                            Eritrea     xgboost   4123.0
1608         population ages 15-64   0.499507    0.251023   0.501022     -8.526998                                            Eritrea    base-gru   6004.0
897          population ages 15-64   0.568978    0.326417   0.571329    -11.388382                                            Eritrea   base-lstm   6495.0
186          population ages 15-64   0.676113    0.462460   0.680044    -16.551568                                            Eritrea  simple-rnn   7177.0
3032  population ages 65 and above   0.007527    0.000107   0.010355      0.974912                                            Eritrea          rf    115.0
1610  population ages 65 and above   0.011783    0.000182   0.013474      0.957523                                            Eritrea    base-gru    189.0
2321  population ages 65 and above   0.021255    0.000870   0.029491      0.796516                                            Eritrea     xgboost    619.0
899   population ages 65 and above   0.061024    0.003761   0.061325      0.120138                                            Eritrea   base-lstm   1343.0
188   population ages 65 and above   0.063462    0.004061   0.063723      0.049991                                            Eritrea  simple-rnn   1407.0
2323          population ages 0-14   0.170712    0.040046   0.200116     -2.037726                                            Estonia     xgboost   3194.0
3034          population ages 0-14   0.376010    0.183214   0.428035    -12.897703                                            Estonia          rf   5629.0
1612          population ages 0-14   3.062356    9.609197   3.099870   -727.906972                                            Estonia    base-gru  12985.0
901           population ages 0-14   3.467945   12.378750   3.518345   -937.991842                                            Estonia   base-lstm  13374.0
190           population ages 0-14   3.729190   14.281380   3.779071  -1082.316121                                            Estonia  simple-rnn  13582.0
2322         population ages 15-64   0.129470    0.029826   0.172702      0.887713                                            Estonia     xgboost   1457.0
3033         population ages 15-64   0.250808    0.089127   0.298542      0.664459                                            Estonia          rf   2824.0
1611         population ages 15-64   2.336827    5.741180   2.396076    -20.614084                                            Estonia    base-gru  11572.0
189          population ages 15-64   3.011118    9.499847   3.082182    -34.764509                                            Estonia  simple-rnn  12433.0
900          population ages 15-64   3.478230   12.611431   3.551258    -46.478829                                            Estonia   base-lstm  12929.0
3035  population ages 65 and above   0.114943    0.027007   0.164338      0.835218                                            Estonia          rf   1452.0
2324  population ages 65 and above   0.135323    0.034975   0.187017      0.786599                                            Estonia     xgboost   1777.0
1613  population ages 65 and above   0.245979    0.066879   0.258610      0.591942                                            Estonia    base-gru   2671.0
902   population ages 65 and above   0.424499    0.189444   0.435251     -0.155880                                            Estonia   base-lstm   4398.0
191   population ages 65 and above   0.433035    0.194767   0.441324     -0.188359                                            Estonia  simple-rnn   4466.0
3037          population ages 0-14   0.469862    0.270542   0.520136      0.789527                                           Eswatini          rf   4165.0
2326          population ages 0-14   0.720707    0.634241   0.796393      0.506580                                           Eswatini     xgboost   5784.0
904           population ages 0-14   2.933330    9.627501   3.102822     -6.489895                                           Eswatini   base-lstm  11823.0
193           population ages 0-14   3.133422   10.975560   3.312938     -7.538643                                           Eswatini  simple-rnn  12080.0
1615          population ages 0-14   3.196399   11.527689   3.395245     -7.968182                                           Eswatini    base-gru  12173.0
2325         population ages 15-64   0.438919    0.231055   0.480682      0.604323                                           Eswatini     xgboost   4138.0
3036         population ages 15-64   0.545365    0.362008   0.601671      0.380070                                           Eswatini          rf   4956.0
903          population ages 15-64   2.674047    8.113329   2.848391    -12.893870                                           Eswatini   base-lstm  11844.0
192          population ages 15-64   2.825157    8.978695   2.996447    -14.375787                                           Eswatini  simple-rnn  11998.0
1614         population ages 15-64   2.913821    9.591147   3.096958    -15.424595                                           Eswatini    base-gru  12125.0
2327  population ages 65 and above   0.200872    0.060098   0.245148      0.569017                                           Eswatini     xgboost   2512.0
3038  population ages 65 and above   0.234834    0.074657   0.273234      0.464605                                           Eswatini          rf   2813.0
905   population ages 65 and above   0.311713    0.107819   0.328358      0.226789                                           Eswatini   base-lstm   3450.0
194   population ages 65 and above   0.341071    0.136962   0.370084      0.017790                                           Eswatini  simple-rnn   3849.0
1616  population ages 65 and above   0.440732    0.215234   0.463933     -0.543528                                           Eswatini    base-gru   4717.0
3040          population ages 0-14   0.277459    0.103525   0.321752      0.961844                                           Ethiopia          rf   2450.0
196           population ages 0-14   0.322649    0.118952   0.344895      0.956158                                           Ethiopia  simple-rnn   2689.0
907           population ages 0-14   0.502686    0.272433   0.521951      0.899590                                           Ethiopia   base-lstm   3997.0
1618          population ages 0-14   0.686824    0.504660   0.710394      0.813998                                           Ethiopia    base-gru   5104.0
2329          population ages 0-14   1.233653    1.899285   1.378145      0.299984                                           Ethiopia     xgboost   7906.0
3039         population ages 15-64   0.505371    0.311483   0.558106      0.866601                                           Ethiopia          rf   4167.0
195          population ages 15-64   0.849187    0.771483   0.878341      0.669597                                           Ethiopia  simple-rnn   6050.0
906          population ages 15-64   0.949003    0.981843   0.990880      0.579506                                           Ethiopia   base-lstm   6534.0
2328         population ages 15-64   0.845631    1.106074   1.051700      0.526302                                           Ethiopia     xgboost   6603.0
1617         population ages 15-64   1.073273    1.266650   1.125455      0.457532                                           Ethiopia    base-gru   7112.0
3041  population ages 65 and above   0.043829    0.003150   0.056121      0.778445                                           Ethiopia          rf    782.0
2330  population ages 65 and above   0.074672    0.008372   0.091498      0.411086                                           Ethiopia     xgboost   1361.0
197   population ages 65 and above   0.352363    0.153940   0.392352     -9.828879                                           Ethiopia  simple-rnn   5328.0
908   population ages 65 and above   0.407004    0.196414   0.443186    -12.816683                                           Ethiopia   base-lstm   5760.0
1619  population ages 65 and above   0.410149    0.204859   0.452613    -13.410743                                           Ethiopia    base-gru   5817.0
3043          population ages 0-14   0.266094    0.118780   0.344645     -1.561200                                          Euro area          rf   4071.0
2332          population ages 0-14   0.383215    0.229650   0.479218     -3.951844                                          Euro area     xgboost   5380.0
1621          population ages 0-14   3.052173    9.701160   3.114669   -208.181899                                          Euro area    base-gru  12802.0
910           population ages 0-14   3.499523   12.810031   3.579110   -275.217127                                          Euro area   base-lstm  13259.0
199           population ages 0-14   3.940273   16.236553   4.029461   -349.101736                                          Euro area  simple-rnn  13575.0
2331         population ages 15-64   0.129625    0.022061   0.148529      0.969584                                          Euro area     xgboost   1075.0
3042         population ages 15-64   0.541664    0.396117   0.629379      0.453862                                          Euro area          rf   4973.0
1620         population ages 15-64   1.589582    3.219985   1.794432     -3.439482                                          Euro area    base-gru   9858.0
198          population ages 15-64   2.478626    7.336085   2.708521     -9.114461                                          Euro area  simple-rnn  11565.0
909          population ages 15-64   3.068806   10.848552   3.293714    -13.957196                                          Euro area   base-lstm  12275.0
3044  population ages 65 and above   0.147591    0.031922   0.178668      0.971784                                          Euro area          rf   1313.0
200   population ages 65 and above   0.272757    0.100562   0.317115      0.911112                                          Euro area  simple-rnn   2561.0
911   population ages 65 and above   0.306408    0.121811   0.349014      0.892330                                          Euro area   base-lstm   2856.0
1622  population ages 65 and above   0.386016    0.203989   0.451652      0.819691                                          Euro area    base-gru   3649.0
2333  population ages 65 and above   0.366855    0.287576   0.536261      0.745808                                          Euro area     xgboost   4049.0
2335          population ages 0-14   0.170870    0.048778   0.220858     -0.840791                              Europe & Central Asia     xgboost   2946.0
3046          population ages 0-14   0.725406    0.642864   0.801788    -23.260435                              Europe & Central Asia          rf   7778.0
1624          population ages 0-14   3.327911   12.183419   3.490476   -458.778338                              Europe & Central Asia    base-gru  13244.0
913           population ages 0-14   3.840544   16.233921   4.029134   -611.636336                              Europe & Central Asia   base-lstm  13644.0
202           population ages 0-14   3.954282   17.171392   4.143838   -647.014635                              Europe & Central Asia  simple-rnn  13690.0
1623         population ages 15-64   0.750841    0.835516   0.914066      0.195293                              Europe & Central Asia    base-gru   6353.0
3045         population ages 15-64   0.975015    1.747156   1.321800     -0.682731                              Europe & Central Asia          rf   7964.0
2334         population ages 15-64   1.149113    2.289247   1.513026     -1.204832                              Europe & Central Asia     xgboost   8659.0
201          population ages 15-64   1.258720    2.490688   1.578192     -1.398845                              Europe & Central Asia  simple-rnn   8930.0
912          population ages 15-64   1.763251    4.336095   2.082329     -3.176203                              Europe & Central Asia   base-lstm  10298.0
3047  population ages 65 and above   0.492386    0.381144   0.617369      0.485449                              Europe & Central Asia          rf   4816.0
2336  population ages 65 and above   1.065607    1.866123   1.366061     -1.519298                              Europe & Central Asia     xgboost   8400.0
914   population ages 65 and above   1.745236    3.087265   1.757061     -3.167861                              Europe & Central Asia   base-lstm   9876.0
203   population ages 65 and above   1.761243    3.152642   1.775568     -3.256121                              Europe & Central Asia  simple-rnn   9935.0
1625  population ages 65 and above   1.996554    4.056441   2.014061     -4.476266                              Europe & Central Asia    base-gru  10508.0
2338          population ages 0-14   0.157709    0.044590   0.211162      0.056682       Europe & Central Asia (IDA & IBRD countries)     xgboost   2501.0
3049          population ages 0-14   0.357670    0.184214   0.429202     -2.897163       Europe & Central Asia (IDA & IBRD countries)          rf   4964.0
1627          population ages 0-14   3.373650   11.765317   3.430061   -247.902302       Europe & Central Asia (IDA & IBRD countries)    base-gru  13130.0
916           population ages 0-14   3.672414   14.039917   3.746988   -296.022814       Europe & Central Asia (IDA & IBRD countries)   base-lstm  13397.0
205           population ages 0-14   3.692293   14.138494   3.760119   -298.108276       Europe & Central Asia (IDA & IBRD countries)  simple-rnn  13409.0
2337         population ages 15-64   0.354386    0.199365   0.446503      0.616280       Europe & Central Asia (IDA & IBRD countries)     xgboost   3822.0
3048         population ages 15-64   0.502227    0.391461   0.625669      0.246551       Europe & Central Asia (IDA & IBRD countries)          rf   4996.0
1626         population ages 15-64   1.526065    2.681318   1.637473     -4.160757       Europe & Central Asia (IDA & IBRD countries)    base-gru   9697.0
204          population ages 15-64   1.919683    4.209835   2.051788     -7.102709       Europe & Central Asia (IDA & IBRD countries)  simple-rnn  10708.0
915          population ages 15-64   2.300655    5.998535   2.449191    -10.545436       Europe & Central Asia (IDA & IBRD countries)   base-lstm  11340.0
2339  population ages 65 and above   0.093741    0.011157   0.105628      0.956769       Europe & Central Asia (IDA & IBRD countries)     xgboost    737.0
3050  population ages 65 and above   0.358432    0.161520   0.401895      0.374156       Europe & Central Asia (IDA & IBRD countries)          rf   3847.0
917   population ages 65 and above   1.290715    1.675085   1.294251     -5.490488       Europe & Central Asia (IDA & IBRD countries)   base-lstm   9077.0
206   population ages 65 and above   1.308364    1.720894   1.311829     -5.667985       Europe & Central Asia (IDA & IBRD countries)  simple-rnn   9142.0
1628  population ages 65 and above   1.485040    2.215471   1.488446     -7.584330       Europe & Central Asia (IDA & IBRD countries)    base-gru   9693.0
2341          population ages 0-14   0.235561    0.082504   0.287236     -2.639606      Europe & Central Asia (excluding high income)     xgboost   3920.0
3052          population ages 0-14   0.274127    0.107716   0.328202     -3.751798      Europe & Central Asia (excluding high income)          rf   4426.0
1630          population ages 0-14   3.354282   11.795785   3.434499   -519.359600      Europe & Central Asia (excluding high income)    base-gru  13233.0
919           population ages 0-14   3.557014   13.356653   3.654675   -588.215782      Europe & Central Asia (excluding high income)   base-lstm  13430.0
208           population ages 0-14   3.588239   13.536759   3.679233   -596.160983      Europe & Central Asia (excluding high income)  simple-rnn  13452.0
2340         population ages 15-64   0.054642    0.004004   0.063275      0.983770      Europe & Central Asia (excluding high income)     xgboost    309.0
3051         population ages 15-64   0.343107    0.194269   0.440760      0.212516      Europe & Central Asia (excluding high income)          rf   4054.0
1629         population ages 15-64   1.380437    2.268764   1.506242     -8.196597      Europe & Central Asia (excluding high income)    base-gru   9674.0
207          population ages 15-64   1.737913    3.528879   1.878531    -13.304563      Europe & Central Asia (excluding high income)  simple-rnn  10603.0
918          population ages 15-64   2.001177    4.681419   2.163659    -17.976465      Europe & Central Asia (excluding high income)   base-lstm  11205.0
2342  population ages 65 and above   0.066818    0.006654   0.081574      0.944840      Europe & Central Asia (excluding high income)     xgboost    566.0
3053  population ages 65 and above   0.222012    0.067512   0.259831      0.440361      Europe & Central Asia (excluding high income)          rf   2738.0
920   population ages 65 and above   1.541046    2.400354   1.549308    -18.897565      Europe & Central Asia (excluding high income)   base-lstm  10172.0
209   population ages 65 and above   1.601454    2.594816   1.610843    -20.509546      Europe & Central Asia (excluding high income)  simple-rnn  10349.0
1631  population ages 65 and above   1.753021    3.104201   1.761874    -24.732050      Europe & Central Asia (excluding high income)    base-gru  10716.0
2344          population ages 0-14   0.123864    0.022003   0.148335     -0.093320                                     European Union     xgboost   2125.0
3055          population ages 0-14   0.220312    0.072478   0.269218     -2.601390                                     European Union          rf   3788.0
1633          population ages 0-14   3.017914    9.573246   3.094066   -474.687886                                     European Union    base-gru  12906.0
922           population ages 0-14   3.486901   12.832570   3.582258   -636.641375                                     European Union   base-lstm  13384.0
211           population ages 0-14   3.882199   15.885558   3.985669   -788.342198                                     European Union  simple-rnn  13650.0
2343         population ages 15-64   0.101272    0.017034   0.130513      0.983615                                     European Union     xgboost    808.0
3054         population ages 15-64   0.591473    0.539340   0.734398      0.481194                                     European Union          rf   5396.0
1632         population ages 15-64   1.527513    3.168046   1.779901     -2.047428                                     European Union    base-gru   9550.0
210          population ages 15-64   2.360662    6.955350   2.637300     -5.690535                                     European Union  simple-rnn  11263.0
921          population ages 15-64   2.973757   10.500940   3.240515     -9.101132                                     European Union   base-lstm  12036.0
3056  population ages 65 and above   0.128711    0.036201   0.190266      0.973097                                     European Union          rf   1299.0
923   population ages 65 and above   0.347916    0.158628   0.398282      0.882113                                     European Union   base-lstm   3218.0
2345  population ages 65 and above   0.313247    0.170816   0.413298      0.873056                                     European Union     xgboost   3227.0
212   population ages 65 and above   0.345965    0.162688   0.403346      0.879096                                     European Union  simple-rnn   3261.0
1634  population ages 65 and above   0.476928    0.302572   0.550065      0.775139                                     European Union    base-gru   4277.0
2347          population ages 0-14   0.416736    0.294111   0.542320      0.383748                                               Fiji     xgboost   4496.0
3058          population ages 0-14   0.888571    1.122033   1.059261     -1.350998                                               Fiji          rf   7497.0
925           population ages 0-14   2.979521   10.497703   3.240016    -20.995850                                               Fiji   base-lstm  12350.0
1636          population ages 0-14   3.170404   11.742540   3.426739    -23.604159                                               Fiji    base-gru  12578.0
214           population ages 0-14   3.299682   12.703808   3.564240    -25.618304                                               Fiji  simple-rnn  12729.0
3057         population ages 15-64   0.686864    0.810111   0.900062    -18.467135                                               Fiji          rf   7906.0
2346         population ages 15-64   0.728217    0.820663   0.905904    -18.720690                                               Fiji     xgboost   7986.0
1635         population ages 15-64   2.450672    6.767990   2.601536   -161.636129                                               Fiji    base-gru  12252.0
924          population ages 15-64   2.507713    7.228484   2.688584   -172.701884                                               Fiji   base-lstm  12338.0
213          population ages 15-64   2.668400    7.954909   2.820445   -190.158011                                               Fiji  simple-rnn  12505.0
3059  population ages 65 and above   0.137512    0.035959   0.189630      0.850622                                               Fiji          rf   1657.0
2348  population ages 65 and above   0.305113    0.148753   0.385686      0.382070                                               Fiji     xgboost   3645.0
926   population ages 65 and above   0.941570    1.002996   1.001497     -3.166507                                               Fiji   base-lstm   7791.0
215   population ages 65 and above   1.080657    1.341775   1.158350     -4.573815                                               Fiji  simple-rnn   8479.0
1637  population ages 65 and above   1.120322    1.409152   1.187077     -4.853703                                               Fiji    base-gru   8625.0
3061          population ages 0-14   0.240403    0.089995   0.299991      0.273312                                            Finland          rf   3100.0
2350          population ages 0-14   0.522703    0.448027   0.669348     -2.617726                                            Finland     xgboost   6088.0
1639          population ages 0-14   1.566693    2.695393   1.641765    -20.764730                                            Finland    base-gru  10369.0
928           population ages 0-14   2.143632    5.028957   2.242534    -39.607777                                            Finland   base-lstm  11590.0
217           population ages 0-14   2.231118    5.440130   2.332409    -42.927910                                            Finland  simple-rnn  11696.0
3060         population ages 15-64   2.067804    5.731023   2.393956     -1.187742                                            Finland          rf  10294.0
1638         population ages 15-64   2.133281    6.551834   2.559655     -1.501076                                            Finland    base-gru  10519.0
2349         population ages 15-64   2.464522    8.706937   2.950752     -2.323758                                            Finland     xgboost  11100.0
216          population ages 15-64   2.849501   11.183105   3.344115     -3.269002                                            Finland  simple-rnn  11633.0
927          population ages 15-64   3.314337   14.423984   3.797892     -4.506164                                            Finland   base-lstm  12188.0
2351  population ages 65 and above   1.008458    1.330671   1.153547      0.643848                                            Finland     xgboost   6943.0
1640  population ages 65 and above   1.529465    3.302162   1.817185      0.116181                                            Finland    base-gru   8876.0
929   population ages 65 and above   1.664268    3.911343   1.977712     -0.046865                                            Finland   base-lstm   9262.0
3062  population ages 65 and above   1.714983    3.868940   1.966962     -0.035516                                            Finland          rf   9263.0
218   population ages 65 and above   1.687612    4.014670   2.003664     -0.074520                                            Finland  simple-rnn   9322.0
3064          population ages 0-14   0.627943    0.539796   0.734708     -4.914885           Fragile and conflict affected situations          rf   6812.0
1642          population ages 0-14   0.811165    0.726643   0.852434     -6.962296           Fragile and conflict affected situations    base-gru   7619.0
2353          population ages 0-14   0.812856    0.849353   0.921604     -8.306907           Fragile and conflict affected situations     xgboost   7856.0
931           population ages 0-14   1.001019    1.118051   1.057379    -11.251198           Fragile and conflict affected situations   base-lstm   8519.0
220           population ages 0-14   1.241664    1.710326   1.307794    -17.741141           Fragile and conflict affected situations  simple-rnn   9499.0
1641         population ages 15-64   0.147001    0.026226   0.161945      0.589918           Fragile and conflict affected situations    base-gru   1858.0
930          population ages 15-64   0.200567    0.052317   0.228730      0.181950           Fragile and conflict affected situations   base-lstm   2661.0
219          population ages 15-64   0.365915    0.180190   0.424487     -1.817499           Fragile and conflict affected situations  simple-rnn   4766.0
3063         population ages 15-64   0.588571    0.522688   0.722972     -7.172911           Fragile and conflict affected situations          rf   6863.0
2352         population ages 15-64   0.637665    0.557921   0.746941     -7.723826           Fragile and conflict affected situations     xgboost   7069.0
3065  population ages 65 and above   0.016119    0.000408   0.020189      0.859378           Fragile and conflict affected situations          rf    450.0
2354  population ages 65 and above   0.076650    0.006859   0.082820     -1.366415           Fragile and conflict affected situations     xgboost   2060.0
221   population ages 65 and above   0.621115    0.418473   0.646895   -143.372524           Fragile and conflict affected situations  simple-rnn   7515.0
932   population ages 65 and above   0.637836    0.436256   0.660497   -149.507688           Fragile and conflict affected situations   base-lstm   7600.0
1643  population ages 65 and above   0.761723    0.626409   0.791460   -215.110136           Fragile and conflict affected situations    base-gru   8249.0
3067          population ages 0-14   0.403489    0.266922   0.516645     -1.746539                                             France          rf   5190.0
2356          population ages 0-14   0.549383    0.441170   0.664206     -3.539498                                             France     xgboost   6282.0
1645          population ages 0-14   3.999669   16.570173   4.070648   -169.501689                                             France    base-gru  13497.0
934           population ages 0-14   4.449011   20.573446   4.535796   -210.694075                                             France   base-lstm  13654.0
223           population ages 0-14   4.717662   23.164543   4.812956   -237.355616                                             France  simple-rnn  13738.0
2355         population ages 15-64   1.258385    2.323142   1.524186     -0.557645                                             France     xgboost   8571.0
3066         population ages 15-64   1.816690    4.710396   2.170345     -2.158276                                             France          rf  10226.0
1644         population ages 15-64   2.403483    7.692912   2.773610     -4.158025                                             France    base-gru  11261.0
222          population ages 15-64   3.216060   12.984431   3.603392     -7.705938                                             France  simple-rnn  12290.0
933          population ages 15-64   3.752467   17.150519   4.141318    -10.499260                                             France   base-lstm  12733.0
1646  population ages 65 and above   0.583261    0.462339   0.679955      0.794563                                             France    base-gru   4871.0
224   population ages 65 and above   0.606428    0.461326   0.679210      0.795014                                             France  simple-rnn   4905.0
935   population ages 65 and above   0.670051    0.577189   0.759730      0.743531                                             France   base-lstm   5364.0
3068  population ages 65 and above   1.161327    2.125624   1.457952      0.055496                                             France          rf   8106.0
2357  population ages 65 and above   2.087705    6.573477   2.563879     -1.920871                                             France     xgboost  10612.0
2359          population ages 0-14   0.117335    0.020408   0.142856      0.986929                                   French Polynesia     xgboost    911.0
3070          population ages 0-14   0.112900    0.022833   0.151106      0.985376                                   French Polynesia          rf    953.0
937           population ages 0-14   1.646739    3.222252   1.795063     -1.063793                                   French Polynesia   base-lstm   9398.0
1648          population ages 0-14   1.681475    3.237853   1.799404     -1.073786                                   French Polynesia    base-gru   9442.0
226           population ages 0-14   1.782566    3.681384   1.918693     -1.357859                                   French Polynesia  simple-rnn   9734.0
2358         population ages 15-64   0.140237    0.029081   0.170530      0.445132                                   French Polynesia     xgboost   2023.0
3069         population ages 15-64   0.539724    0.420481   0.648445     -7.022964                                   French Polynesia          rf   6495.0
1647         population ages 15-64   1.974719    4.186671   2.046136    -78.883456                                   French Polynesia    base-gru  11478.0
225          population ages 15-64   2.135069    4.902124   2.214074    -92.534587                                   French Polynesia  simple-rnn  11748.0
936          population ages 15-64   2.356304    6.117851   2.473429   -115.731182                                   French Polynesia   base-lstm  12063.0
938   population ages 65 and above   0.088673    0.012420   0.111444      0.988426                                   French Polynesia   base-lstm    626.0
227   population ages 65 and above   0.269428    0.085281   0.292028      0.920529                                   French Polynesia  simple-rnn   2396.0
1649  population ages 65 and above   0.275849    0.085306   0.292073      0.920505                                   French Polynesia    base-gru   2412.0
2360  population ages 65 and above   0.321157    0.206527   0.454452      0.807542                                   French Polynesia     xgboost   3550.0
3071  population ages 65 and above   0.527271    0.454428   0.674113      0.576528                                   French Polynesia          rf   4977.0
2362          population ages 0-14   0.190107    0.052701   0.229566      0.590713                                              Gabon     xgboost   2365.0
3073          population ages 0-14   0.998090    1.738693   1.318595    -12.503187                                              Gabon          rf   9134.0
1651          population ages 0-14   1.504852    3.074428   1.753405    -22.876894                                              Gabon    base-gru  10491.0
940           population ages 0-14   1.702002    3.917374   1.979236    -29.423453                                              Gabon   base-lstm  10999.0
229           population ages 0-14   2.116449    5.759783   2.399955    -43.732126                                              Gabon  simple-rnn  11712.0
2361         population ages 15-64   0.215443    0.062040   0.249079      0.705076                                              Gabon     xgboost   2454.0
1650         population ages 15-64   0.417192    0.206321   0.454226      0.019202                                              Gabon    base-gru   4370.0
939          population ages 15-64   0.574604    0.508859   0.713343     -1.418982                                              Gabon   base-lstm   6093.0
228          population ages 15-64   0.712604    0.866945   0.931099     -3.121229                                              Gabon  simple-rnn   7292.0
3072         population ages 15-64   0.741429    0.905267   0.951455     -3.303404                                              Gabon          rf   7422.0
3074  population ages 65 and above   0.016132    0.000330   0.018164      0.967518                                              Gabon          rf    159.0
2363  population ages 65 and above   0.087948    0.019574   0.139908     -0.926998                                              Gabon     xgboost   2262.0
941   population ages 65 and above   1.245419    1.699877   1.303793   -166.346420                                              Gabon   base-lstm  10049.0
230   population ages 65 and above   1.310556    1.897543   1.377513   -185.805895                                              Gabon  simple-rnn  10258.0
1652  population ages 65 and above   1.448839    2.307221   1.518954   -226.137097                                              Gabon    base-gru  10612.0
2365          population ages 0-14   0.189973    0.044442   0.210812      0.915720                                        Gambia, The     xgboost   1759.0
3076          population ages 0-14   0.188004    0.083950   0.289742      0.840796                                        Gambia, The          rf   2321.0
1654          population ages 0-14   1.103478    1.253213   1.119470     -1.376601                                        Gambia, The    base-gru   7899.0
943           population ages 0-14   1.345132    1.859850   1.363763     -2.527030                                        Gambia, The   base-lstm   8902.0
232           population ages 0-14   1.361750    1.938731   1.392383     -2.676620                                        Gambia, The  simple-rnn   8984.0
231          population ages 15-64   0.159168    0.031206   0.176652      0.956442                                        Gambia, The  simple-rnn   1376.0
2364         population ages 15-64   0.196384    0.078325   0.279866      0.890672                                        Gambia, The     xgboost   2208.0
942          population ages 15-64   0.237901    0.068620   0.261954      0.904219                                        Gambia, The   base-lstm   2216.0
1653         population ages 15-64   0.225523    0.099045   0.314713      0.861751                                        Gambia, The    base-gru   2527.0
3075         population ages 15-64   0.268713    0.135170   0.367655      0.811326                                        Gambia, The          rf   3013.0
3077  population ages 65 and above   0.109100    0.016275   0.127573      0.098588                                        Gambia, The          rf   1851.0
2366  population ages 65 and above   0.264817    0.101626   0.318788     -4.628702                                        Gambia, The     xgboost   4447.0
233   population ages 65 and above   1.148390    1.446177   1.202571    -79.098525                                        Gambia, The  simple-rnn   9609.0
944   population ages 65 and above   1.192984    1.554693   1.246873    -85.108836                                        Gambia, The   base-lstm   9779.0
1655  population ages 65 and above   1.279842    1.776542   1.332870    -97.396281                                        Gambia, The    base-gru  10050.0
2368          population ages 0-14   0.422645    0.293383   0.541648     -0.418976                                            Georgia     xgboost   4906.0
3079          population ages 0-14   0.563728    0.425670   0.652434     -1.058796                                            Georgia          rf   5739.0
1657          population ages 0-14   3.930195   15.907087   3.988369    -75.936294                                            Georgia    base-gru  13311.0
946           population ages 0-14   4.203273   18.261162   4.273308    -87.322026                                            Georgia   base-lstm  13445.0
235           population ages 0-14   4.296349   19.023549   4.361599    -91.009391                                            Georgia  simple-rnn  13491.0
2367         population ages 15-64   0.093027    0.012471   0.111674      0.968351                                            Georgia     xgboost    737.0
3078         population ages 15-64   0.221911    0.057527   0.239848      0.854007                                            Georgia          rf   2175.0
1656         population ages 15-64   1.503257    2.506178   1.583091     -5.360181                                            Georgia    base-gru   9715.0
234          population ages 15-64   1.917266    4.071352   2.017759     -9.332278                                            Georgia  simple-rnn  10770.0
945          population ages 15-64   2.221182    5.425882   2.329352    -12.769807                                            Georgia   base-lstm  11291.0
2369  population ages 65 and above   0.046035    0.003177   0.056368      0.894623                                            Georgia     xgboost    559.0
3080  population ages 65 and above   0.087221    0.012862   0.113411      0.573432                                            Georgia          rf   1394.0
947   population ages 65 and above   1.414509    2.038842   1.427880    -66.617997                                            Georgia   base-lstm  10225.0
236   population ages 65 and above   1.418801    2.052313   1.432589    -67.064751                                            Georgia  simple-rnn  10236.0
1658  population ages 65 and above   1.648498    2.762562   1.662096    -90.620105                                            Georgia    base-gru  10806.0
3082          population ages 0-14   0.177912    0.052015   0.228068     -0.174945                                            Germany          rf   2777.0
2371          population ages 0-14   0.286386    0.122458   0.349939     -1.766135                                            Germany     xgboost   4212.0
1660          population ages 0-14   2.478588    6.779283   2.603706   -152.133922                                            Germany    base-gru  12265.0
949           population ages 0-14   2.917783    9.361415   3.059643   -210.460425                                            Germany   base-lstm  12722.0
238           population ages 0-14   3.490108   13.253463   3.640531   -298.376006                                            Germany  simple-rnn  13295.0
3081         population ages 15-64   0.382292    0.193793   0.440219      0.637359                                            Germany          rf   3848.0
2370         population ages 15-64   0.543497    0.503129   0.709316      0.058504                                            Germany     xgboost   5470.0
1659         population ages 15-64   1.334497    2.309576   1.519729     -3.321865                                            Germany    base-gru   9282.0
237          population ages 15-64   2.276931    6.104474   2.470723    -10.423181                                            Germany  simple-rnn  11342.0
948          population ages 15-64   2.953510    9.786281   3.128303    -17.312875                                            Germany   base-lstm  12199.0
1661  population ages 65 and above   0.210388    0.074367   0.272703      0.794326                                            Germany    base-gru   2414.0
239   population ages 65 and above   0.244242    0.121053   0.347926      0.665211                                            Germany  simple-rnn   3026.0
950   population ages 65 and above   0.378827    0.211626   0.460028      0.414716                                            Germany   base-lstm   4101.0
2372  population ages 65 and above   0.784457    0.942906   0.971034     -1.607750                                            Germany     xgboost   7230.0
3083  population ages 65 and above   0.968947    1.285183   1.133659     -2.554368                                            Germany          rf   8037.0
3085          population ages 0-14   0.663042    0.585339   0.765075     -0.432391                                              Ghana          rf   6087.0
2374          population ages 0-14   0.613867    0.726318   0.852243     -0.777380                                              Ghana     xgboost   6420.0
1663          population ages 0-14   2.302803    6.072779   2.464301    -13.860766                                              Ghana    base-gru  11449.0
952           population ages 0-14   2.397187    6.617982   2.572544    -15.194941                                              Ghana   base-lstm  11604.0
241           population ages 0-14   2.649783    8.037772   2.835096    -18.669324                                              Ghana  simple-rnn  11961.0
2373         population ages 15-64   0.366374    0.282525   0.531531      0.009442                                              Ghana     xgboost   4545.0
3084         population ages 15-64   0.667618    0.578308   0.760466     -1.027601                                              Ghana          rf   6278.0
1662         population ages 15-64   0.964924    1.143660   1.069420     -3.009777                                              Ghana    base-gru   7943.0
951          population ages 15-64   1.238267    1.907008   1.380945     -5.686146                                              Ghana   base-lstm   9205.0
240          population ages 15-64   1.398339    2.386927   1.544968     -7.368784                                              Ghana  simple-rnn   9722.0
2375  population ages 65 and above   0.100448    0.013201   0.114896     -0.024776                                              Ghana     xgboost   1826.0
3086  population ages 65 and above   0.086009    0.014944   0.122244     -0.160045                                              Ghana          rf   1870.0
953   population ages 65 and above   1.054331    1.215166   1.102346    -93.330983                                              Ghana   base-lstm   9296.0
242   population ages 65 and above   1.096359    1.326269   1.151638   -101.955717                                              Ghana  simple-rnn   9479.0
1664  population ages 65 and above   1.231121    1.667387   1.291273   -128.436054                                              Ghana    base-gru   9967.0
3088          population ages 0-14   0.243400    0.086670   0.294398     -1.654704                                             Greece          rf   3797.0
2377          population ages 0-14   0.448146    0.271560   0.521114     -7.317895                                             Greece     xgboost   5953.0
1666          population ages 0-14   3.298908   11.194744   3.345855   -341.895706                                             Greece    base-gru  13100.0
955           population ages 0-14   3.609097   13.461797   3.669032   -411.335680                                             Greece   base-lstm  13394.0
244           population ages 0-14   4.247247   18.640677   4.317485   -569.965081                                             Greece  simple-rnn  13750.0
2376         population ages 15-64   0.093933    0.015344   0.123870      0.982798                                             Greece     xgboost    742.0
3087         population ages 15-64   0.324000    0.135242   0.367753      0.848380                                             Greece          rf   3078.0
1665         population ages 15-64   2.140149    5.307033   2.303700     -4.949690                                             Greece    base-gru  10877.0
243          population ages 15-64   3.090353   10.772665   3.282174    -11.077183                                             Greece  simple-rnn  12183.0
954          population ages 15-64   3.641525   14.689241   3.832655    -15.468038                                             Greece   base-lstm  12759.0
1667  population ages 65 and above   0.308255    0.135732   0.368419      0.885218                                             Greece    base-gru   2989.0
245   population ages 65 and above   0.322073    0.160576   0.400719      0.864210                                             Greece  simple-rnn   3213.0
3089  population ages 65 and above   0.307396    0.172903   0.415816      0.853785                                             Greece          rf   3244.0
956   population ages 65 and above   0.459729    0.318425   0.564291      0.730725                                             Greece   base-lstm   4355.0
2378  population ages 65 and above   0.623078    0.823332   0.907377      0.303752                                             Greece     xgboost   6088.0
2380          population ages 0-14   0.224413    0.058579   0.242031    -10.100615                                            Grenada     xgboost   4281.0
3091          population ages 0-14   0.607367    0.537016   0.732814   -100.763213                                            Grenada          rf   7724.0
1669          population ages 0-14   3.037064   10.464691   3.234917  -1982.033595                                            Grenada    base-gru  13091.0
247           population ages 0-14   3.241830   11.933640   3.454510  -2260.395872                                            Grenada  simple-rnn  13315.0
958           population ages 0-14   3.320046   12.545357   3.541943  -2376.314778                                            Grenada   base-lstm  13390.0
2379         population ages 15-64   0.275023    0.138369   0.371979     -0.880821                                            Grenada     xgboost   4054.0
3090         population ages 15-64   0.300700    0.137316   0.370561     -0.866507                                            Grenada          rf   4113.0
1668         population ages 15-64   0.665757    0.668236   0.817457     -8.083221                                            Grenada    base-gru   7353.0
246          population ages 15-64   0.848369    1.075011   1.036827    -13.612447                                            Grenada  simple-rnn   8352.0
957          population ages 15-64   1.180572    2.069539   1.438589    -27.130912                                            Grenada   base-lstm   9829.0
2381  population ages 65 and above   0.140023    0.027075   0.164546      0.549288                                            Grenada     xgboost   1903.0
3092  population ages 65 and above   0.266692    0.159210   0.399011     -1.650312                                            Grenada          rf   4376.0
959   population ages 65 and above   1.956178    4.063546   2.015824    -66.644363                                            Grenada   base-lstm  11399.0
248   population ages 65 and above   2.043382    4.432842   2.105432    -72.791904                                            Grenada  simple-rnn  11572.0
1670  population ages 65 and above   2.165397    4.959451   2.226982    -81.558167                                            Grenada    base-gru  11745.0
3094          population ages 0-14   0.047261    0.007440   0.086256      0.809475                                               Guam          rf    862.0
1672          population ages 0-14   0.819281    0.695264   0.833825    -16.804397                                               Guam    base-gru   7896.0
2383          population ages 0-14   0.825459    0.962975   0.981313    -23.659975                                               Guam     xgboost   8404.0
961           population ages 0-14   0.957760    0.971675   0.985736    -23.882765                                               Guam   base-lstm   8574.0
250           population ages 0-14   1.047623    1.144205   1.069675    -28.300932                                               Guam  simple-rnn   8928.0
2382         population ages 15-64   0.069352    0.010915   0.104473      0.850790                                               Guam     xgboost    912.0
3093         population ages 15-64   0.248601    0.067580   0.259961      0.076136                                               Guam          rf   3024.0
1671         population ages 15-64   1.593718    2.554708   1.598345    -33.924621                                               Guam    base-gru  10494.0
249          population ages 15-64   2.141870    4.615158   2.148292    -62.092401                                               Guam  simple-rnn  11605.0
960          population ages 15-64   2.148871    4.649887   2.156360    -62.567172                                               Guam   base-lstm  11621.0
2384  population ages 65 and above   0.073662    0.007580   0.087061      0.963599                                               Guam     xgboost    550.0
3095  population ages 65 and above   0.368570    0.173878   0.416987      0.164938                                               Guam          rf   4035.0
1673  population ages 65 and above   1.236361    1.533164   1.238210     -6.363132                                               Guam    base-gru   9005.0
251   population ages 65 and above   1.337064    1.794969   1.339765     -7.620473                                               Guam  simple-rnn   9358.0
962   population ages 65 and above   1.372218    1.890208   1.374848     -8.077862                                               Guam   base-lstm   9461.0
3097          population ages 0-14   0.076746    0.010084   0.100419      0.997684                                          Guatemala          rf    480.0
2386          population ages 0-14   0.197880    0.057503   0.239797      0.986791                                          Guatemala     xgboost   1721.0
1675          population ages 0-14   1.014356    1.312214   1.145519      0.698564                                          Guatemala    base-gru   6881.0
964           population ages 0-14   1.155137    1.678863   1.295709      0.614339                                          Guatemala   base-lstm   7450.0
253           population ages 0-14   1.382316    2.276123   1.508683      0.477139                                          Guatemala  simple-rnn   8124.0
1674         population ages 15-64   0.125190    0.021243   0.145749      0.993718                                          Guatemala    base-gru    916.0
3096         population ages 15-64   0.208890    0.094839   0.307960      0.971952                                          Guatemala          rf   2148.0
963          population ages 15-64   0.291408    0.162079   0.402591      0.952067                                          Guatemala   base-lstm   2916.0
252          population ages 15-64   0.358167    0.208604   0.456732      0.938308                                          Guatemala  simple-rnn   3330.0
2385         population ages 15-64   0.771149    0.821066   0.906127      0.757180                                          Guatemala     xgboost   5923.0
2387  population ages 65 and above   0.018086    0.000634   0.025187      0.990019                                          Guatemala     xgboost     78.0
3098  population ages 65 and above   0.035117    0.002548   0.050482      0.959904                                          Guatemala          rf    312.0
965   population ages 65 and above   0.712764    0.593287   0.770251     -8.334310                                          Guatemala   base-lstm   7279.0
254   population ages 65 and above   0.735374    0.648320   0.805183     -9.200158                                          Guatemala  simple-rnn   7461.0
1676  population ages 65 and above   0.882317    0.910739   0.954326    -13.328852                                          Guatemala    base-gru   8185.0
967           population ages 0-14   0.154023    0.031975   0.178816      0.944934                                             Guinea   base-lstm   1422.0
1678          population ages 0-14   0.223140    0.058632   0.242140      0.899028                                             Guinea    base-gru   2105.0
3100          population ages 0-14   0.226471    0.102400   0.320000      0.823652                                             Guinea          rf   2633.0
256           population ages 0-14   0.227641    0.102877   0.320745      0.822830                                             Guinea  simple-rnn   2646.0
2389          population ages 0-14   0.451746    0.308622   0.555538      0.468508                                             Guinea     xgboost   4545.0
3099         population ages 15-64   0.081471    0.009989   0.099947      0.986334                                             Guinea          rf    553.0
2388         population ages 15-64   0.208802    0.049960   0.223518      0.931651                                             Guinea     xgboost   1851.0
255          population ages 15-64   0.554442    0.311551   0.558167      0.573778                                             Guinea  simple-rnn   4656.0
966          population ages 15-64   0.675763    0.462422   0.680016      0.367375                                             Guinea   base-lstm   5441.0
1677         population ages 15-64   0.957372    0.945143   0.972185     -0.293020                                             Guinea    base-gru   7009.0
3101  population ages 65 and above   0.046414    0.003200   0.056569      0.659614                                             Guinea          rf    920.0
2390  population ages 65 and above   0.165571    0.037600   0.193908     -2.999483                                             Guinea     xgboost   3329.0
257   population ages 65 and above   0.488826    0.277062   0.526367    -28.470771                                             Guinea  simple-rnn   6529.0
968   population ages 65 and above   0.537117    0.327665   0.572420    -33.853355                                             Guinea   base-lstm   6816.0
1679  population ages 65 and above   0.602621    0.411127   0.641192    -42.731131                                             Guinea    base-gru   7210.0
2392          population ages 0-14   0.379088    0.215140   0.463832      0.750986                                      Guinea-Bissau     xgboost   3818.0
1681          population ages 0-14   1.001106    1.040160   1.019883     -0.203932                                      Guinea-Bissau    base-gru   7141.0
970           population ages 0-14   1.090729    1.241596   1.114269     -0.437083                                      Guinea-Bissau   base-lstm   7555.0
3103          population ages 0-14   0.877288    1.502065   1.225588     -0.738564                                      Guinea-Bissau          rf   7688.0
259           population ages 0-14   1.336456    1.874351   1.369069     -1.169465                                      Guinea-Bissau  simple-rnn   8592.0
1680         population ages 15-64   0.111699    0.016306   0.127697      0.977939                                      Guinea-Bissau    base-gru    846.0
969          population ages 15-64   0.217708    0.055650   0.235902      0.924712                                      Guinea-Bissau   base-lstm   1979.0
3102         population ages 15-64   0.211155    0.094176   0.306882      0.872589                                      Guinea-Bissau          rf   2436.0
2391         population ages 15-64   0.295949    0.137247   0.370468      0.814318                                      Guinea-Bissau     xgboost   3111.0
258          population ages 15-64   0.417546    0.199056   0.446157      0.730697                                      Guinea-Bissau  simple-rnn   3846.0
2393  population ages 65 and above   0.047524    0.003850   0.062050      0.270822                                      Guinea-Bissau     xgboost   1240.0
3104  population ages 65 and above   0.065114    0.006611   0.081306     -0.251961                                      Guinea-Bissau          rf   1627.0
260   population ages 65 and above   0.803659    0.668557   0.817653   -125.614087                                      Guinea-Bissau  simple-rnn   8343.0
971   population ages 65 and above   0.839930    0.727220   0.852772   -136.723947                                      Guinea-Bissau   base-lstm   8520.0
1682  population ages 65 and above   0.907077    0.850825   0.922402   -160.132793                                      Guinea-Bissau    base-gru   8800.0
2395          population ages 0-14   0.284130    0.116708   0.341625      0.932802                                             Guyana     xgboost   2635.0
3106          population ages 0-14   0.304121    0.134372   0.366568      0.922632                                             Guyana          rf   2854.0
973           population ages 0-14   0.937717    1.631412   1.277268      0.060672                                             Guyana   base-lstm   7521.0
1684          population ages 0-14   0.991692    1.733749   1.316719      0.001749                                             Guyana    base-gru   7690.0
262           population ages 0-14   1.146209    2.138927   1.462507     -0.231543                                             Guyana  simple-rnn   8224.0
2394         population ages 15-64   0.417860    0.319674   0.565397      0.492137                                             Guyana     xgboost   4490.0
1683         population ages 15-64   0.500000    0.377798   0.614652      0.399795                                             Guyana    base-gru   4883.0
3105         population ages 15-64   0.525542    0.505213   0.710783      0.197372                                             Guyana          rf   5358.0
972          population ages 15-64   0.573472    0.505061   0.710677      0.197612                                             Guyana   base-lstm   5454.0
261          population ages 15-64   0.585896    0.589601   0.767855      0.063305                                             Guyana  simple-rnn   5740.0
3107  population ages 65 and above   0.357336    0.179216   0.423340      0.414831                                             Guyana          rf   3891.0
2396  population ages 65 and above   0.414913    0.264009   0.513818      0.137969                                             Guyana     xgboost   4522.0
974   population ages 65 and above   0.783063    0.677818   0.823297     -1.213182                                             Guyana   base-lstm   6734.0
263   population ages 65 and above   0.814367    0.748020   0.864882     -1.442402                                             Guyana  simple-rnn   6959.0
1685  population ages 65 and above   0.914227    0.920224   0.959283     -2.004676                                             Guyana    base-gru   7437.0
2398          population ages 0-14   0.126500    0.020882   0.144506      0.986988                                              Haiti     xgboost    950.0
3109          population ages 0-14   0.200430    0.099654   0.315680      0.937904                                              Haiti          rf   2263.0
1687          population ages 0-14   2.397254    7.095200   2.663682     -3.421167                                              Haiti    base-gru  11063.0
976           population ages 0-14   2.410551    7.152236   2.674366     -3.456707                                              Haiti   base-lstm  11091.0
265           population ages 0-14   2.419133    7.160769   2.675961     -3.462024                                              Haiti  simple-rnn  11100.0
3108         population ages 15-64   0.136857    0.033656   0.183455      0.972577                                              Haiti          rf   1309.0
2397         population ages 15-64   0.542485    0.369793   0.608106      0.698694                                              Haiti     xgboost   4693.0
1686         population ages 15-64   0.641618    0.637681   0.798549      0.480420                                              Haiti    base-gru   5695.0
264          population ages 15-64   0.744186    0.872236   0.933936      0.289305                                              Haiti  simple-rnn   6343.0
975          population ages 15-64   1.011048    1.513423   1.230212     -0.233132                                              Haiti   base-lstm   7668.0
3110  population ages 65 and above   0.057868    0.004304   0.065609      0.847355                                              Haiti          rf    701.0
2399  population ages 65 and above   0.211513    0.052378   0.228862     -0.857410                                              Haiti     xgboost   3148.0
977   population ages 65 and above   1.169443    1.579839   1.256916    -55.023895                                              Haiti   base-lstm   9689.0
266   population ages 65 and above   1.332455    2.063282   1.436413    -72.167642                                              Haiti  simple-rnn  10195.0
1688  population ages 65 and above   1.514087    2.651418   1.628318    -93.024011                                              Haiti    base-gru  10688.0
2401          population ages 0-14   0.099029    0.012589   0.112200      0.977751             Heavily indebted poor countries (HIPC)     xgboost    722.0
3112          population ages 0-14   0.267114    0.112744   0.335774      0.800742             Heavily indebted poor countries (HIPC)          rf   2879.0
1690          population ages 0-14   0.409083    0.189903   0.435779      0.664375             Heavily indebted poor countries (HIPC)    base-gru   3842.0
979           population ages 0-14   0.541366    0.335479   0.579206      0.407092             Heavily indebted poor countries (HIPC)   base-lstm   4849.0
268           population ages 0-14   0.793806    0.710445   0.842879     -0.255602             Heavily indebted poor countries (HIPC)  simple-rnn   6477.0
978          population ages 15-64   0.069273    0.005860   0.076552      0.988511             Heavily indebted poor countries (HIPC)   base-lstm    382.0
1689         population ages 15-64   0.096689    0.012831   0.113273      0.974845             Heavily indebted poor countries (HIPC)    base-gru    734.0
3111         population ages 15-64   0.125326    0.021741   0.147449      0.957375             Heavily indebted poor countries (HIPC)          rf   1087.0
2400         population ages 15-64   0.235830    0.079738   0.282380      0.843668             Heavily indebted poor countries (HIPC)     xgboost   2413.0
267          population ages 15-64   0.262377    0.080572   0.283851      0.842034             Heavily indebted poor countries (HIPC)  simple-rnn   2499.0
3113  population ages 65 and above   0.036251    0.001901   0.043599     -0.254117             Heavily indebted poor countries (HIPC)          rf   1417.0
2402  population ages 65 and above   0.030403    0.002032   0.045075     -0.340478             Heavily indebted poor countries (HIPC)     xgboost   1450.0
269   population ages 65 and above   0.429956    0.211517   0.459910   -138.550026             Heavily indebted poor countries (HIPC)  simple-rnn   6512.0
980   population ages 65 and above   0.473585    0.250785   0.500784   -164.457150             Heavily indebted poor countries (HIPC)   base-lstm   6762.0
1691  population ages 65 and above   0.519843    0.304342   0.551672   -199.791667             Heavily indebted poor countries (HIPC)    base-gru   7043.0
3115          population ages 0-14   0.215225    0.083535   0.289024     -0.787471                                        High income          rf   3425.0
2404          population ages 0-14   0.617504    0.433268   0.658231     -8.271002                                        High income     xgboost   6747.0
1693          population ages 0-14   2.568155    7.100975   2.664766   -150.945738                                        High income    base-gru  12315.0
982           population ages 0-14   3.050379   10.062623   3.172164   -214.318683                                        High income   base-lstm  12837.0
271           population ages 0-14   3.229258   11.249164   3.353977   -239.708136                                        High income  simple-rnn  13034.0
1692         population ages 15-64   0.681015    0.668409   0.817563      0.254281                                        High income    base-gru   5960.0
3114         population ages 15-64   1.169279    2.233938   1.494636     -1.492321                                        High income          rf   8714.0
2403         population ages 15-64   1.200146    2.257713   1.502569     -1.518846                                        High income     xgboost   8771.0
270          population ages 15-64   1.249363    2.386947   1.544975     -1.663027                                        High income  simple-rnn   8939.0
981          population ages 15-64   1.733636    4.102611   2.025490     -3.577129                                        High income   base-lstm  10270.0
3116  population ages 65 and above   0.612940    0.583105   0.763613      0.565673                                        High income          rf   5450.0
983   population ages 65 and above   0.967179    0.952354   0.975886      0.290636                                        High income   base-lstm   6734.0
272   population ages 65 and above   1.006738    1.024894   1.012370      0.236605                                        High income  simple-rnn   6900.0
1694  population ages 65 and above   1.268929    1.615792   1.271138     -0.203527                                        High income    base-gru   8008.0
2405  population ages 65 and above   1.601037    3.965052   1.991244     -1.953380                                        High income     xgboost   9857.0
2407          population ages 0-14   0.338237    0.202244   0.449715      0.967255                                           Honduras     xgboost   3161.0
3118          population ages 0-14   1.185011    2.077214   1.441254      0.663686                                           Honduras          rf   7672.0
985           population ages 0-14   1.658080    3.125322   1.767858      0.493990                                           Honduras   base-lstm   8677.0
1696          population ages 0-14   1.658278    3.140563   1.772163      0.491523                                           Honduras    base-gru   8689.0
274           population ages 0-14   1.916142    4.126576   2.031398      0.331881                                           Honduras  simple-rnn   9282.0
3117         population ages 15-64   0.232059    0.082563   0.287338      0.981547                                           Honduras          rf   2054.0
2406         population ages 15-64   0.524831    0.415107   0.644288      0.907221                                           Honduras     xgboost   4394.0
1695         population ages 15-64   0.738192    0.653551   0.808425      0.853928                                           Honduras    base-gru   5420.0
984          population ages 15-64   0.873241    0.959223   0.979399      0.785608                                           Honduras   base-lstm   6207.0
273          population ages 15-64   1.007127    1.204896   1.097678      0.730699                                           Honduras  simple-rnn   6719.0
2408  population ages 65 and above   0.128706    0.024659   0.157031      0.824952                                           Honduras     xgboost   1471.0
3119  population ages 65 and above   0.166220    0.048077   0.219266      0.658708                                           Honduras          rf   2159.0
986   population ages 65 and above   0.951545    0.975400   0.987624     -5.924176                                           Honduras   base-lstm   8045.0
275   population ages 65 and above   1.000556    1.101963   1.049744     -6.822617                                           Honduras  simple-rnn   8312.0
1697  population ages 65 and above   1.139392    1.408934   1.186985     -9.001746                                           Honduras    base-gru   8898.0
3121          population ages 0-14   0.145446    0.023916   0.154649      0.973089                               Hong Kong SAR, China          rf   1135.0
2410          population ages 0-14   0.583447    0.647356   0.804584      0.271582                               Hong Kong SAR, China     xgboost   5738.0
988           population ages 0-14   1.176666    1.487848   1.219774     -0.674156                               Hong Kong SAR, China   base-lstm   8002.0
1699          population ages 0-14   1.274419    1.653542   1.285901     -0.860598                               Hong Kong SAR, China    base-gru   8282.0
277           population ages 0-14   1.309644    1.797525   1.340718     -1.022611                               Hong Kong SAR, China  simple-rnn   8464.0
2409         population ages 15-64   0.180898    0.046273   0.215112      0.788570                               Hong Kong SAR, China     xgboost   2034.0
3120         population ages 15-64   0.433893    0.282816   0.531805     -0.292235                               Hong Kong SAR, China          rf   4839.0
1698         population ages 15-64   0.870216    0.990590   0.995284     -3.526175                               Hong Kong SAR, China    base-gru   7740.0
276          population ages 15-64   1.076110    1.455242   1.206334     -5.649251                               Hong Kong SAR, China  simple-rnn   8677.0
987          population ages 15-64   1.338685    2.116938   1.454970     -8.672652                               Hong Kong SAR, China   base-lstm   9600.0
2411  population ages 65 and above   0.067555    0.006673   0.081690      0.971211                               Hong Kong SAR, China     xgboost    485.0
3122  population ages 65 and above   0.062908    0.007883   0.088789      0.965989                               Hong Kong SAR, China          rf    521.0
989   population ages 65 and above   0.278044    0.081845   0.286085      0.646908                               Hong Kong SAR, China   base-lstm   2841.0
1700  population ages 65 and above   0.662638    0.479840   0.692705     -1.070110                               Hong Kong SAR, China    base-gru   6076.0
278   population ages 65 and above   0.896777    1.017053   1.008491     -3.387737                               Hong Kong SAR, China  simple-rnn   7792.0
3124          population ages 0-14   0.133655    0.026510   0.162819     -1.533925                                            Hungary          rf   2748.0
2413          population ages 0-14   0.124020    0.028531   0.168912     -1.727103                                            Hungary     xgboost   2799.0
1702          population ages 0-14   3.372951   12.013392   3.466034  -1147.276892                                            Hungary    base-gru  13333.0
991           population ages 0-14   3.755107   14.969719   3.869072  -1429.851607                                            Hungary   base-lstm  13618.0
280           population ages 0-14   4.191866   18.609830   4.313911  -1777.784624                                            Hungary  simple-rnn  13814.0
3123         population ages 15-64   0.685366    0.772870   0.879130      0.529677                                            Hungary          rf   5956.0
2412         population ages 15-64   0.796779    0.935196   0.967055      0.430895                                            Hungary     xgboost   6418.0
1701         population ages 15-64   1.668608    3.908397   1.976967     -1.378419                                            Hungary    base-gru   9747.0
279          population ages 15-64   2.301931    7.038949   2.653102     -3.283489                                            Hungary  simple-rnn  10997.0
990          population ages 15-64   3.111025   11.823241   3.438494     -6.194926                                            Hungary   base-lstm  12067.0
3125  population ages 65 and above   0.171444    0.036821   0.191887      0.978524                                            Hungary          rf   1429.0
992   population ages 65 and above   0.491926    0.289497   0.538049      0.831151                                            Hungary   base-lstm   4157.0
281   population ages 65 and above   0.577635    0.401458   0.633607      0.765849                                            Hungary  simple-rnn   4766.0
2414  population ages 65 and above   0.514137    0.492105   0.701502      0.712979                                            Hungary     xgboost   4925.0
1703  population ages 65 and above   0.696127    0.591817   0.769297      0.654822                                            Hungary    base-gru   5524.0
3127          population ages 0-14   0.080541    0.010476   0.102353      0.979983                                          IBRD only          rf    588.0
2416          population ages 0-14   0.229463    0.091461   0.302426      0.825244                                          IBRD only     xgboost   2551.0
1705          population ages 0-14   3.045824   10.440713   3.231209    -18.949186                                          IBRD only    base-gru  12316.0
994           population ages 0-14   3.191293   11.619165   3.408690    -21.200868                                          IBRD only   base-lstm  12536.0
283           population ages 0-14   3.266242   12.029820   3.468403    -21.985511                                          IBRD only  simple-rnn  12611.0
3126         population ages 15-64   0.107912    0.019408   0.139314      0.386723                                          IBRD only          rf   1764.0
2415         population ages 15-64   0.172332    0.047261   0.217395     -0.493362                                          IBRD only     xgboost   2821.0
1704         population ages 15-64   1.337655    2.174666   1.474675    -67.715913                                          IBRD only    base-gru  10263.0
282          population ages 15-64   1.608926    3.123631   1.767380    -97.701659                                          IBRD only  simple-rnn  10953.0
993          population ages 15-64   1.997279    4.869160   2.206617   -152.857522                                          IBRD only   base-lstm  11774.0
3128  population ages 65 and above   0.361128    0.263832   0.513645      0.632581                                          IBRD only          rf   4077.0
2417  population ages 65 and above   0.667315    0.840454   0.916763     -0.170438                                          IBRD only     xgboost   6427.0
995   population ages 65 and above   1.486214    2.313030   1.520865     -2.221186                                          IBRD only   base-lstm   9208.0
284   population ages 65 and above   1.691191    3.023699   1.738879     -3.210882                                          IBRD only  simple-rnn   9813.0
1706  population ages 65 and above   1.762099    3.271114   1.808622     -3.555438                                          IBRD only    base-gru  10021.0
2419          population ages 0-14   0.285242    0.097912   0.312908      0.721926                                   IDA & IBRD total     xgboost   2926.0
3130          population ages 0-14   0.621603    0.482025   0.694280     -0.368975                                   IDA & IBRD total          rf   5771.0
1708          population ages 0-14   2.927512    9.719300   3.117579    -26.603312                                   IDA & IBRD total    base-gru  12344.0
997           population ages 0-14   3.051165   10.656343   3.264405    -29.264561                                   IDA & IBRD total   base-lstm  12490.0
286           population ages 0-14   3.158415   11.302441   3.361910    -31.099511                                   IDA & IBRD total  simple-rnn  12623.0
2418         population ages 15-64   0.876475    1.195026   1.093172   -136.398277                                   IDA & IBRD total     xgboost   9133.0
3129         population ages 15-64   0.894279    1.215913   1.102684   -138.799798                                   IDA & IBRD total          rf   9195.0
1707         population ages 15-64   1.256543    1.961303   1.400465   -224.501119                                   IDA & IBRD total    base-gru  10269.0
285          population ages 15-64   1.611300    3.159090   1.777383   -362.216897                                   IDA & IBRD total  simple-rnn  11161.0
996          population ages 15-64   1.804153    4.040841   2.010184   -463.596345                                   IDA & IBRD total   base-lstm  11611.0
3131  population ages 65 and above   0.176588    0.066164   0.257224      0.828885                                   IDA & IBRD total          rf   2165.0
2420  population ages 65 and above   0.349987    0.218331   0.467259      0.435347                                   IDA & IBRD total     xgboost   4048.0
998   population ages 65 and above   1.226992    1.590820   1.261277     -3.114218                                   IDA & IBRD total   base-lstm   8700.0
287   population ages 65 and above   1.391944    2.069087   1.438432     -4.351122                                   IDA & IBRD total  simple-rnn   9308.0
1709  population ages 65 and above   1.512838    2.438119   1.561448     -5.305521                                   IDA & IBRD total    base-gru   9679.0
2422          population ages 0-14   0.410913    0.263938   0.513749      0.119668                                          IDA blend     xgboost   4528.0
3133          population ages 0-14   0.486641    0.346817   0.588912     -0.156762                                          IDA blend          rf   5073.0
1711          population ages 0-14   1.185628    1.693718   1.301429     -4.649176                                          IDA blend    base-gru   8922.0
1000          population ages 0-14   1.299367    2.041758   1.428901     -5.810017                                          IDA blend   base-lstm   9341.0
289           population ages 0-14   1.554123    2.854202   1.689438     -8.519818                                          IDA blend  simple-rnn  10082.0
1710         population ages 15-64   0.243577    0.076861   0.277238      0.648836                                          IDA blend    base-gru   2707.0
2421         population ages 15-64   0.364671    0.186669   0.432052      0.147141                                          IDA blend     xgboost   4097.0
999          population ages 15-64   0.436388    0.269771   0.519395     -0.232541                                          IDA blend   base-lstm   4790.0
288          population ages 15-64   0.593237    0.484805   0.696279     -1.214996                                          IDA blend  simple-rnn   6021.0
3132         population ages 15-64   0.572052    0.513056   0.716279     -1.344071                                          IDA blend          rf   6078.0
3134  population ages 65 and above   0.017683    0.000503   0.022428      0.921632                                          IDA blend          rf    316.0
2423  population ages 65 and above   0.082281    0.009962   0.099810     -0.552129                                          IDA blend     xgboost   1897.0
1001  population ages 65 and above   0.736766    0.606942   0.779065    -93.564159                                          IDA blend   base-lstm   8068.0
290   population ages 65 and above   0.741991    0.624074   0.789983    -96.233324                                          IDA blend  simple-rnn   8101.0
1712  population ages 65 and above   0.866243    0.841748   0.917468   -130.147808                                          IDA blend    base-gru   8688.0
3136          population ages 0-14   0.305762    0.126886   0.356211      0.807959                                           IDA only          rf   3073.0
2425          population ages 0-14   0.375557    0.216192   0.464965      0.672794                                           IDA only     xgboost   3887.0
1714          population ages 0-14   1.083000    1.293346   1.137254     -0.957470                                           IDA only    base-gru   7775.0
1003          population ages 0-14   1.172136    1.522611   1.233941     -1.304462                                           IDA only   base-lstm   8241.0
292           population ages 0-14   1.398436    2.160029   1.469704     -2.269190                                           IDA only  simple-rnn   9082.0
2424         population ages 15-64   0.264638    0.087364   0.295574      0.808446                                           IDA only     xgboost   2656.0
1713         population ages 15-64   0.281094    0.093812   0.306287      0.794308                                           IDA only    base-gru   2789.0
1002         population ages 15-64   0.467068    0.259985   0.509888      0.429955                                           IDA only   base-lstm   4463.0
3135         population ages 15-64   0.487752    0.325489   0.570517      0.286331                                           IDA only          rf   4783.0
291          population ages 15-64   0.642226    0.473797   0.688329     -0.038848                                           IDA only  simple-rnn   5627.0
2426  population ages 65 and above   0.182972    0.053824   0.231999     -1.827814                                           IDA only     xgboost   3358.0
3137  population ages 65 and above   0.180944    0.058080   0.240997     -2.051418                                           IDA only          rf   3452.0
1004  population ages 65 and above   0.592051    0.379939   0.616392    -18.961389                                           IDA only   base-lstm   6875.0
293   population ages 65 and above   0.598958    0.395017   0.628504    -19.753557                                           IDA only  simple-rnn   6926.0
1715  population ages 65 and above   0.723771    0.571444   0.755939    -29.022764                                           IDA only    base-gru   7687.0
3139          population ages 0-14   0.294940    0.125625   0.354436      0.780581                                          IDA total          rf   3087.0
2428          population ages 0-14   0.573040    0.462730   0.680243      0.191790                                          IDA total     xgboost   5357.0
1717          population ages 0-14   1.084681    1.346765   1.160502     -1.352277                                          IDA total    base-gru   7962.0
1006          population ages 0-14   1.190027    1.628778   1.276236     -1.844843                                          IDA total   base-lstm   8464.0
295           population ages 0-14   1.428854    2.322653   1.524025     -3.056774                                          IDA total  simple-rnn   9304.0
2427         population ages 15-64   0.209108    0.051855   0.227717      0.871726                                          IDA total     xgboost   2035.0
1716         population ages 15-64   0.226973    0.069411   0.263460      0.828298                                          IDA total    base-gru   2343.0
1005         population ages 15-64   0.428023    0.241091   0.491010      0.403612                                          IDA total   base-lstm   4307.0
294          population ages 15-64   0.610698    0.456715   0.675807     -0.129778                                          IDA total  simple-rnn   5565.0
3138         population ages 15-64   0.608316    0.524853   0.724467     -0.298333                                          IDA total          rf   5812.0
3140  population ages 65 and above   0.048887    0.004100   0.064031      0.722027                                          IDA total          rf    904.0
2429  population ages 65 and above   0.130896    0.030890   0.175755     -1.094281                                          IDA total     xgboost   2684.0
1007  population ages 65 and above   0.636618    0.447442   0.668911    -29.335837                                          IDA total   base-lstm   7261.0
296   population ages 65 and above   0.641851    0.462192   0.679847    -30.335850                                          IDA total  simple-rnn   7320.0
1718  population ages 65 and above   0.765851    0.651251   0.807001    -43.153774                                          IDA total    base-gru   8025.0
2431          population ages 0-14   0.169711    0.047888   0.218833      0.931372                                            Iceland     xgboost   1692.0
3142          population ages 0-14   0.633036    0.705536   0.839962     -0.011101                                            Iceland          rf   6108.0
1720          population ages 0-14   1.773295    3.235122   1.798644     -3.636244                                            Iceland    base-gru  10025.0
298           population ages 0-14   2.245690    5.261231   2.293737     -6.539857                                            Iceland  simple-rnn  11028.0
1009          population ages 0-14   2.308812    5.599455   2.366317     -7.024565                                            Iceland   base-lstm  11141.0
297          population ages 15-64   0.416055    0.225531   0.474902     -1.137305                                            Iceland  simple-rnn   4886.0
1719         population ages 15-64   0.468661    0.317709   0.563657     -2.010847                                            Iceland    base-gru   5531.0
2430         population ages 15-64   0.563569    0.530619   0.728436     -4.028546                                            Iceland     xgboost   6575.0
1008         population ages 15-64   0.708013    0.592995   0.770062     -4.619669                                            Iceland   base-lstm   7016.0
3141         population ages 15-64   0.725391    0.635893   0.797429     -5.026198                                            Iceland          rf   7173.0
2432  population ages 65 and above   0.127837    0.023681   0.153887      0.972721                                            Iceland     xgboost   1069.0
3143  population ages 65 and above   0.856334    0.887352   0.941994     -0.022167                                            Iceland          rf   6674.0
299   population ages 65 and above   1.216316    1.568229   1.252290     -0.806489                                            Iceland  simple-rnn   8149.0
1010  population ages 65 and above   1.276927    1.732118   1.316100     -0.995278                                            Iceland   base-lstm   8379.0
1721  population ages 65 and above   1.484789    2.331027   1.526770     -1.685179                                            Iceland    base-gru   9099.0
3145          population ages 0-14   0.317705    0.134367   0.366561      0.959436                                              India          rf   2776.0
2434          population ages 0-14   0.771619    0.941989   0.970561      0.715624                                              India     xgboost   6140.0
1723          population ages 0-14   3.534184   13.980119   3.739000     -3.220448                                              India    base-gru  12067.0
1012          population ages 0-14   3.698093   15.406698   3.925137     -3.651116                                              India   base-lstm  12233.0
301           population ages 0-14   3.718253   15.431905   3.928346     -3.658726                                              India  simple-rnn  12243.0
3144         population ages 15-64   0.249400    0.089255   0.298756      0.940333                                              India          rf   2333.0
2433         population ages 15-64   0.679439    0.633441   0.795890      0.576546                                              India     xgboost   5662.0
1722         population ages 15-64   1.603385    3.035911   1.742386     -1.029500                                              India    base-gru   9274.0
300          population ages 15-64   1.875080    4.122568   2.030411     -1.755929                                              India  simple-rnn  10044.0
1011         population ages 15-64   2.343156    6.624366   2.573784     -3.428375                                              India   base-lstm  10997.0
3146  population ages 65 and above   0.214259    0.098037   0.313109      0.734089                                              India          rf   2718.0
2435  population ages 65 and above   0.315354    0.213108   0.461636      0.421978                                              India     xgboost   3960.0
1013  population ages 65 and above   1.370848    2.048376   1.431215     -4.555895                                              India   base-lstm   9295.0
302   population ages 65 and above   1.633003    2.949311   1.717356     -6.999537                                              India  simple-rnn  10093.0
1724  population ages 65 and above   1.817848    3.614487   1.901180     -8.803720                                              India    base-gru  10538.0
2437          population ages 0-14   0.274457    0.099694   0.315743      0.846241                                          Indonesia     xgboost   2697.0
3148          population ages 0-14   0.286758    0.121819   0.349026      0.812117                                          Indonesia          rf   2960.0
1726          population ages 0-14   3.265817   11.777460   3.431830    -17.164518                                          Indonesia    base-gru  12495.0
1015          population ages 0-14   3.349623   12.538232   3.540937    -18.337867                                          Indonesia   base-lstm  12596.0
304           population ages 0-14   3.504217   13.583550   3.685587    -19.950073                                          Indonesia  simple-rnn  12751.0
3147         population ages 15-64   0.183029    0.072186   0.268674      0.735565                                          Indonesia          rf   2390.0
2436         population ages 15-64   0.381191    0.236557   0.486371      0.133428                                          Indonesia     xgboost   4350.0
1725         population ages 15-64   1.351599    1.870063   1.367502     -5.850548                                          Indonesia    base-gru   9298.0
303          population ages 15-64   1.717032    3.035252   1.742197    -10.118950                                          Indonesia  simple-rnn  10312.0
1014         population ages 15-64   1.918569    3.900604   1.974995    -13.288971                                          Indonesia   base-lstm  10841.0
2438  population ages 65 and above   0.121319    0.020239   0.142263      0.755186                                          Indonesia     xgboost   1486.0
3149  population ages 65 and above   0.289913    0.105693   0.325105     -0.278493                                          Indonesia          rf   3630.0
1016  population ages 65 and above   1.742176    3.549103   1.883906    -41.930940                                          Indonesia   base-lstm  10996.0
305   population ages 65 and above   1.945009    4.431117   2.105022    -52.600017                                          Indonesia  simple-rnn  11447.0
1727  population ages 65 and above   2.033381    4.781047   2.186561    -56.832875                                          Indonesia    base-gru  11579.0
2440          population ages 0-14   1.569339    3.380553   1.838628     -5.112086                                 Iran, Islamic Rep.     xgboost  10069.0
3151          population ages 0-14   2.358889    7.910807   2.812616    -13.302848                                 Iran, Islamic Rep.          rf  11725.0
1018          population ages 0-14   5.013888   32.803590   5.727442    -58.309341                                 Iran, Islamic Rep.   base-lstm  13595.0
307           population ages 0-14   5.193700   34.477591   5.871762    -61.335959                                 Iran, Islamic Rep.  simple-rnn  13625.0
1729          population ages 0-14   5.248732   35.209739   5.933779    -62.659691                                 Iran, Islamic Rep.    base-gru  13639.0
2439         population ages 15-64   3.226670   18.255323   4.272625     -7.329241                                 Iran, Islamic Rep.     xgboost  12506.0
306          population ages 15-64   4.752311   28.715785   5.358711    -12.101970                                 Iran, Islamic Rep.  simple-rnn  13072.0
1728         population ages 15-64   4.875632   29.922850   5.470178    -12.652710                                 Iran, Islamic Rep.    base-gru  13104.0
1017         population ages 15-64   5.165056   34.101017   5.839608    -14.559056                                 Iran, Islamic Rep.   base-lstm  13191.0
3150         population ages 15-64   5.392844   53.157016   7.290886    -23.253616                                 Iran, Islamic Rep.          rf  13422.0
2441  population ages 65 and above   0.075937    0.010432   0.102138      0.982960                                 Iran, Islamic Rep.     xgboost    551.0
3152  population ages 65 and above   0.239457    0.113760   0.337283      0.814184                                 Iran, Islamic Rep.          rf   2770.0
1019  population ages 65 and above   1.522704    2.428849   1.558476     -2.967306                                 Iran, Islamic Rep.   base-lstm   9415.0
1730  population ages 65 and above   1.682979    2.984618   1.727605     -3.875104                                 Iran, Islamic Rep.    base-gru   9894.0
308   population ages 65 and above   1.820788    3.541418   1.881866     -4.784588                                 Iran, Islamic Rep.  simple-rnn  10262.0
2443          population ages 0-14   0.115277    0.018233   0.135030      0.982998                                               Iraq     xgboost    878.0
3154          population ages 0-14   0.311170    0.134269   0.366428      0.874796                                               Iraq          rf   2991.0
1732          population ages 0-14   1.982347    4.349732   2.085601     -3.056047                                               Iraq    base-gru  10409.0
1021          population ages 0-14   2.222860    5.533563   2.352353     -4.159947                                               Iraq   base-lstm  10871.0
310           population ages 0-14   2.592814    7.479105   2.734795     -5.974131                                               Iraq  simple-rnn  11452.0
2442         population ages 15-64   0.060955    0.004200   0.064805      0.994901                                               Iraq     xgboost    275.0
3153         population ages 15-64   0.239788    0.078731   0.280590      0.904407                                               Iraq          rf   2306.0
1731         population ages 15-64   1.016719    1.145283   1.070179     -0.390574                                               Iraq    base-gru   7351.0
1020         population ages 15-64   1.339467    2.061797   1.435896     -1.503381                                               Iraq   base-lstm   8784.0
309          population ages 15-64   1.746188    3.430287   1.852103     -3.164968                                               Iraq  simple-rnn   9999.0
2444  population ages 65 and above   0.109227    0.025551   0.159846     -0.413688                                               Iraq     xgboost   2271.0
3155  population ages 65 and above   0.104765    0.025983   0.161192     -0.437606                                               Iraq          rf   2275.0
311   population ages 65 and above   0.817792    0.722334   0.849902    -38.965811                                               Iraq  simple-rnn   8221.0
1022  population ages 65 and above   0.898548    0.859664   0.927180    -46.564088                                               Iraq   base-lstm   8561.0
1733  population ages 65 and above   1.002748    1.078592   1.038553    -58.677131                                               Iraq    base-gru   8984.0
2446          population ages 0-14   0.248227    0.082809   0.287766      0.681158                                            Ireland     xgboost   2733.0
3157          population ages 0-14   0.634374    0.533609   0.730485     -1.054563                                            Ireland          rf   6134.0
1735          population ages 0-14   4.706730   23.078452   4.804004    -87.859409                                            Ireland    base-gru  13583.0
1024          population ages 0-14   5.032701   26.580482   5.155626   -101.343343                                            Ireland   base-lstm  13670.0
313           population ages 0-14   5.101632   27.319342   5.226791   -104.188190                                            Ireland  simple-rnn  13691.0
2445         population ages 15-64   1.015575    1.352320   1.162893     -0.530921                                            Ireland     xgboost   7630.0
3156         population ages 15-64   1.329650    1.960174   1.400062     -1.219054                                            Ireland          rf   8657.0
1734         population ages 15-64   2.324281    6.036789   2.456988     -5.834065                                            Ireland    base-gru  11123.0
312          population ages 15-64   2.874177    9.256612   3.042468     -9.479129                                            Ireland  simple-rnn  11891.0
1023         population ages 15-64   3.429054   13.111330   3.620957    -13.842939                                            Ireland   base-lstm  12567.0
2447  population ages 65 and above   0.174013    0.054956   0.234427      0.960304                                            Ireland     xgboost   1729.0
3158  population ages 65 and above   0.893822    1.022519   1.011197      0.261418                                            Ireland          rf   6751.0
1025  population ages 65 and above   1.441235    2.154238   1.467732     -0.556040                                            Ireland   base-lstm   8623.0
314   population ages 65 and above   1.580009    2.591068   1.609679     -0.871570                                            Ireland  simple-rnn   9050.0
1736  population ages 65 and above   1.896088    3.728278   1.930875     -1.692995                                            Ireland    base-gru   9908.0
3160          population ages 0-14   0.253729    0.157030   0.396270      0.196326                                        Isle of Man          rf   3637.0
2449          population ages 0-14   0.475395    0.399169   0.631799     -1.042941                                        Isle of Man     xgboost   5512.0
1738          population ages 0-14   3.052315    9.571363   3.093762    -47.986026                                        Isle of Man    base-gru  12535.0
1027          population ages 0-14   3.365330   11.708740   3.421804    -58.925078                                        Isle of Man   base-lstm  12871.0
316           population ages 0-14   3.843062   15.250104   3.905138    -77.049701                                        Isle of Man  simple-rnn  13269.0
2448         population ages 15-64   0.064410    0.006647   0.081532      0.944511                                        Isle of Man     xgboost    559.0
3159         population ages 15-64   0.425347    0.241175   0.491096     -1.013196                                        Isle of Man          rf   4908.0
1737         population ages 15-64   2.941116    9.152170   3.025255    -75.397178                                        Isle of Man    base-gru  12545.0
315          population ages 15-64   3.646018   14.010944   3.743120   -115.955494                                        Isle of Man  simple-rnn  13266.0
1026         population ages 15-64   4.204764   18.598067   4.312548   -154.246214                                        Isle of Man   base-lstm  13561.0
3161  population ages 65 and above   0.652969    0.542841   0.736778      0.046142                                        Isle of Man          rf   5775.0
2450  population ages 65 and above   0.985840    1.476094   1.214946     -1.593731                                        Isle of Man     xgboost   8052.0
1739  population ages 65 and above   1.206615    1.500591   1.224986     -1.636776                                        Isle of Man    base-gru   8328.0
317   population ages 65 and above   1.437019    2.105998   1.451206     -2.700570                                        Isle of Man  simple-rnn   9124.0
1028  population ages 65 and above   1.575820    2.545400   1.595431     -3.472670                                        Isle of Man   base-lstm   9612.0
3550          population ages 0-14   0.399665    0.202374   0.449860     -4.374449                                             Israel          rf   5339.0
2839          population ages 0-14   0.456224    0.243475   0.493432     -5.465973                                             Israel     xgboost   5708.0
2128          population ages 0-14   3.779856   15.410746   3.925652   -408.263491                                             Israel    base-gru  13530.0
706           population ages 0-14   4.166195   18.700475   4.324405   -495.628891                                             Israel  simple-rnn  13719.0
1417          population ages 0-14   4.307966   20.163010   4.490324   -534.469457                                             Israel   base-lstm  13769.0
2127         population ages 15-64   1.165522    1.837016   1.355366     -1.573650                                             Israel    base-gru   8513.0
2838         population ages 15-64   1.203747    2.093046   1.446736     -1.932346                                             Israel     xgboost   8779.0
3549         population ages 15-64   1.301472    2.376559   1.541609     -2.329547                                             Israel          rf   9122.0
705          population ages 15-64   1.924934    4.628100   2.151302     -5.483944                                             Israel  simple-rnn  10694.0
1416         population ages 15-64   2.024343    5.200499   2.280460     -6.285872                                             Israel   base-lstm  10924.0
3551  population ages 65 and above   0.874428    1.176224   1.084539     -1.650867                                             Israel          rf   7610.0
2840  population ages 65 and above   0.964093    1.371892   1.171278     -2.091847                                             Israel     xgboost   8033.0
707   population ages 65 and above   1.280233    1.667280   1.291232     -2.757564                                             Israel  simple-rnn   8744.0
1418  population ages 65 and above   1.384316    1.957316   1.399041     -3.411223                                             Israel   base-lstm   9133.0
2129  population ages 65 and above   1.603804    2.638581   1.624371     -4.946596                                             Israel    base-gru   9812.0
2452          population ages 0-14   0.252743    0.140076   0.374267      0.384150                                              Italy     xgboost   3455.0
3163          population ages 0-14   0.502395    0.477976   0.691358     -1.101451                                              Italy          rf   5784.0
1741          population ages 0-14   2.857995    8.281123   2.877694    -35.408433                                              Italy    base-gru  12247.0
1030          population ages 0-14   3.245307   10.741981   3.277496    -46.227736                                              Italy   base-lstm  12694.0
319           population ages 0-14   3.906295   15.569636   3.945838    -67.452804                                              Italy  simple-rnn  13267.0
3162         population ages 15-64   0.232550    0.076151   0.275954      0.818314                                              Italy          rf   2436.0
2451         population ages 15-64   1.065687    1.561384   1.249553     -2.725259                                              Italy     xgboost   8438.0
1740         population ages 15-64   2.194716    5.480871   2.341126    -12.076645                                              Italy    base-gru  11275.0
318          population ages 15-64   3.200208   11.361060   3.370617    -26.106014                                              Italy  simple-rnn  12595.0
1029         population ages 15-64   3.817557   15.906861   3.988341    -36.951706                                              Italy   base-lstm  13127.0
1742  population ages 65 and above   0.563667    0.483994   0.695697      0.594920                                              Italy    base-gru   5110.0
320   population ages 65 and above   0.674873    0.733614   0.856513      0.386000                                              Italy  simple-rnn   6015.0
3164  population ages 65 and above   0.730315    0.982450   0.991186      0.177736                                              Italy          rf   6529.0
1031  population ages 65 and above   0.812317    1.048077   1.023756      0.122810                                              Italy   base-lstm   6749.0
2453  population ages 65 and above   1.488655    3.335559   1.826351     -1.791702                                              Italy     xgboost   9534.0
3166          population ages 0-14   0.122927    0.020407   0.142853      0.992919                                            Jamaica          rf    895.0
2455          population ages 0-14   0.335165    0.163099   0.403855      0.943407                                            Jamaica     xgboost   3061.0
1744          population ages 0-14   1.774051    3.744003   1.934943     -0.299125                                            Jamaica    base-gru   9389.0
1033          population ages 0-14   1.769286    3.807891   1.951382     -0.321294                                            Jamaica   base-lstm   9418.0
322           population ages 0-14   1.844516    4.017109   2.004273     -0.393890                                            Jamaica  simple-rnn   9568.0
2454         population ages 15-64   0.148526    0.031482   0.177431      0.981616                                            Jamaica     xgboost   1254.0
1743         population ages 15-64   0.316448    0.148044   0.384765      0.913547                                            Jamaica    base-gru   3012.0
321          population ages 15-64   0.322570    0.152574   0.390607      0.910902                                            Jamaica  simple-rnn   3061.0
3165         population ages 15-64   0.477618    0.326294   0.571222      0.809455                                            Jamaica          rf   4293.0
1032         population ages 15-64   0.687369    0.687168   0.828956      0.598717                                            Jamaica   base-lstm   5762.0
3167  population ages 65 and above   0.173618    0.042994   0.207351      0.734276                                            Jamaica          rf   2058.0
2456  population ages 65 and above   0.429411    0.256879   0.506832     -0.587629                                            Jamaica     xgboost   4862.0
1034  population ages 65 and above   1.741522    3.399750   1.843841    -20.012004                                            Jamaica   base-lstm  10723.0
323   population ages 65 and above   1.923787    4.172425   2.042652    -24.787494                                            Jamaica  simple-rnn  11161.0
1745  population ages 65 and above   1.965252    4.289130   2.071021    -25.508781                                            Jamaica    base-gru  11235.0
3169          population ages 0-14   0.687798    0.706597   0.840593     -2.057525                                              Japan          rf   6860.0
1747          population ages 0-14   0.863498    0.769830   0.877400     -2.331141                                              Japan    base-gru   7247.0
2458          population ages 0-14   0.842661    1.022537   1.011206     -3.424633                                              Japan     xgboost   7727.0
1036          population ages 0-14   1.196065    1.490866   1.221010     -5.451142                                              Japan   base-lstm   8846.0
325           population ages 0-14   1.818106    3.413352   1.847526    -13.769957                                              Japan  simple-rnn  10649.0
2457         population ages 15-64   1.996801    6.443319   2.538369     -0.893888                                              Japan     xgboost  10257.0
3168         population ages 15-64   2.820068   12.282098   3.504582     -2.610084                                              Japan          rf  11607.0
1746         population ages 15-64   4.844558   27.194005   5.214787     -6.993150                                              Japan    base-gru  12853.0
324          population ages 15-64   5.869057   39.090792   6.252263    -10.489979                                              Japan  simple-rnn  13118.0
1035         population ages 15-64   6.376163   45.487439   6.744438    -12.370149                                              Japan   base-lstm  13198.0
3170  population ages 65 and above   3.622503   18.413648   4.291113     -2.459934                                              Japan          rf  12127.0
2459  population ages 65 and above   4.762318   31.072968   5.574313     -4.838627                                              Japan     xgboost  12727.0
1748  population ages 65 and above   5.166363   30.427137   5.516080     -4.717275                                              Japan    base-gru  12735.0
326   population ages 65 and above   5.262816   31.551704   5.617090     -4.928581                                              Japan  simple-rnn  12767.0
1037  population ages 65 and above   5.485970   34.016643   5.832379     -5.391745                                              Japan   base-lstm  12813.0
3172          population ages 0-14   0.265119    0.086810   0.294636      0.905604                                             Jordan          rf   2447.0
2461          population ages 0-14   0.521490    0.439773   0.663154      0.521798                                             Jordan     xgboost   4980.0
1750          population ages 0-14   1.840430    4.064326   2.016017     -3.419490                                             Jordan    base-gru  10301.0
1039          population ages 0-14   1.910346    4.474892   2.115394     -3.865933                                             Jordan   base-lstm  10513.0
328           population ages 0-14   2.347053    6.550857   2.559464     -6.123308                                             Jordan  simple-rnn  11234.0
3171         population ages 15-64   0.126871    0.017950   0.133979      0.972419                                             Jordan          rf    967.0
2460         population ages 15-64   0.120825    0.019313   0.138971      0.970326                                             Jordan     xgboost    981.0
1749         population ages 15-64   0.928279    1.344634   1.159584     -1.066018                                             Jordan    base-gru   7692.0
1038         population ages 15-64   1.323030    2.544294   1.595084     -2.909285                                             Jordan   base-lstm   9323.0
327          population ages 15-64   1.519836    3.101955   1.761237     -3.766125                                             Jordan  simple-rnn   9816.0
3173  population ages 65 and above   0.200810    0.055189   0.234923     -0.458409                                             Jordan          rf   3019.0
2462  population ages 65 and above   0.302928    0.145414   0.381332     -2.842680                                             Jordan     xgboost   4629.0
1040  population ages 65 and above   0.845688    0.738911   0.859599    -18.526254                                             Jordan   base-lstm   8038.0
329   population ages 65 and above   1.026968    1.095282   1.046557    -27.943641                                             Jordan  simple-rnn   8851.0
1751  population ages 65 and above   1.123232    1.306442   1.142997    -33.523688                                             Jordan    base-gru   9240.0
2464          population ages 0-14   0.045486    0.002907   0.053913      0.993709                                         Kazakhstan     xgboost    199.0
3175          population ages 0-14   1.041258    1.330468   1.153459     -1.879682                                         Kazakhstan          rf   8022.0
1042          population ages 0-14   5.055389   26.556439   5.153294    -56.479090                                         Kazakhstan   base-lstm  13541.0
1753          population ages 0-14   5.085084   26.770651   5.174036    -56.942734                                         Kazakhstan    base-gru  13552.0
331           population ages 0-14   5.189995   27.985034   5.290088    -59.571160                                         Kazakhstan  simple-rnn  13582.0
2463         population ages 15-64   0.099588    0.021120   0.145327      0.977191                                         Kazakhstan     xgboost    914.0
3174         population ages 15-64   0.786797    0.804523   0.896952      0.131157                                         Kazakhstan          rf   6400.0
1752         population ages 15-64   3.805060   15.315865   3.913549    -15.540327                                         Kazakhstan    base-gru  12804.0
330          population ages 15-64   4.259807   19.122093   4.372881    -19.650852                                         Kazakhstan  simple-rnn  13084.0
1041         population ages 15-64   4.287826   19.518464   4.417971    -20.078913                                         Kazakhstan   base-lstm  13102.0
2465  population ages 65 and above   0.054089    0.004361   0.066038      0.945558                                         Kazakhstan     xgboost    465.0
3176  population ages 65 and above   0.146240    0.027841   0.166857      0.652437                                         Kazakhstan          rf   1844.0
1043  population ages 65 and above   0.825787    0.693359   0.832682     -7.655761                                         Kazakhstan   base-lstm   7622.0
332   population ages 65 and above   0.959132    0.930247   0.964493    -10.613028                                         Kazakhstan  simple-rnn   8220.0
1754  population ages 65 and above   1.159662    1.355779   1.164379    -15.925285                                         Kazakhstan    base-gru   9087.0
3178          population ages 0-14   0.343363    0.202465   0.449961      0.923871                                              Kenya          rf   3310.0
2467          population ages 0-14   0.388682    0.253761   0.503747      0.904583                                              Kenya     xgboost   3668.0
1045          population ages 0-14   1.849163    3.672713   1.916432     -0.380973                                              Kenya   base-lstm   9442.0
1756          population ages 0-14   1.914174    3.935983   1.983931     -0.479965                                              Kenya    base-gru   9616.0
334           population ages 0-14   2.099470    4.731979   2.175311     -0.779267                                              Kenya  simple-rnn  10001.0
3177         population ages 15-64   0.404968    0.268871   0.518528      0.858804                                              Kenya          rf   3856.0
2466         population ages 15-64   0.474728    0.313287   0.559720      0.835479                                              Kenya     xgboost   4183.0
1044         population ages 15-64   1.228372    1.653323   1.285816      0.131766                                              Kenya   base-lstm   7815.0
1755         population ages 15-64   1.262759    1.724235   1.313101      0.094527                                              Kenya    base-gru   7935.0
333          population ages 15-64   1.437453    2.241748   1.497247     -0.177243                                              Kenya  simple-rnn   8503.0
2468  population ages 65 and above   0.291561    0.123901   0.351995     -0.888065                                              Kenya     xgboost   3980.0
3179  population ages 65 and above   0.349204    0.173274   0.416262     -1.640449                                              Kenya          rf   4653.0
335   population ages 65 and above   0.659730    0.465218   0.682069     -6.089244                                              Kenya  simple-rnn   6773.0
1046  population ages 65 and above   0.667335    0.468912   0.684771     -6.145530                                              Kenya   base-lstm   6805.0
1757  population ages 65 and above   0.772028    0.633147   0.795705     -8.648235                                              Kenya    base-gru   7462.0
2470          population ages 0-14   0.763985    1.091036   1.044527    -26.298772                                           Kiribati     xgboost   8499.0
3181          population ages 0-14   1.358344    3.096515   1.759692    -76.477783                                           Kiribati          rf  10704.0
1048          population ages 0-14   1.926572    5.719887   2.391629   -142.117087                                           Kiribati   base-lstm  11849.0
1759          population ages 0-14   2.020596    6.090636   2.467921   -151.393579                                           Kiribati    base-gru  11970.0
337           population ages 0-14   2.185332    7.039484   2.653203   -175.134675                                           Kiribati  simple-rnn  12182.0
2469         population ages 15-64   0.883381    1.337263   1.156401    -12.511316                                           Kiribati     xgboost   8649.0
1758         population ages 15-64   1.079825    2.066022   1.437366    -19.874483                                           Kiribati    base-gru   9582.0
1047         population ages 15-64   1.168297    2.411957   1.553048    -23.369717                                           Kiribati   base-lstm   9954.0
336          population ages 15-64   1.271294    2.864046   1.692349    -27.937493                                           Kiribati  simple-rnn  10297.0
3180         population ages 15-64   1.390305    3.307987   1.818787    -32.422947                                           Kiribati          rf  10607.0
2471  population ages 65 and above   0.125547    0.029454   0.171621     -0.442470                                           Kiribati     xgboost   2419.0
3182  population ages 65 and above   0.131650    0.035413   0.188182     -0.734298                                           Kiribati          rf   2659.0
1049  population ages 65 and above   0.943287    0.958232   0.978893    -45.928648                                           Kiribati   base-lstm   8722.0
338   population ages 65 and above   1.025395    1.142716   1.068979    -54.963594                                           Kiribati  simple-rnn   9069.0
1760  population ages 65 and above   1.113040    1.333312   1.154691    -64.297842                                           Kiribati    base-gru   9407.0
1762          population ages 0-14   0.224531    0.062535   0.250069      0.975182                                        Korea, Rep.    base-gru   1899.0
1051          population ages 0-14   0.338663    0.123833   0.351899      0.950854                                        Korea, Rep.   base-lstm   2780.0
340           population ages 0-14   0.391390    0.156421   0.395500      0.937921                                        Korea, Rep.  simple-rnn   3156.0
3184          population ages 0-14   1.669993    3.852321   1.962733     -0.528870                                        Korea, Rep.          rf   9447.0
2473          population ages 0-14   1.985522    5.969894   2.443337     -1.369271                                        Korea, Rep.     xgboost  10340.0
1761         population ages 15-64   0.441786    0.314161   0.560501     -0.107748                                        Korea, Rep.    base-gru   4884.0
339          population ages 15-64   1.085139    1.290359   1.135940     -3.549867                                        Korea, Rep.  simple-rnn   8338.0
1050         population ages 15-64   1.705891    3.120300   1.766437    -10.002326                                        Korea, Rep.   base-lstm  10349.0
2472         population ages 15-64   1.817311    6.623148   2.573548    -22.353537                                        Korea, Rep.     xgboost  11489.0
3183         population ages 15-64   2.710018   20.437028   4.520733    -71.061940                                        Korea, Rep.          rf  13101.0
2474  population ages 65 and above   0.235054    0.081345   0.285210      0.975103                                        Korea, Rep.     xgboost   2066.0
1763  population ages 65 and above   0.235693    0.119969   0.346366      0.963281                                        Korea, Rep.    base-gru   2428.0
341   population ages 65 and above   0.233084    0.135640   0.368293      0.958485                                        Korea, Rep.  simple-rnn   2561.0
1052  population ages 65 and above   0.249708    0.176506   0.420126      0.945977                                        Korea, Rep.   base-lstm   2885.0
3185  population ages 65 and above   0.963768    1.786263   1.336512      0.453284                                        Korea, Rep.          rf   7441.0
2476          population ages 0-14   0.366328    0.251867   0.501863      0.899072                                             Kuwait     xgboost   3620.0
3187          population ages 0-14   0.472948    0.292667   0.540987      0.882722                                             Kuwait          rf   4041.0
1054          population ages 0-14   0.865226    0.853192   0.923684      0.658109                                             Kuwait   base-lstm   6200.0
1765          population ages 0-14   0.983375    1.101155   1.049360      0.558744                                             Kuwait    base-gru   6733.0
343           population ages 0-14   1.188299    1.685799   1.298383      0.324466                                             Kuwait  simple-rnn   7701.0
3186         population ages 15-64   0.181643    0.049570   0.222644      0.977476                                             Kuwait          rf   1601.0
1053         population ages 15-64   0.279975    0.134610   0.366893      0.938836                                             Kuwait   base-lstm   2750.0
2475         population ages 15-64   0.321233    0.224473   0.473786      0.898004                                             Kuwait     xgboost   3426.0
1764         population ages 15-64   0.397445    0.209891   0.458139      0.904630                                             Kuwait    base-gru   3522.0
342          population ages 15-64   0.538971    0.307511   0.554537      0.860273                                             Kuwait  simple-rnn   4231.0
3188  population ages 65 and above   0.086534    0.010225   0.101120     -0.082961                                             Kuwait          rf   1719.0
2477  population ages 65 and above   0.174294    0.046046   0.214582     -3.876712                                             Kuwait     xgboost   3567.0
1055  population ages 65 and above   1.538544    2.408845   1.552045   -254.121833                                             Kuwait   base-lstm  10742.0
344   population ages 65 and above   1.559649    2.512471   1.585078   -265.096953                                             Kuwait  simple-rnn  10822.0
1766  population ages 65 and above   1.587222    2.571181   1.603490   -271.314953                                             Kuwait    base-gru  10880.0
2479          population ages 0-14   0.614997    0.613795   0.783451     -0.993812                                    Kyrgyz Republic     xgboost   6260.0
3190          population ages 0-14   0.890179    1.061449   1.030266     -2.447943                                    Kyrgyz Republic          rf   7676.0
1057          population ages 0-14   4.528588   21.449534   4.631364    -68.675290                                    Kyrgyz Republic   base-lstm  13496.0
1768          population ages 0-14   4.616276   22.266296   4.718718    -71.328408                                    Kyrgyz Republic    base-gru  13524.0
346           population ages 0-14   4.751090   23.666439   4.864816    -75.876541                                    Kyrgyz Republic  simple-rnn  13567.0
2478         population ages 15-64   0.278473    0.133334   0.365149      0.743146                                    Kyrgyz Republic     xgboost   3139.0
3189         population ages 15-64   0.673787    0.625230   0.790715     -0.204443                                    Kyrgyz Republic          rf   6095.0
1767         population ages 15-64   3.491119   13.254477   3.640670    -24.533424                                    Kyrgyz Republic    base-gru  12809.0
1056         population ages 15-64   3.622545   14.361288   3.789629    -26.665586                                    Kyrgyz Republic   base-lstm  12948.0
345          population ages 15-64   3.871818   16.165237   4.020602    -30.140712                                    Kyrgyz Republic  simple-rnn  13097.0
3191  population ages 65 and above   0.140172    0.030212   0.173817     -0.043210                                    Kyrgyz Republic          rf   2313.0
2480  population ages 65 and above   0.152906    0.031243   0.176758     -0.078804                                    Kyrgyz Republic     xgboost   2384.0
1058  population ages 65 and above   1.145249    1.318061   1.148068    -44.511522                                    Kyrgyz Republic   base-lstm   9348.0
347   population ages 65 and above   1.250110    1.567682   1.252071    -53.130728                                    Kyrgyz Republic  simple-rnn   9752.0
1769  population ages 65 and above   1.369507    1.881682   1.371744    -63.972850                                    Kyrgyz Republic    base-gru  10104.0
2482          population ages 0-14   0.204813    0.054086   0.232565      0.921991                                            Lao PDR     xgboost   1923.0
3193          population ages 0-14   0.282509    0.106066   0.325678      0.847022                                            Lao PDR          rf   2765.0
1060          population ages 0-14   1.333583    2.161167   1.470091     -2.117043                                            Lao PDR   base-lstm   8994.0
1771          population ages 0-14   1.344241    2.210219   1.486680     -2.187790                                            Lao PDR    base-gru   9035.0
349           population ages 0-14   1.497626    2.690938   1.640408     -2.881130                                            Lao PDR  simple-rnn   9498.0
2481         population ages 15-64   0.073464    0.007821   0.088435      0.986761                                            Lao PDR     xgboost    466.0
3192         population ages 15-64   0.182870    0.054311   0.233048      0.908062                                            Lao PDR          rf   1899.0
1770         population ages 15-64   0.367210    0.219472   0.468479      0.628480                                            Lao PDR    base-gru   3937.0
1059         population ages 15-64   0.411438    0.269678   0.519306      0.543493                                            Lao PDR   base-lstm   4279.0
348          population ages 15-64   0.466933    0.349683   0.591340      0.408062                                            Lao PDR  simple-rnn   4746.0
2483  population ages 65 and above   0.143338    0.026171   0.161774     -3.146200                                            Lao PDR     xgboost   3075.0
3194  population ages 65 and above   0.169433    0.039120   0.197787     -5.197676                                            Lao PDR          rf   3605.0
1061  population ages 65 and above   1.073391    1.267289   1.125739   -199.774735                                            Lao PDR   base-lstm   9491.0
350   population ages 65 and above   1.101020    1.341790   1.158357   -211.577881                                            Lao PDR  simple-rnn   9605.0
1772  population ages 65 and above   1.175617    1.513097   1.230080   -238.717747                                            Lao PDR    base-gru   9888.0
3196          population ages 0-14   0.211336    0.075834   0.275379      0.195895                          Late-demographic dividend          rf   2925.0
2485          population ages 0-14   0.424170    0.357975   0.598311     -2.795803                          Late-demographic dividend     xgboost   5711.0
1774          population ages 0-14   2.753927    8.646262   2.940453    -90.680870                          Late-demographic dividend    base-gru  12457.0
1063          population ages 0-14   2.939577   10.068933   3.173158   -105.766201                          Late-demographic dividend   base-lstm  12704.0
352           population ages 0-14   3.001074   10.349245   3.217024   -108.738503                          Late-demographic dividend  simple-rnn  12751.0
2484         population ages 15-64   0.336400    0.195527   0.442184      0.802808                          Late-demographic dividend     xgboost   3543.0
1773         population ages 15-64   1.268260    2.111279   1.453024     -1.129259                          Late-demographic dividend    base-gru   8663.0
3195         population ages 15-64   1.248561    2.558666   1.599583     -1.580457                          Late-demographic dividend          rf   9003.0
351          population ages 15-64   1.439988    2.769130   1.664070     -1.792713                          Late-demographic dividend  simple-rnn   9275.0
1062         population ages 15-64   2.077860    5.481061   2.341167     -4.527741                          Late-demographic dividend   base-lstm  10842.0
2486  population ages 65 and above   0.236058    0.102725   0.320508      0.937081                          Late-demographic dividend     xgboost   2405.0
3197  population ages 65 and above   0.643288    0.817833   0.904341      0.499080                          Late-demographic dividend          rf   5978.0
1064  population ages 65 and above   1.565355    2.485678   1.576603     -0.522469                          Late-demographic dividend   base-lstm   8874.0
1775  population ages 65 and above   1.797909    3.287465   1.813137     -1.013562                          Late-demographic dividend    base-gru   9504.0
353   population ages 65 and above   1.814832    3.356655   1.832118     -1.055940                          Late-demographic dividend  simple-rnn   9553.0
3199          population ages 0-14   0.138254    0.022638   0.150459      0.988765                          Latin America & Caribbean          rf   1026.0
2488          population ages 0-14   0.201379    0.052492   0.229111      0.973949                          Latin America & Caribbean     xgboost   1715.0
1777          population ages 0-14   2.302442    5.858142   2.420360     -1.907296                          Latin America & Caribbean    base-gru  10575.0
1066          population ages 0-14   2.530507    7.170762   2.677828     -2.558727                          Latin America & Caribbean   base-lstm  10987.0
355           population ages 0-14   2.641813    7.705313   2.775845     -2.824015                          Latin America & Caribbean  simple-rnn  11152.0
3198         population ages 15-64   0.093347    0.013329   0.115452      0.973718                          Latin America & Caribbean          rf    735.0
2487         population ages 15-64   0.129055    0.023955   0.154774      0.952766                          Latin America & Caribbean     xgboost   1152.0
1776         population ages 15-64   1.111635    1.338456   1.156917     -1.639133                          Latin America & Caribbean    base-gru   8055.0
354          population ages 15-64   1.624392    2.868523   1.693671     -4.656080                          Latin America & Caribbean  simple-rnn   9883.0
1065         population ages 15-64   1.836479    3.776656   1.943362     -6.446714                          Latin America & Caribbean   base-lstm  10485.0
3200  population ages 65 and above   0.177996    0.052849   0.229890      0.897638                          Latin America & Caribbean          rf   1897.0
2489  population ages 65 and above   0.204703    0.058643   0.242164      0.886416                          Latin America & Caribbean     xgboost   2083.0
1067  population ages 65 and above   0.875263    0.871430   0.933504     -0.687835                          Latin America & Caribbean   base-lstm   6963.0
356   population ages 65 and above   0.994152    1.134890   1.065312     -1.198120                          Latin America & Caribbean  simple-rnn   7591.0
1778  population ages 65 and above   1.143736    1.474651   1.214352     -1.856188                          Latin America & Caribbean    base-gru   8288.0
3202          population ages 0-14   0.086323    0.013773   0.117357      0.993664  Latin America & Caribbean (excluding high income)          rf    622.0
2491          population ages 0-14   0.263690    0.082215   0.286731      0.962178  Latin America & Caribbean (excluding high income)     xgboost   2219.0
1780          population ages 0-14   2.308137    5.874835   2.423806     -1.702647  Latin America & Caribbean (excluding high income)    base-gru  10549.0
1069          population ages 0-14   2.521207    7.101235   2.664814     -2.266838  Latin America & Caribbean (excluding high income)   base-lstm  10928.0
358           population ages 0-14   2.641384    7.686365   2.772429     -2.536019  Latin America & Caribbean (excluding high income)  simple-rnn  11103.0
2490         population ages 15-64   0.127756    0.025406   0.159393      0.958981  Latin America & Caribbean (excluding high income)     xgboost   1149.0
3201         population ages 15-64   0.144160    0.033536   0.183127      0.945857  Latin America & Caribbean (excluding high income)          rf   1417.0
1779         population ages 15-64   1.127083    1.363865   1.167846     -1.201968  Latin America & Caribbean (excluding high income)    base-gru   7990.0
357          population ages 15-64   1.626202    2.853858   1.689336     -3.607571  Latin America & Caribbean (excluding high income)  simple-rnn   9771.0
1068         population ages 15-64   1.837498    3.754690   1.937702     -5.061969  Latin America & Caribbean (excluding high income)   base-lstm  10366.0
2492  population ages 65 and above   0.144511    0.031544   0.177607      0.934804  Latin America & Caribbean (excluding high income)     xgboost   1405.0
3203  population ages 65 and above   0.173814    0.049562   0.222626      0.897563  Latin America & Caribbean (excluding high income)          rf   1831.0
1070  population ages 65 and above   0.883678    0.896538   0.946857     -0.852979  Latin America & Caribbean (excluding high income)   base-lstm   7050.0
359   population ages 65 and above   1.011493    1.185586   1.088846     -1.450390  Latin America & Caribbean (excluding high income)  simple-rnn   7727.0
1781  population ages 65 and above   1.156282    1.518545   1.232293     -2.138555  Latin America & Caribbean (excluding high income)    base-gru   8416.0
3205          population ages 0-14   0.188051    0.053168   0.230581      0.974335  Latin America & the Caribbean (IDA & IBRD coun...          rf   1690.0
2494          population ages 0-14   0.290712    0.107750   0.328254      0.947987  Latin America & the Caribbean (IDA & IBRD coun...     xgboost   2550.0
1783          population ages 0-14   2.318650    5.956993   2.440695     -1.875548  Latin America & the Caribbean (IDA & IBRD coun...    base-gru  10601.0
1072          population ages 0-14   2.537462    7.226100   2.688141     -2.488169  Latin America & the Caribbean (IDA & IBRD coun...   base-lstm  10997.0
361           population ages 0-14   2.659832    7.827243   2.797721     -2.778351  Latin America & the Caribbean (IDA & IBRD coun...  simple-rnn  11155.0
2493         population ages 15-64   0.102447    0.014338   0.119740      0.973834  Latin America & the Caribbean (IDA & IBRD coun...     xgboost    794.0
3204         population ages 15-64   0.268441    0.116602   0.341470      0.787205  Latin America & the Caribbean (IDA & IBRD coun...          rf   2930.0
1782         population ages 15-64   1.163816    1.475136   1.214552     -1.692089  Latin America & the Caribbean (IDA & IBRD coun...    base-gru   8285.0
360          population ages 15-64   1.676077    3.065391   1.750826     -4.594270  Latin America & the Caribbean (IDA & IBRD coun...  simple-rnn   9986.0
1071         population ages 15-64   1.874010    3.947215   1.986760     -6.203579  Latin America & the Caribbean (IDA & IBRD coun...   base-lstm  10553.0
2495  population ages 65 and above   0.148302    0.029951   0.173064      0.940679  Latin America & the Caribbean (IDA & IBRD coun...     xgboost   1381.0
3206  population ages 65 and above   0.153241    0.041844   0.204557      0.917125  Latin America & the Caribbean (IDA & IBRD coun...          rf   1624.0
1073  population ages 65 and above   0.865639    0.849896   0.921898     -0.683294  Latin America & the Caribbean (IDA & IBRD coun...   base-lstm   6919.0
362   population ages 65 and above   0.986865    1.116334   1.056567     -1.210997  Latin America & the Caribbean (IDA & IBRD coun...  simple-rnn   7561.0
1784  population ages 65 and above   1.133018    1.444181   1.201741     -1.860326  Latin America & the Caribbean (IDA & IBRD coun...    base-gru   8240.0
3208          population ages 0-14   0.046963    0.003136   0.056001      0.884976                                             Latvia          rf    579.0
2497          population ages 0-14   0.221406    0.068411   0.261555     -1.509144                                             Latvia     xgboost   3528.0
1786          population ages 0-14   3.099384    9.818618   3.133467   -359.122350                                             Latvia    base-gru  12911.0
1075          population ages 0-14   3.446301   12.203579   3.493362   -446.596764                                             Latvia   base-lstm  13277.0
364           population ages 0-14   3.819070   14.932269   3.864229   -546.678302                                             Latvia  simple-rnn  13549.0
2496         population ages 15-64   0.146819    0.039421   0.198548      0.903783                                             Latvia     xgboost   1634.0
3207         population ages 15-64   0.414996    0.251705   0.501702      0.385659                                             Latvia          rf   4332.0
1785         population ages 15-64   2.594578    7.113497   2.667114    -16.362050                                             Latvia    base-gru  11763.0
363          population ages 15-64   3.297652   11.429215   3.380712    -26.895509                                             Latvia  simple-rnn  12635.0
1074         population ages 15-64   3.790443   15.007251   3.873919    -35.628491                                             Latvia   base-lstm  13072.0
3209  population ages 65 and above   0.146864    0.029469   0.171665      0.875577                                             Latvia          rf   1540.0
2498  population ages 65 and above   0.432702    0.333637   0.577613     -0.408682                                             Latvia     xgboost   5045.0
1787  population ages 65 and above   0.573431    0.372112   0.610010     -0.571129                                             Latvia    base-gru   5477.0
365   population ages 65 and above   0.744708    0.600812   0.775121     -1.536745                                             Latvia  simple-rnn   6610.0
1076  population ages 65 and above   0.762056    0.633419   0.795877     -1.674421                                             Latvia   base-lstm   6724.0
3211          population ages 0-14   0.071462    0.006897   0.083050      0.990041       Least developed countries: UN classification          rf    412.0
2500          population ages 0-14   0.202172    0.052936   0.230078      0.923569       Least developed countries: UN classification     xgboost   1890.0
1789          population ages 0-14   0.806410    0.721432   0.849372     -0.041631       Least developed countries: UN classification    base-gru   6413.0
1078          population ages 0-14   0.903061    0.908620   0.953216     -0.311900       Least developed countries: UN classification   base-lstm   6904.0
367           population ages 0-14   1.137994    1.435791   1.198245     -1.073049       Least developed countries: UN classification  simple-rnn   8018.0
1788         population ages 15-64   0.092321    0.009653   0.098248      0.981349       Least developed countries: UN classification    base-gru    604.0
3210         population ages 15-64   0.141790    0.039503   0.198754      0.923673       Least developed countries: UN classification          rf   1554.0
1077         population ages 15-64   0.266658    0.085315   0.292088      0.835155       Least developed countries: UN classification   base-lstm   2577.0
366          population ages 15-64   0.454941    0.236267   0.486073      0.543487       Least developed countries: UN classification  simple-rnn   4237.0
2499         population ages 15-64   0.492076    0.286796   0.535533      0.445856       Least developed countries: UN classification     xgboost   4575.0
3212  population ages 65 and above   0.076595    0.011359   0.106578      0.132609       Least developed countries: UN classification          rf   1599.0
2501  population ages 65 and above   0.104672    0.021392   0.146259     -0.633498       Least developed countries: UN classification     xgboost   2269.0
368   population ages 65 and above   0.536081    0.320459   0.566091    -23.470861       Least developed countries: UN classification  simple-rnn   6681.0
1079  population ages 65 and above   0.541289    0.320566   0.566186    -23.479014       Least developed countries: UN classification   base-lstm   6693.0
1790  population ages 65 and above   0.654631    0.472001   0.687023    -35.042838       Least developed countries: UN classification    base-gru   7411.0
2503          population ages 0-14   1.549990    3.692239   1.921520     -2.250649                                            Lebanon     xgboost   9790.0
3214          population ages 0-14   1.682281    3.709789   1.926081     -2.266100                                            Lebanon          rf   9894.0
1792          population ages 0-14   5.554962   35.080004   5.922838    -29.884450                                            Lebanon    base-gru  13461.0
1081          population ages 0-14   5.610938   35.786877   5.982213    -30.506781                                            Lebanon   base-lstm  13476.0
370           population ages 0-14   5.746732   37.252627   6.103493    -31.797228                                            Lebanon  simple-rnn  13496.0
2502         population ages 15-64   1.154821    1.961619   1.400578      0.475189                                            Lebanon     xgboost   7759.0
3213         population ages 15-64   2.941710   11.633138   3.410739     -2.112324                                            Lebanon          rf  11497.0
1791         population ages 15-64   6.102965   42.897593   6.549625    -10.476802                                            Lebanon    base-gru  13133.0
369          population ages 15-64   6.575559   48.994927   6.999638    -12.108079                                            Lebanon  simple-rnn  13198.0
1080         population ages 15-64   6.763861   52.199533   7.224924    -12.965438                                            Lebanon   base-lstm  13227.0
2504  population ages 65 and above   0.203516    0.052317   0.228729      0.932245                                            Lebanon     xgboost   1859.0
1793  population ages 65 and above   0.368970    0.158262   0.397822      0.795038                                            Lebanon    base-gru   3437.0
371   population ages 65 and above   0.437673    0.232433   0.482113      0.698981                                            Lebanon  simple-rnn   4057.0
1082  population ages 65 and above   0.550787    0.369022   0.607471      0.522087                                            Lebanon   base-lstm   4862.0
3215  population ages 65 and above   0.889051    1.024712   1.012281     -0.327085                                            Lebanon          rf   7045.0
2506          population ages 0-14   0.220694    0.093994   0.306585      0.802050                                            Lesotho     xgboost   2600.0
3217          population ages 0-14   0.693006    0.791200   0.889494     -0.666249                                            Lesotho          rf   6581.0
1084          population ages 0-14   3.141618   11.213674   3.348682    -22.615751                                            Lesotho   base-lstm  12510.0
1795          population ages 0-14   3.242707   11.989497   3.462585    -24.249617                                            Lesotho    base-gru  12636.0
373           population ages 0-14   3.276583   12.101407   3.478708    -24.485297                                            Lesotho  simple-rnn  12664.0
2505         population ages 15-64   0.282312    0.095907   0.309688      0.803681                                            Lesotho     xgboost   2785.0
3216         population ages 15-64   0.371331    0.254985   0.504960      0.478053                                            Lesotho          rf   4190.0
1794         population ages 15-64   1.292206    2.075102   1.440522     -3.247677                                            Lesotho    base-gru   9101.0
372          population ages 15-64   1.408398    2.441152   1.562419     -3.996970                                            Lesotho  simple-rnn   9489.0
1083         population ages 15-64   1.436942    2.597241   1.611596     -4.316481                                            Lesotho   base-lstm   9612.0
3218  population ages 65 and above   0.039722    0.001908   0.043680     -0.500224                                            Lesotho          rf   1523.0
2507  population ages 65 and above   0.072363    0.005863   0.076570     -3.610049                                            Lesotho     xgboost   2451.0
1085  population ages 65 and above   1.378807    2.137139   1.461896  -1679.431046                                            Lesotho   base-lstm  10675.0
374   population ages 65 and above   1.462077    2.431060   1.559186  -1910.541453                                            Lesotho  simple-rnn  10902.0
1796  population ages 65 and above   1.641549    3.036180   1.742464  -2386.346970                                            Lesotho    base-gru  11281.0
2509          population ages 0-14   0.616041    0.593130   0.770149     -0.454888                                            Liberia     xgboost   6056.0
3220          population ages 0-14   0.765950    0.930555   0.964653     -1.282558                                            Liberia          rf   7080.0
1798          population ages 0-14   1.225218    1.851771   1.360798     -3.542209                                            Liberia    base-gru   8942.0
1087          population ages 0-14   1.453891    2.552107   1.597532     -5.260061                                            Liberia   base-lstm   9699.0
376           population ages 0-14   1.728088    3.511547   1.873912     -7.613471                                            Liberia  simple-rnn  10388.0
3219         population ages 15-64   0.207931    0.082907   0.287935      0.831109                                            Liberia          rf   2398.0
2508         population ages 15-64   0.350408    0.174489   0.417718      0.644545                                            Liberia     xgboost   3666.0
1797         population ages 15-64   0.710053    0.577006   0.759609     -0.175433                                            Liberia    base-gru   6018.0
1086         population ages 15-64   0.993665    1.102448   1.049975     -1.245825                                            Liberia   base-lstm   7574.0
375          population ages 15-64   1.245852    1.702684   1.304869     -2.468583                                            Liberia  simple-rnn   8702.0
3221  population ages 65 and above   0.256664    0.103818   0.322208    -22.061610                                            Liberia          rf   5049.0
2510  population ages 65 and above   0.254552    0.113881   0.337463    -24.296976                                            Liberia     xgboost   5154.0
377   population ages 65 and above   0.376287    0.232366   0.482043    -50.616443                                            Liberia  simple-rnn   6274.0
1088  population ages 65 and above   0.416077    0.269682   0.519309    -58.905666                                            Liberia   base-lstm   6532.0
1799  population ages 65 and above   0.494146    0.355787   0.596479    -78.032589                                            Liberia    base-gru   7014.0
2512          population ages 0-14   1.727785    4.341289   2.083576     -2.961279                                              Libya     xgboost  10224.0
3223          population ages 0-14   1.870052    4.972198   2.229843     -3.536961                                              Libya          rf  10546.0
1090          population ages 0-14   5.251515   30.435033   5.516796    -26.770932                                              Libya   base-lstm  13408.0
1801          population ages 0-14   5.306380   31.069230   5.573978    -27.349616                                              Libya    base-gru  13420.0
379           population ages 0-14   5.618501   34.764123   5.896111    -30.721080                                              Libya  simple-rnn  13472.0
2511         population ages 15-64   1.777615    4.749411   2.179314     -3.012347                                              Libya     xgboost  10356.0
3222         population ages 15-64   2.217459    6.625232   2.573953     -4.597058                                              Libya          rf  11072.0
1800         population ages 15-64   4.428934   21.817832   4.670956    -17.431908                                              Libya    base-gru  13104.0
1089         population ages 15-64   4.745320   24.875024   4.987487    -20.014652                                              Libya   base-lstm  13217.0
378          population ages 15-64   4.995084   27.336920   5.228472    -22.094485                                              Libya  simple-rnn  13301.0
2513  population ages 65 and above   0.200291    0.055413   0.235399      0.379041                                              Libya     xgboost   2608.0
3224  population ages 65 and above   0.269468    0.099083   0.314775     -0.110335                                              Libya          rf   3463.0
1091  population ages 65 and above   1.263821    1.915914   1.384166    -20.469913                                              Libya   base-lstm   9731.0
380   population ages 65 and above   1.363946    2.267303   1.505756    -24.407602                                              Libya  simple-rnn  10068.0
1802  population ages 65 and above   1.458634    2.501522   1.581620    -27.032288                                              Libya    base-gru  10301.0
3226          population ages 0-14   0.111239    0.016623   0.128930      0.479269                                          Lithuania          rf   1643.0
2515          population ages 0-14   0.236171    0.102424   0.320038     -2.208532                                          Lithuania     xgboost   4034.0
1804          population ages 0-14   2.801934    8.192502   2.862255   -255.638032                                          Lithuania    base-gru  12590.0
1093          population ages 0-14   3.128954   10.247771   3.201214   -320.021313                                          Lithuania   base-lstm  12930.0
382           population ages 0-14   3.532436   12.998766   3.605380   -406.198865                                          Lithuania  simple-rnn  13353.0
3225         population ages 15-64   0.065271    0.005588   0.074756      0.981907                                          Lithuania          rf    390.0
2514         population ages 15-64   0.260997    0.097237   0.311829      0.685192                                          Lithuania     xgboost   2890.0
1803         population ages 15-64   1.706808    3.253979   1.803879     -9.534839                                          Lithuania    base-gru  10381.0
381          population ages 15-64   2.316925    5.867225   2.422236    -17.995286                                          Lithuania  simple-rnn  11519.0
1092         population ages 15-64   2.889606    8.924718   2.987427    -27.893995                                          Lithuania   base-lstm  12250.0
1805  population ages 65 and above   0.115772    0.014689   0.121196      0.896893                                          Lithuania    base-gru   1088.0
3227  population ages 65 and above   0.144822    0.030829   0.175583      0.783591                                          Lithuania          rf   1733.0
2516  population ages 65 and above   0.166656    0.059479   0.243882      0.582487                                          Lithuania     xgboost   2394.0
383   population ages 65 and above   0.248192    0.062865   0.250728      0.558718                                          Lithuania  simple-rnn   2682.0
1094  population ages 65 and above   0.320521    0.106315   0.326059      0.253717                                          Lithuania   base-lstm   3450.0
2518          population ages 0-14   0.436848    0.239100   0.488979      0.445159                                Low & middle income     xgboost   4295.0
3229          population ages 0-14   0.505869    0.337882   0.581276      0.215933                                Low & middle income          rf   4894.0
1807          population ages 0-14   2.826271    9.093868   3.015604    -20.102644                                Low & middle income    base-gru  12137.0
1096          population ages 0-14   2.928543    9.855379   3.139328    -21.869757                                Low & middle income   base-lstm  12290.0
385           population ages 0-14   3.056909   10.624174   3.259475    -23.653773                                Low & middle income  simple-rnn  12426.0
2517         population ages 15-64   0.559531    0.592586   0.769796    -59.202252                                Low & middle income     xgboost   7637.0
3228         population ages 15-64   0.898547    1.281274   1.131934   -129.167675                                Low & middle income          rf   9263.0
1806         population ages 15-64   1.313232    2.109406   1.452379   -213.299666                                Low & middle income    base-gru  10397.0
384          population ages 15-64   1.659969    3.316739   1.821192   -335.955504                                Low & middle income  simple-rnn  11235.0
1095         population ages 15-64   1.822133    4.089628   2.022283   -414.475174                                Low & middle income   base-lstm  11634.0
2519  population ages 65 and above   0.248002    0.125993   0.354954      0.665898                                Low & middle income     xgboost   3078.0
3230  population ages 65 and above   0.311443    0.183504   0.428374      0.513390                                Low & middle income          rf   3733.0
1097  population ages 65 and above   1.120256    1.330501   1.153473     -2.528168                                Low & middle income   base-lstm   8243.0
386   population ages 65 and above   1.288197    1.780595   1.334389     -3.721709                                Low & middle income  simple-rnn   8996.0
1808  population ages 65 and above   1.399835    2.095005   1.447413     -4.555449                                Low & middle income    base-gru   9346.0
3232          population ages 0-14   0.120068    0.019445   0.139446      0.948453                                         Low income          rf   1048.0
2521          population ages 0-14   0.158014    0.044877   0.211841      0.881039                                         Low income     xgboost   1763.0
1810          population ages 0-14   0.240534    0.068602   0.261920      0.818146                                         Low income    base-gru   2396.0
1099          population ages 0-14   0.343395    0.135179   0.367667      0.641660                                         Low income   base-lstm   3425.0
388           population ages 0-14   0.554805    0.343845   0.586383      0.088519                                         Low income  simple-rnn   5075.0
1098         population ages 15-64   0.091904    0.011689   0.108114      0.964812                                         Low income   base-lstm    718.0
3231         population ages 15-64   0.095259    0.015943   0.126267      0.952003                                         Low income          rf    882.0
387          population ages 15-64   0.127171    0.021490   0.146596      0.935305                                         Low income  simple-rnn   1157.0
2520         population ages 15-64   0.172278    0.037296   0.193123      0.887722                                         Low income     xgboost   1722.0
1809         population ages 15-64   0.193130    0.054764   0.234016      0.835138                                         Low income    base-gru   2092.0
2522  population ages 65 and above   0.046776    0.003888   0.062356     -1.597936                                         Low income     xgboost   1954.0
3233  population ages 65 and above   0.051545    0.004774   0.069097     -2.190040                                         Low income          rf   2127.0
389   population ages 65 and above   0.329059    0.126082   0.355080    -83.242347                                         Low income  simple-rnn   5735.0
1100  population ages 65 and above   0.361305    0.147083   0.383513    -97.274175                                         Low income   base-lstm   5982.0
1811  population ages 65 and above   0.421914    0.202002   0.449447   -133.969280                                         Low income    base-gru   6441.0
2524          population ages 0-14   0.098628    0.015917   0.126162      0.989136                                Lower middle income     xgboost    739.0
3235          population ages 0-14   0.256482    0.081414   0.285332      0.944432                                Lower middle income          rf   2255.0
1813          population ages 0-14   2.190105    5.485514   2.342117     -2.744040                                Lower middle income    base-gru  10651.0
1102          population ages 0-14   2.281380    5.981848   2.445782     -3.082804                                Lower middle income   base-lstm  10804.0
391           population ages 0-14   2.446377    6.802308   2.608123     -3.642794                                Lower middle income  simple-rnn  11103.0
2523         population ages 15-64   0.096936    0.012788   0.113084      0.980740                                Lower middle income     xgboost    714.0
3234         population ages 15-64   0.691004    0.756817   0.869952     -0.139830                                Lower middle income          rf   6339.0
1812         population ages 15-64   0.779371    0.767761   0.876220     -0.156312                                Lower middle income    base-gru   6484.0
390          population ages 15-64   1.115115    1.517225   1.231757     -1.285067                                Lower middle income  simple-rnn   8152.0
1101         population ages 15-64   1.145574    1.675039   1.294233     -1.522749                                Lower middle income   base-lstm   8361.0
2525  population ages 65 and above   0.107525    0.022284   0.149277      0.862702                                Lower middle income     xgboost   1283.0
3236  population ages 65 and above   0.113507    0.026947   0.164154      0.833971                                Lower middle income          rf   1450.0
1103  population ages 65 and above   1.053168    1.204989   1.097720     -6.424420                                Lower middle income   base-lstm   8459.0
392   population ages 65 and above   1.174672    1.519445   1.232658     -8.361911                                Lower middle income  simple-rnn   9042.0
1814  population ages 65 and above   1.305459    1.861058   1.364206    -10.466727                                Lower middle income    base-gru   9474.0
3238          population ages 0-14   0.084579    0.010983   0.104798     -0.211282                                         Luxembourg          rf   1786.0
2527          population ages 0-14   0.317213    0.147829   0.384485    -15.304071                                         Luxembourg     xgboost   5377.0
1816          population ages 0-14   2.008016    4.107983   2.026816   -452.071156                                         Luxembourg    base-gru  11755.0
1105          population ages 0-14   2.276724    5.313249   2.305049   -585.000447                                         Luxembourg   base-lstm  12145.0
394           population ages 0-14   2.520205    6.482076   2.545992   -713.910866                                         Luxembourg  simple-rnn  12425.0
2526         population ages 15-64   0.123507    0.019336   0.139053     -2.237042                                         Luxembourg     xgboost   2734.0
3237         population ages 15-64   0.184879    0.050579   0.224897     -7.467439                                         Luxembourg          rf   3940.0
1104         population ages 15-64   0.291154    0.118761   0.344618    -18.882056                                         Luxembourg   base-lstm   5166.0
393          population ages 15-64   0.332797    0.162269   0.402826    -26.165710                                         Luxembourg  simple-rnn   5703.0
1815         population ages 15-64   0.801740    0.655795   0.809811   -108.787702                                         Luxembourg    base-gru   8289.0
3239  population ages 65 and above   0.184909    0.045471   0.213239     -0.677865                                         Luxembourg          rf   2900.0
2528  population ages 65 and above   0.264357    0.124307   0.352571     -3.586886                                         Luxembourg     xgboost   4498.0
1106  population ages 65 and above   1.760502    3.133762   1.770243   -114.635185                                         Luxembourg   base-lstm  11093.0
395   population ages 65 and above   1.834911    3.407681   1.845991   -124.742771                                         Luxembourg  simple-rnn  11248.0
1817  population ages 65 and above   2.049883    4.246814   2.060780   -155.706619                                         Luxembourg    base-gru  11670.0
1819          population ages 0-14   0.262718    0.139071   0.372922      0.918356                                         Madagascar    base-gru   2809.0
1108          population ages 0-14   0.277691    0.155768   0.394675      0.908553                                         Madagascar   base-lstm   2957.0
397           population ages 0-14   0.542020    0.421257   0.649043      0.752693                                         Madagascar  simple-rnn   4753.0
2530          population ages 0-14   0.781405    0.718702   0.847763      0.578072                                         Madagascar     xgboost   5989.0
3241          population ages 0-14   0.772435    0.808163   0.898979      0.525552                                         Madagascar          rf   6136.0
1107         population ages 15-64   0.265022    0.086624   0.294319      0.929413                                         Madagascar   base-lstm   2374.0
396          population ages 15-64   0.254694    0.104536   0.323320      0.914817                                         Madagascar  simple-rnn   2539.0
1818         population ages 15-64   0.292997    0.115593   0.339990      0.905807                                         Madagascar    base-gru   2735.0
3240         population ages 15-64   0.298792    0.126853   0.356164      0.896631                                         Madagascar          rf   2877.0
2529         population ages 15-64   0.793653    0.800022   0.894439      0.348085                                         Madagascar     xgboost   6278.0
2531  population ages 65 and above   0.186010    0.070631   0.265765     -0.595280                                         Madagascar     xgboost   3157.0
3242  population ages 65 and above   0.205732    0.086165   0.293539     -0.946127                                         Madagascar          rf   3461.0
398   population ages 65 and above   0.362494    0.133783   0.365764     -2.021631                                         Madagascar  simple-rnn   4523.0
1109  population ages 65 and above   0.366330    0.135475   0.368069     -2.059835                                         Madagascar   base-lstm   4561.0
1820  population ages 65 and above   0.486593    0.241913   0.491847     -4.463860                                         Madagascar    base-gru   5671.0
3244          population ages 0-14   0.641249    0.844542   0.918989      0.337354                                             Malawi          rf   6119.0
2533          population ages 0-14   0.860617    1.136997   1.066301      0.107887                                             Malawi     xgboost   6911.0
1822          population ages 0-14   1.971577    4.368613   2.090123     -2.427710                                             Malawi    base-gru  10314.0
1111          population ages 0-14   2.335002    6.103806   2.470588     -3.789180                                             Malawi   base-lstm  10960.0
400           population ages 0-14   2.417362    6.459300   2.541515     -4.068108                                             Malawi  simple-rnn  11072.0
2532         population ages 15-64   0.342380    0.261946   0.511807      0.883053                                             Malawi     xgboost   3662.0
1821         population ages 15-64   0.566606    0.368644   0.607161      0.835417                                             Malawi    base-gru   4507.0
399          population ages 15-64   0.849759    0.768562   0.876677      0.656872                                             Malawi  simple-rnn   6065.0
3243         population ages 15-64   0.736269    0.928469   0.963571      0.585480                                             Malawi          rf   6182.0
1110         population ages 15-64   0.960023    0.961674   0.980650      0.570656                                             Malawi   base-lstm   6536.0
2534  population ages 65 and above   0.506985    0.399422   0.631998     -1.534925                                             Malawi     xgboost   5721.0
3245  population ages 65 and above   0.611770    0.511794   0.715398     -2.248095                                             Malawi          rf   6365.0
401   population ages 65 and above   1.185601    1.848553   1.359615    -10.731820                                             Malawi  simple-rnn   9366.0
1112  population ages 65 and above   1.210153    1.884196   1.372660    -10.958030                                             Malawi   base-lstm   9426.0
1823  population ages 65 and above   1.282097    2.108891   1.452202    -12.384052                                             Malawi    base-gru   9660.0
3247          population ages 0-14   0.091904    0.012352   0.111141      0.995811                                           Malaysia          rf    596.0
2536          population ages 0-14   0.291161    0.124623   0.353020      0.957740                                           Malaysia     xgboost   2649.0
1825          population ages 0-14   1.618110    3.346183   1.829257     -0.134700                                           Malaysia    base-gru   9085.0
1114          population ages 0-14   1.618525    3.430448   1.852147     -0.163275                                           Malaysia   base-lstm   9127.0
403           population ages 0-14   1.848430    4.245274   2.060406     -0.439585                                           Malaysia  simple-rnn   9655.0
2535         population ages 15-64   0.405250    0.254875   0.504851      0.762859                                           Malaysia     xgboost   3980.0
3246         population ages 15-64   0.630639    0.521823   0.722373      0.514483                                           Malaysia          rf   5400.0
1824         population ages 15-64   1.029569    1.407312   1.186302     -0.309397                                           Malaysia    base-gru   7608.0
402          population ages 15-64   1.341744    2.244786   1.498261     -1.088603                                           Malaysia  simple-rnn   8779.0
1113         population ages 15-64   1.487425    2.926445   1.710685     -1.722834                                           Malaysia   base-lstm   9359.0
3248  population ages 65 and above   0.107437    0.025902   0.160942      0.949319                                           Malaysia          rf   1112.0
2537  population ages 65 and above   0.135554    0.025381   0.159314      0.950339                                           Malaysia     xgboost   1206.0
1115  population ages 65 and above   0.835116    0.754602   0.868679     -0.476478                                           Malaysia   base-lstm   6681.0
404   population ages 65 and above   1.039895    1.197184   1.094159     -1.342447                                           Malaysia  simple-rnn   7750.0
1826  population ages 65 and above   1.091543    1.289227   1.135441     -1.522540                                           Malaysia    base-gru   7960.0
2539          population ages 0-14   0.239634    0.088838   0.298057      0.916473                                           Maldives     xgboost   2362.0
3250          population ages 0-14   0.228828    0.092612   0.304322      0.912925                                           Maldives          rf   2371.0
1828          population ages 0-14   1.186924    2.387729   1.545228     -1.244979                                           Maldives    base-gru   8764.0
1117          population ages 0-14   1.206954    2.522250   1.588159     -1.371457                                           Maldives   base-lstm   8885.0
406           population ages 0-14   1.232510    2.638588   1.624373     -1.480840                                           Maldives  simple-rnn   8978.0
1116         population ages 15-64   0.991854    1.286669   1.134314     -0.219255                                           Maldives   base-lstm   7405.0
2538         population ages 15-64   0.818428    1.503083   1.226003     -0.424331                                           Maldives     xgboost   7504.0
1827         population ages 15-64   1.112951    1.495013   1.222707     -0.416683                                           Maldives    base-gru   7839.0
405          population ages 15-64   1.149593    1.698016   1.303080     -0.609051                                           Maldives  simple-rnn   8111.0
3249         population ages 15-64   1.225418    4.188874   2.046674     -2.969404                                           Maldives          rf   9795.0
3251  population ages 65 and above   0.142078    0.038748   0.196845     -2.104139                                           Maldives          rf   3113.0
2540  population ages 65 and above   0.208694    0.074218   0.272430     -4.945696                                           Maldives     xgboost   4064.0
1118  population ages 65 and above   2.668442    7.827480   2.797763   -626.069395                                           Maldives   base-lstm  12661.0
1829  population ages 65 and above   2.850097    8.937509   2.989567   -714.995191                                           Maldives    base-gru  12819.0
407   population ages 65 and above   2.873110    9.113100   3.018791   -729.062025                                           Maldives  simple-rnn  12856.0
3253          population ages 0-14   0.347045    0.136641   0.369649     -1.052392                                               Mali          rf   4272.0
1831          population ages 0-14   0.714679    0.534574   0.731145     -7.029491                                               Mali    base-gru   7080.0
2542          population ages 0-14   0.731803    0.661385   0.813256     -8.934238                                               Mali     xgboost   7472.0
1120          population ages 0-14   0.891692    0.829590   0.910818    -11.460749                                               Mali   base-lstm   8047.0
409           population ages 0-14   1.171898    1.436986   1.198743    -20.584050                                               Mali  simple-rnn   9273.0
2541         population ages 15-64   0.251833    0.075934   0.275562      0.113954                                               Mali     xgboost   3089.0
3252         population ages 15-64   0.272857    0.095398   0.308866     -0.113162                                               Mali          rf   3435.0
1830         population ages 15-64   0.412711    0.210659   0.458976     -1.458092                                               Mali    base-gru   4915.0
1119         population ages 15-64   0.603006    0.398224   0.631050     -3.646711                                               Mali   base-lstm   6298.0
408          population ages 15-64   0.804771    0.686433   0.828512     -7.009708                                               Mali  simple-rnn   7532.0
3254  population ages 65 and above   0.048482    0.003067   0.055376      0.646305                                               Mali          rf    940.0
2543  population ages 65 and above   0.133956    0.026268   0.162073     -2.029703                                               Mali     xgboost   2849.0
410   population ages 65 and above   0.356063    0.145972   0.382063    -15.836468                                               Mali  simple-rnn   5463.0
1832  population ages 65 and above   0.406645    0.185748   0.430985    -20.424184                                               Mali    base-gru   5879.0
1121  population ages 65 and above   0.413040    0.191813   0.437965    -21.123720                                               Mali   base-lstm   5940.0
2545          population ages 0-14   0.320354    0.123724   0.351744      0.307337                                              Malta     xgboost   3538.0
3256          population ages 0-14   0.641612    0.555326   0.745202     -2.108962                                              Malta          rf   6475.0
1834          population ages 0-14   0.828605    0.842891   0.918091     -3.718883                                              Malta    base-gru   7528.0
1123          population ages 0-14   1.386494    2.284793   1.511553    -11.791298                                              Malta   base-lstm   9819.0
412           population ages 0-14   1.621339    3.035283   1.742206    -15.992879                                              Malta  simple-rnn  10413.0
1833         population ages 15-64   0.442769    0.255234   0.505207     -3.173493                                              Malta    base-gru   5500.0
3255         population ages 15-64   0.638936    0.830935   0.911556    -12.587133                                              Malta          rf   7715.0
2544         population ages 15-64   0.844220    1.361954   1.167028    -21.270150                                              Malta     xgboost   8834.0
411          population ages 15-64   1.234551    1.571574   1.253624    -24.697780                                              Malta  simple-rnn   9537.0
1122         population ages 15-64   1.854594    3.533535   1.879770    -56.779018                                              Malta   base-lstm  11122.0
3257  population ages 65 and above   0.366823    0.186802   0.432206      0.529000                                              Malta          rf   3867.0
2546  population ages 65 and above   0.487306    0.309989   0.556766      0.218400                                              Malta     xgboost   4762.0
1835  population ages 65 and above   0.612194    0.456835   0.675895     -0.151855                                              Malta    base-gru   5583.0
413   population ages 65 and above   0.665237    0.568192   0.753785     -0.432630                                              Malta  simple-rnn   6062.0
1124  population ages 65 and above   0.681766    0.600105   0.774664     -0.513093                                              Malta   base-lstm   6189.0
2548          population ages 0-14   0.119932    0.022019   0.148387      0.988241                                   Marshall Islands     xgboost    944.0
3259          population ages 0-14   0.404205    0.346906   0.588987      0.814736                                   Marshall Islands          rf   4182.0
1126          population ages 0-14   1.297638    1.775476   1.332470      0.051811                                   Marshall Islands   base-lstm   8023.0
1837          population ages 0-14   1.440716    2.211870   1.487236     -0.181244                                   Marshall Islands    base-gru   8490.0
415           population ages 0-14   1.656410    2.907141   1.705034     -0.552552                                   Marshall Islands  simple-rnn   9121.0
3258         population ages 15-64   0.053948    0.006511   0.080690      0.990460                                   Marshall Islands          rf    345.0
2547         population ages 15-64   0.330570    0.131170   0.362174      0.807807                                   Marshall Islands     xgboost   3155.0
1125         population ages 15-64   1.664589    3.055402   1.747971     -3.476826                                   Marshall Islands   base-lstm   9857.0
1836         population ages 15-64   1.810787    3.633876   1.906273     -4.324417                                   Marshall Islands    base-gru  10245.0
414          population ages 15-64   2.057052    4.555622   2.134390     -5.674975                                   Marshall Islands  simple-rnn  10764.0
416   population ages 65 and above   0.185056    0.049327   0.222096      0.832944                                   Marshall Islands  simple-rnn   1982.0
1127  population ages 65 and above   0.195135    0.054520   0.233495      0.815356                                   Marshall Islands   base-lstm   2135.0
1838  population ages 65 and above   0.207448    0.056147   0.236953      0.809846                                   Marshall Islands    base-gru   2217.0
2549  population ages 65 and above   0.244119    0.109663   0.331155      0.628599                                   Marshall Islands     xgboost   2996.0
3260  population ages 65 and above   0.573107    0.492478   0.701768     -0.667894                                   Marshall Islands          rf   5811.0
3262          population ages 0-14   0.425967    0.271333   0.520896      0.288406                                         Mauritania          rf   4507.0
2551          population ages 0-14   0.777752    0.703653   0.838840     -0.845391                                         Mauritania     xgboost   6656.0
1840          population ages 0-14   1.536277    2.564993   1.601560     -5.726916                                         Mauritania    base-gru   9802.0
1129          population ages 0-14   1.646070    2.956813   1.719539     -6.754496                                         Mauritania   base-lstm  10095.0
418           population ages 0-14   1.970549    4.221941   2.054736    -10.072403                                         Mauritania  simple-rnn  10874.0
3261         population ages 15-64   0.416877    0.234027   0.483763      0.606421                                         Mauritania          rf   4094.0
2550         population ages 15-64   0.487999    0.342295   0.585060      0.424338                                         Mauritania     xgboost   4757.0
1839         population ages 15-64   1.319952    1.900329   1.378524     -2.195916                                         Mauritania    base-gru   8856.0
1128         population ages 15-64   1.520907    2.467701   1.570892     -3.150106                                         Mauritania   base-lstm   9469.0
417          population ages 15-64   1.795290    3.394444   1.842402     -4.708675                                         Mauritania  simple-rnn  10202.0
1130  population ages 65 and above   0.297110    0.118970   0.344921     -3.305412                                         Mauritania   base-lstm   4498.0
419   population ages 65 and above   0.303408    0.122764   0.350376     -3.442684                                         Mauritania  simple-rnn   4566.0
1841  population ages 65 and above   0.337145    0.165965   0.407388     -5.006112                                         Mauritania    base-gru   5089.0
3263  population ages 65 and above   0.610887    0.580769   0.762082    -20.017429                                         Mauritania          rf   7393.0
2552  population ages 65 and above   0.612174    0.585158   0.764956    -20.176256                                         Mauritania     xgboost   7411.0
3265          population ages 0-14   0.125536    0.031347   0.177052      0.990707                                          Mauritius          rf   1113.0
2554          population ages 0-14   0.584031    0.668655   0.817713      0.801767                                          Mauritius     xgboost   5325.0
1843          population ages 0-14   2.876305    8.749691   2.957988     -1.593987                                          Mauritius    base-gru  11067.0
1132          population ages 0-14   3.120585   10.488863   3.238651     -2.109593                                          Mauritius   base-lstm  11437.0
421           population ages 0-14   3.181391   10.766548   3.281242     -2.191917                                          Mauritius  simple-rnn  11514.0
2553         population ages 15-64   0.460817    0.269967   0.519584     -2.085934                                          Mauritius     xgboost   5400.0
3264         population ages 15-64   0.896109    1.262564   1.123639    -13.432089                                          Mauritius          rf   8635.0
1842         population ages 15-64   2.269059    5.858303   2.420393    -65.964935                                          Mauritius    base-gru  11858.0
420          population ages 15-64   2.582890    7.449509   2.729379    -84.153653                                          Mauritius  simple-rnn  12280.0
1131         population ages 15-64   3.290750   12.483866   3.533251   -141.700252                                          Mauritius   base-lstm  13086.0
1133  population ages 65 and above   0.777990    0.634780   0.796731      0.754379                                          Mauritius   base-lstm   5638.0
422   population ages 65 and above   1.068564    1.148043   1.071467      0.555777                                          Mauritius  simple-rnn   6883.0
2555  population ages 65 and above   0.873171    1.396644   1.181797      0.459583                                          Mauritius     xgboost   7009.0
1844  population ages 65 and above   1.122831    1.271255   1.127499      0.508101                                          Mauritius    base-gru   7130.0
3266  population ages 65 and above   1.240697    2.540758   1.593975      0.016881                                          Mauritius          rf   8409.0
2557          population ages 0-14   0.177185    0.053155   0.230555      0.981819                                             Mexico     xgboost   1623.0
3268          population ages 0-14   0.342341    0.155655   0.394532      0.946761                                             Mexico          rf   2999.0
1846          population ages 0-14   2.576785    7.263480   2.695084     -1.484319                                             Mexico    base-gru  10787.0
1135          population ages 0-14   2.732313    8.247433   2.871834     -1.820859                                             Mexico   base-lstm  11048.0
424           population ages 0-14   2.913648    9.289390   3.047850     -2.177239                                             Mexico  simple-rnn  11275.0
2556         population ages 15-64   0.091113    0.013260   0.115150      0.987188                                             Mexico     xgboost    663.0
3267         population ages 15-64   0.261238    0.082679   0.287539      0.920115                                             Mexico          rf   2354.0
1845         population ages 15-64   1.463956    2.323194   1.524203     -1.244707                                             Mexico    base-gru   8964.0
423          population ages 15-64   1.981726    4.260384   2.064070     -3.116451                                             Mexico  simple-rnn  10391.0
1134         population ages 15-64   2.127554    5.047274   2.246614     -3.876756                                             Mexico   base-lstm  10740.0
2558  population ages 65 and above   0.172629    0.050431   0.224569      0.895975                                             Mexico     xgboost   1849.0
3269  population ages 65 and above   0.221676    0.088430   0.297373      0.817594                                             Mexico          rf   2514.0
1136  population ages 65 and above   0.707732    0.572226   0.756456     -0.180331                                             Mexico   base-lstm   6015.0
425   population ages 65 and above   0.853559    0.845543   0.919534     -0.744102                                             Mexico  simple-rnn   6919.0
1847  population ages 65 and above   1.027119    1.190310   1.091013     -1.455253                                             Mexico    base-gru   7759.0
3271          population ages 0-14   0.106987    0.017693   0.133013      0.970495                              Micronesia, Fed. Sts.          rf    904.0
2560          population ages 0-14   0.178178    0.053936   0.232241      0.910052                              Micronesia, Fed. Sts.     xgboost   1874.0
1138          population ages 0-14   0.351785    0.144130   0.379644      0.759638                              Micronesia, Fed. Sts.   base-lstm   3375.0
1849          population ages 0-14   0.517592    0.303977   0.551341      0.493065                              Micronesia, Fed. Sts.    base-gru   4626.0
427           population ages 0-14   0.543211    0.335114   0.578890      0.441138                              Micronesia, Fed. Sts.  simple-rnn   4833.0
2559         population ages 15-64   0.202801    0.054615   0.233698     -0.605761                              Micronesia, Fed. Sts.     xgboost   3073.0
1137         population ages 15-64   0.642019    0.581401   0.762497    -16.094154                              Micronesia, Fed. Sts.   base-lstm   7373.0
3270         population ages 15-64   0.650908    0.595964   0.771987    -16.522326                              Micronesia, Fed. Sts.          rf   7449.0
1848         population ages 15-64   0.769852    0.793900   0.891010    -22.341960                              Micronesia, Fed. Sts.    base-gru   8071.0
426          population ages 15-64   0.791004    0.856150   0.925284    -24.172232                              Micronesia, Fed. Sts.  simple-rnn   8226.0
428   population ages 65 and above   0.203518    0.056084   0.236821      0.839756                              Micronesia, Fed. Sts.  simple-rnn   2136.0
1850  population ages 65 and above   0.207481    0.058695   0.242269      0.832298                              Micronesia, Fed. Sts.    base-gru   2193.0
1139  population ages 65 and above   0.223765    0.068260   0.261267      0.804967                              Micronesia, Fed. Sts.   base-lstm   2372.0
2561  population ages 65 and above   0.283301    0.113863   0.337436      0.674670                              Micronesia, Fed. Sts.     xgboost   3082.0
3272  population ages 65 and above   0.450903    0.306142   0.553301      0.125291                              Micronesia, Fed. Sts.          rf   4735.0
2563          population ages 0-14   0.109198    0.017294   0.131507      0.337329                         Middle East & North Africa     xgboost   1754.0
3274          population ages 0-14   0.591934    0.433769   0.658611    -15.620928                         Middle East & North Africa          rf   6931.0
1141          population ages 0-14   3.464092   14.272865   3.777945   -545.900625                         Middle East & North Africa   base-lstm  13454.0
1852          population ages 0-14   3.531761   14.706184   3.834864   -562.504332                         Middle East & North Africa    base-gru  13501.0
430           population ages 0-14   3.707367   16.195104   4.024314   -619.556012                         Middle East & North Africa  simple-rnn  13621.0
2562         population ages 15-64   0.465889    0.411287   0.641316     -2.767956                         Middle East & North Africa     xgboost   5912.0
3273         population ages 15-64   1.518830    3.065741   1.750926    -27.086428                         Middle East & North Africa          rf  10553.0
1851         population ages 15-64   2.389643    7.018678   2.649279    -63.300802                         Middle East & North Africa    base-gru  12069.0
1140         population ages 15-64   2.663604    8.853397   2.975466    -80.109359                         Middle East & North Africa   base-lstm  12430.0
429          population ages 15-64   2.710604    8.996420   2.999403    -81.419653                         Middle East & North Africa  simple-rnn  12473.0
2564  population ages 65 and above   0.073888    0.008254   0.090852      0.940490                         Middle East & North Africa     xgboost    652.0
3275  population ages 65 and above   0.481213    0.391428   0.625642     -1.822154                         Middle East & North Africa          rf   5717.0
1142  population ages 65 and above   1.220139    1.566640   1.251655    -10.295307                         Middle East & North Africa   base-lstm   9187.0
431   population ages 65 and above   1.367925    1.996871   1.413107    -13.397227                         Middle East & North Africa  simple-rnn   9698.0
1853  population ages 65 and above   1.445619    2.215875   1.488582    -14.976220                         Middle East & North Africa    base-gru   9910.0
2566          population ages 0-14   0.598660    0.462354   0.679966     -8.435790  Middle East & North Africa (IDA & IBRD countries)     xgboost   6792.0
3277          population ages 0-14   1.159040    1.800791   1.341936    -35.750854  Middle East & North Africa (IDA & IBRD countries)          rf   9730.0
1144          population ages 0-14   3.639536   15.950621   3.993823   -324.522943  Middle East & North Africa (IDA & IBRD countries)   base-lstm  13490.0
1855          population ages 0-14   3.656946   15.984273   3.998034   -325.209725  Middle East & North Africa (IDA & IBRD countries)    base-gru  13501.0
433           population ages 0-14   3.894747   18.081798   4.252270   -368.016357  Middle East & North Africa (IDA & IBRD countries)  simple-rnn  13627.0
2565         population ages 15-64   0.950789    1.246203   1.116335     -4.479948  Middle East & North Africa (IDA & IBRD countries)     xgboost   8224.0
3276         population ages 15-64   1.626276    3.743783   1.934886    -15.462600  Middle East & North Africa (IDA & IBRD countries)          rf  10655.0
1854         population ages 15-64   2.415514    7.277764   2.697733    -31.002628  Middle East & North Africa (IDA & IBRD countries)    base-gru  11966.0
1143         population ages 15-64   2.722386    9.380896   3.062825    -40.250766  Middle East & North Africa (IDA & IBRD countries)   base-lstm  12369.0
432          population ages 15-64   2.812807    9.820301   3.133736    -42.182971  Middle East & North Africa (IDA & IBRD countries)  simple-rnn  12443.0
3278  population ages 65 and above   0.150610    0.039996   0.199989      0.755012  Middle East & North Africa (IDA & IBRD countries)          rf   1934.0
2567  population ages 65 and above   0.223820    0.080911   0.284448      0.504392  Middle East & North Africa (IDA & IBRD countries)     xgboost   2797.0
1145  population ages 65 and above   1.212272    1.561672   1.249669     -8.565802  Middle East & North Africa (IDA & IBRD countries)   base-lstm   9117.0
434   population ages 65 and above   1.334675    1.921479   1.386174    -10.769749  Middle East & North Africa (IDA & IBRD countries)  simple-rnn   9563.0
1856  population ages 65 and above   1.442222    2.229876   1.493277    -12.658792  Middle East & North Africa (IDA & IBRD countries)    base-gru   9846.0
2569          population ages 0-14   0.601012    0.475293   0.689415     -9.164849  Middle East & North Africa (excluding high inc...     xgboost   6857.0
3280          population ages 0-14   1.256289    2.043773   1.429606    -42.709090  Middle East & North Africa (excluding high inc...          rf  10003.0
1147          population ages 0-14   3.615139   15.748028   3.968378   -335.794793  Middle East & North Africa (excluding high inc...   base-lstm  13474.0
1858          population ages 0-14   3.630566   15.766639   3.970723   -336.192829  Middle East & North Africa (excluding high inc...    base-gru  13483.0
436           population ages 0-14   3.872576   17.887415   4.229352   -381.548754  Middle East & North Africa (excluding high inc...  simple-rnn  13627.0
2568         population ages 15-64   1.001502    1.388562   1.178373     -5.458733  Middle East & North Africa (excluding high inc...     xgboost   8522.0
3279         population ages 15-64   1.896933    5.113808   2.261373    -22.786275  Middle East & North Africa (excluding high inc...          rf  11324.0
1857         population ages 15-64   2.397616    7.179693   2.679495    -32.395492  Middle East & North Africa (excluding high inc...    base-gru  11944.0
1146         population ages 15-64   2.703591    9.263124   3.043538    -42.086330  Middle East & North Africa (excluding high inc...   base-lstm  12351.0
435          population ages 15-64   2.799864    9.739209   3.120771    -44.300782  Middle East & North Africa (excluding high inc...  simple-rnn  12444.0
3281  population ages 65 and above   0.120667    0.029747   0.172473      0.815587  Middle East & North Africa (excluding high inc...          rf   1556.0
2570  population ages 65 and above   0.218940    0.083453   0.288882      0.482646  Middle East & North Africa (excluding high inc...     xgboost   2835.0
1148  population ages 65 and above   1.198624    1.526956   1.235700     -8.466169  Middle East & North Africa (excluding high inc...   base-lstm   9081.0
437   population ages 65 and above   1.318934    1.876927   1.370010    -10.635776  Middle East & North Africa (excluding high inc...  simple-rnn   9510.0
1859  population ages 65 and above   1.428104    2.186986   1.478846    -12.557944  Middle East & North Africa (excluding high inc...    base-gru   9818.0
3283          population ages 0-14   0.441045    0.279613   0.528785      0.494615                                      Middle income          rf   4420.0
2572          population ages 0-14   0.522826    0.475326   0.689439      0.140876                                      Middle income     xgboost   5304.0
1861          population ages 0-14   2.962990    9.963417   3.156488    -17.008311                                      Middle income    base-gru  12216.0
1150          population ages 0-14   3.064036   10.768218   3.281496    -18.462942                                      Middle income   base-lstm  12354.0
439           population ages 0-14   3.182424   11.487291   3.389291    -19.762626                                      Middle income  simple-rnn  12488.0
3282         population ages 15-64   0.643017    0.592268   0.769589    -42.437639                                      Middle income          rf   7718.0
2571         population ages 15-64   0.719820    0.673651   0.820762    -48.406394                                      Middle income     xgboost   8037.0
1860         population ages 15-64   1.423275    2.443913   1.563302   -178.239596                                      Middle income    base-gru  10637.0
438          population ages 15-64   1.730853    3.576169   1.891076   -261.280675                                      Middle income  simple-rnn  11318.0
1149         population ages 15-64   1.968013    4.722966   2.173239   -345.388201                                      Middle income   base-lstm  11827.0
3284  population ages 65 and above   0.237940    0.121463   0.348516      0.750487                                      Middle income          rf   2933.0
2573  population ages 65 and above   0.313581    0.204897   0.452656      0.579096                                      Middle income     xgboost   3778.0
1151  population ages 65 and above   1.233818    1.611234   1.269344     -2.309833                                      Middle income   base-lstm   8590.0
440   population ages 65 and above   1.423242    2.169214   1.472825     -3.456048                                      Middle income  simple-rnn   9293.0
1862  population ages 65 and above   1.514323    2.444627   1.563530     -4.021807                                      Middle income    base-gru   9578.0
2575          population ages 0-14   0.267882    0.110698   0.332713      0.363259                                            Moldova     xgboost   3281.0
3286          population ages 0-14   0.582479    0.441279   0.664289     -1.538264                                            Moldova          rf   5967.0
1864          population ages 0-14   4.902941   24.838809   4.983855   -141.874276                                            Moldova    base-gru  13705.0
1153          population ages 0-14   5.009471   26.050430   5.103962   -148.843593                                            Moldova   base-lstm  13729.0
442           population ages 0-14   5.138561   27.311247   5.226016   -156.095887                                            Moldova  simple-rnn  13758.0
2574         population ages 15-64   0.276101    0.189308   0.435095      0.824888                                            Moldova     xgboost   3310.0
3285         population ages 15-64   0.804539    0.834941   0.913751      0.227669                                            Moldova          rf   6413.0
1863         population ages 15-64   3.968836   16.768043   4.094880    -14.510639                                            Moldova    base-gru  12874.0
441          population ages 15-64   4.208266   18.870127   4.343976    -16.455091                                            Moldova  simple-rnn  13011.0
1152         population ages 15-64   4.832311   24.850250   4.985003    -21.986776                                            Moldova   base-lstm  13264.0
2576  population ages 65 and above   0.044569    0.004434   0.066590      0.988613                                            Moldova     xgboost    281.0
3287  population ages 65 and above   0.329706    0.137537   0.370859      0.646798                                            Moldova          rf   3426.0
1154  population ages 65 and above   0.356408    0.160429   0.400535      0.588010                                            Moldova   base-lstm   3655.0
443   population ages 65 and above   0.553262    0.330269   0.574691      0.151849                                            Moldova  simple-rnn   4995.0
1865  population ages 65 and above   0.719338    0.545727   0.738733     -0.401459                                            Moldova    base-gru   6078.0
2578          population ages 0-14   0.707275    0.858218   0.926401      0.465751                                           Mongolia     xgboost   6157.0
3289          population ages 0-14   1.813874    4.872641   2.207406     -2.033262                                           Mongolia          rf  10222.0
1156          population ages 0-14   5.948615   40.317004   6.349567    -24.097694                                           Mongolia   base-lstm  13434.0
1867          population ages 0-14   6.276164   44.624644   6.680168    -26.779238                                           Mongolia    base-gru  13482.0
445           population ages 0-14   6.316819   45.177179   6.721397    -27.123196                                           Mongolia  simple-rnn  13489.0
2577         population ages 15-64   0.498456    0.422105   0.649696      0.802520                                           Mongolia     xgboost   4588.0
3288         population ages 15-64   1.979398    5.637120   2.374262     -1.637310                                           Mongolia          rf  10365.0
1155         population ages 15-64   5.736198   37.515028   6.124951    -16.551295                                           Mongolia   base-lstm  13271.0
1866         population ages 15-64   5.778561   37.937079   6.159308    -16.748750                                           Mongolia    base-gru  13279.0
444          population ages 15-64   6.024659   40.658768   6.376423    -18.022085                                           Mongolia  simple-rnn  13324.0
2579  population ages 65 and above   0.081700    0.008442   0.091879      0.797125                                           Mongolia     xgboost   1017.0
3290  population ages 65 and above   0.089418    0.018787   0.137064      0.548514                                           Mongolia          rf   1559.0
1157  population ages 65 and above   1.150933    1.345909   1.160133    -31.345441                                           Mongolia   base-lstm   9304.0
1868  population ages 65 and above   1.305378    1.727897   1.314495    -40.525552                                           Mongolia    base-gru   9842.0
446   population ages 65 and above   1.315048    1.768823   1.329971    -41.509089                                           Mongolia  simple-rnn   9887.0
3292          population ages 0-14   0.042243    0.003951   0.062860     -4.560034                                         Montenegro          rf   2405.0
2581          population ages 0-14   0.104507    0.022885   0.151278    -31.202229                                         Montenegro     xgboost   3786.0
1870          population ages 0-14   1.677919    2.853658   1.689277  -4014.458978                                         Montenegro    base-gru  11248.0
1159          population ages 0-14   1.903959    3.698151   1.923058  -5202.768556                                         Montenegro   base-lstm  11694.0
448           population ages 0-14   2.064599    4.316532   2.077626  -6072.909075                                         Montenegro  simple-rnn  11979.0
2580         population ages 15-64   0.090044    0.011400   0.106770      0.921615                                         Montenegro     xgboost    836.0
3291         population ages 15-64   0.161732    0.036095   0.189988      0.751812                                         Montenegro          rf   1928.0
1869         population ages 15-64   1.041716    1.171353   1.082290     -7.054096                                         Montenegro    base-gru   8442.0
447          population ages 15-64   1.440376    2.202368   1.484038    -14.143249                                         Montenegro  simple-rnn   9875.0
1158         population ages 15-64   1.680343    2.993131   1.730067    -19.580450                                         Montenegro   base-lstm  10517.0
1871  population ages 65 and above   0.092435    0.013616   0.116687      0.918038                                         Montenegro    base-gru    921.0
3293  population ages 65 and above   0.144156    0.031175   0.176564      0.812341                                         Montenegro          rf   1670.0
1160  population ages 65 and above   0.161995    0.031095   0.176337      0.812823                                         Montenegro   base-lstm   1728.0
449   population ages 65 and above   0.164030    0.031945   0.178732      0.807704                                         Montenegro  simple-rnn   1779.0
2582  population ages 65 and above   0.285863    0.110341   0.332176      0.335793                                         Montenegro     xgboost   3335.0
3295          population ages 0-14   0.207512    0.073061   0.270297      0.767327                                            Morocco          rf   2429.0
2584          population ages 0-14   0.992448    1.588774   1.260466     -4.059693                                            Morocco     xgboost   8581.0
1162          population ages 0-14   3.360095   12.892439   3.590604    -40.057945                                            Morocco   base-lstm  12881.0
1873          population ages 0-14   3.475381   13.662001   3.696214    -42.508730                                            Morocco    base-gru  12987.0
451           population ages 0-14   3.568354   14.394857   3.794056    -44.842624                                            Morocco  simple-rnn  13079.0
2583         population ages 15-64   0.674454    0.545505   0.738583    -20.296001                                            Morocco     xgboost   7443.0
3294         population ages 15-64   0.716520    0.802118   0.895610    -30.313956                                            Morocco          rf   8100.0
1872         population ages 15-64   2.680416    8.358137   2.891044   -325.293992                                            Morocco    base-gru  12607.0
450          population ages 15-64   2.918904    9.817910   3.133354   -382.282196                                            Morocco  simple-rnn  12864.0
1161         population ages 15-64   3.060796   11.041647   3.322897   -430.055778                                            Morocco   base-lstm  13054.0
2585  population ages 65 and above   0.215413    0.085729   0.292796      0.825874                                            Morocco     xgboost   2455.0
3296  population ages 65 and above   0.249586    0.127134   0.356558      0.741777                                            Morocco          rf   3032.0
1163  population ages 65 and above   0.657242    0.434776   0.659375      0.116920                                            Morocco   base-lstm   5472.0
452   population ages 65 and above   0.882200    0.800546   0.894732     -0.626000                                            Morocco  simple-rnn   6850.0
1874  population ages 65 and above   0.981179    0.985397   0.992672     -1.001454                                            Morocco    base-gru   7327.0
2587          population ages 0-14   0.114957    0.022507   0.150025      0.926516                                         Mozambique     xgboost   1154.0
3298          population ages 0-14   0.252474    0.098838   0.314386      0.677304                                         Mozambique          rf   2885.0
1876          population ages 0-14   0.745610    0.632167   0.795089     -1.063950                                         Mozambique    base-gru   6523.0
1165          population ages 0-14   0.863403    0.839441   0.916210     -1.740674                                         Mozambique   base-lstm   7215.0
454           population ages 0-14   1.126262    1.405953   1.185729     -3.590270                                         Mozambique  simple-rnn   8498.0
3297         population ages 15-64   0.125804    0.023042   0.151796      0.923896                                         Mozambique          rf   1210.0
2586         population ages 15-64   0.150789    0.038069   0.195112      0.874266                                         Mozambique     xgboost   1695.0
1875         population ages 15-64   0.268751    0.091562   0.302593      0.697587                                         Mozambique    base-gru   2871.0
1164         population ages 15-64   0.424913    0.209915   0.458165      0.306690                                         Mozambique   base-lstm   4250.0
453          population ages 15-64   0.657409    0.475717   0.689722     -0.571203                                         Mozambique  simple-rnn   5892.0
2588  population ages 65 and above   0.026083    0.000936   0.030594     -8.497118                                         Mozambique     xgboost   2529.0
3299  population ages 65 and above   0.024019    0.000966   0.031088     -8.806684                                         Mozambique          rf   2541.0
455   population ages 65 and above   0.422643    0.190320   0.436257  -1930.125686                                         Mozambique  simple-rnn   6681.0
1166  population ages 65 and above   0.480924    0.242617   0.492562  -2460.767911                                         Mozambique   base-lstm   7017.0
1877  population ages 65 and above   0.529322    0.293517   0.541772  -2977.237764                                         Mozambique    base-gru   7297.0
3301          population ages 0-14   0.185365    0.049180   0.221766      0.964983                                            Myanmar          rf   1645.0
2590          population ages 0-14   0.262224    0.095718   0.309383      0.931848                                            Myanmar     xgboost   2432.0
1879          population ages 0-14   2.894014    9.326804   3.053982     -5.640787                                            Myanmar    base-gru  11699.0
1168          population ages 0-14   3.031924   10.291528   3.208041     -6.327681                                            Myanmar   base-lstm  11894.0
457           population ages 0-14   3.151514   11.024600   3.320331     -6.849636                                            Myanmar  simple-rnn  12044.0
3300         population ages 15-64   0.246699    0.076724   0.276992      0.853915                                            Myanmar          rf   2410.0
2589         population ages 15-64   0.644528    0.689967   0.830643     -0.313711                                            Myanmar     xgboost   6236.0
1878         population ages 15-64   0.967762    1.069315   1.034077     -1.035997                                            Myanmar    base-gru   7419.0
456          population ages 15-64   1.231543    1.698493   1.303263     -2.233964                                            Myanmar  simple-rnn   8635.0
1167         population ages 15-64   1.381830    2.252910   1.500970     -3.289585                                            Myanmar   base-lstm   9276.0
3302  population ages 65 and above   0.560042    0.562042   0.749695     -1.324042                                            Myanmar          rf   6159.0
2591  population ages 65 and above   0.653076    0.772379   0.878851     -2.193785                                            Myanmar     xgboost   6933.0
1169  population ages 65 and above   2.013637    4.381403   2.093180    -17.117083                                            Myanmar   base-lstm  11142.0
458   population ages 65 and above   2.139238    4.989585   2.233738    -19.631914                                            Myanmar  simple-rnn  11357.0
1880  population ages 65 and above   2.177931    5.120907   2.262942    -20.174932                                            Myanmar    base-gru  11410.0
2593          population ages 0-14   0.542444    0.469329   0.685076    -32.733191                                            Namibia     xgboost   7186.0
3304          population ages 0-14   0.664412    0.689703   0.830484    -48.572668                                            Namibia          rf   7990.0
1171          population ages 0-14   2.270985    5.989124   2.447269   -429.470714                                            Namibia   base-lstm  12187.0
1882          population ages 0-14   2.411442    6.727603   2.593762   -482.549153                                            Namibia    base-gru  12381.0
460           population ages 0-14   2.526579    7.357413   2.712455   -527.816975                                            Namibia  simple-rnn  12525.0
2592         population ages 15-64   0.081093    0.012267   0.110756     -1.217278                                            Namibia     xgboost   2178.0
3303         population ages 15-64   0.859583    1.241952   1.114429   -223.484802                                            Namibia          rf   9223.0
1170         population ages 15-64   1.841356    4.014963   2.003737   -724.710730                                            Namibia   base-lstm  11689.0
1881         population ages 15-64   1.863461    4.045057   2.011233   -730.150285                                            Namibia    base-gru  11710.0
459          population ages 15-64   2.040979    4.829525   2.197618   -871.944024                                            Namibia  simple-rnn  12009.0
3305  population ages 65 and above   0.042584    0.002817   0.053080      0.887754                                            Namibia          rf    546.0
2594  population ages 65 and above   0.150968    0.034434   0.185565     -0.371844                                            Namibia     xgboost   2577.0
1172  population ages 65 and above   0.467484    0.238242   0.488100     -8.491438                                            Namibia   base-lstm   5899.0
461   population ages 65 and above   0.532178    0.314832   0.561099    -11.542750                                            Namibia  simple-rnn   6392.0
1883  population ages 65 and above   0.611067    0.405777   0.637007    -15.165969                                            Namibia    base-gru   6884.0
3307          population ages 0-14   0.287436    0.133568   0.365469      0.974115                                              Nepal          rf   2626.0
2596          population ages 0-14   1.087270    1.526310   1.235439      0.704206                                              Nepal     xgboost   7177.0
1885          population ages 0-14   2.296044    6.122208   2.474310     -0.186463                                              Nepal    base-gru  10075.0
1174          population ages 0-14   2.381431    6.553957   2.560070     -0.270134                                              Nepal   base-lstm  10212.0
463           population ages 0-14   2.526857    7.280808   2.698297     -0.410995                                              Nepal  simple-rnn  10438.0
3306         population ages 15-64   0.274481    0.142688   0.377741      0.955516                                              Nepal          rf   2746.0
1884         population ages 15-64   1.561308    2.767448   1.663565      0.137231                                              Nepal    base-gru   8673.0
462          population ages 15-64   1.700685    3.264839   1.806886     -0.017834                                              Nepal  simple-rnn   9053.0
2595         population ages 15-64   1.696034    3.625136   1.903979     -0.130159                                              Nepal     xgboost   9221.0
1173         population ages 15-64   1.754990    3.536250   1.880492     -0.102448                                              Nepal   base-lstm   9221.0
2597  population ages 65 and above   0.198017    0.054398   0.233234      0.773745                                              Nepal     xgboost   2225.0
3308  population ages 65 and above   0.218163    0.065655   0.256232      0.726926                                              Nepal          rf   2461.0
1175  population ages 65 and above   0.542017    0.435187   0.659687     -0.810049                                              Nepal   base-lstm   5655.0
464   population ages 65 and above   0.614522    0.560961   0.748973     -1.333172                                              Nepal  simple-rnn   6271.0
1886  population ages 65 and above   0.727163    0.717292   0.846931     -1.983387                                              Nepal    base-gru   6915.0
3310          population ages 0-14   0.528891    0.547079   0.739648     -0.048082                                        Netherlands          rf   5596.0
2599          population ages 0-14   0.836768    1.079653   1.039063     -1.068374                                        Netherlands     xgboost   7286.0
1888          population ages 0-14   2.305740    5.406329   2.325151     -9.357323                                        Netherlands    base-gru  11213.0
1177          population ages 0-14   2.558388    6.709816   2.590331    -11.854512                                        Netherlands   base-lstm  11588.0
466           population ages 0-14   2.958241    9.006324   3.001054    -16.254110                                        Netherlands  simple-rnn  12093.0
2598         population ages 15-64   1.016876    1.429812   1.195747     -1.067674                                        Netherlands     xgboost   7877.0
1887         population ages 15-64   1.095957    1.563438   1.250375     -1.260912                                        Netherlands    base-gru   8162.0
3309         population ages 15-64   1.413232    2.690964   1.640416     -2.891446                                        Netherlands          rf   9439.0
465          population ages 15-64   1.736078    3.752863   1.937231     -4.427074                                        Netherlands  simple-rnn  10241.0
1176         population ages 15-64   2.272643    6.137529   2.477404     -7.875578                                        Netherlands   base-lstm  11251.0
1889  population ages 65 and above   0.339502    0.167574   0.409359      0.929594                                        Netherlands    base-gru   3127.0
467   population ages 65 and above   0.410078    0.200667   0.447959      0.915690                                        Netherlands  simple-rnn   3470.0
2600  population ages 65 and above   0.571032    0.436589   0.660749      0.816567                                        Netherlands     xgboost   4738.0
1178  population ages 65 and above   0.626590    0.475666   0.689685      0.800149                                        Netherlands   base-lstm   4980.0
3311  population ages 65 and above   1.360900    2.489753   1.577895     -0.046069                                        Netherlands          rf   8524.0
2842          population ages 0-14   0.485201    0.313664   0.560057      0.289163                                        New Zealand     xgboost   4742.0
3553          population ages 0-14   0.543933    0.431704   0.657042      0.021655                                        New Zealand          rf   5303.0
2131          population ages 0-14   2.451403    6.360768   2.522056    -13.415027                                        New Zealand    base-gru  11547.0
709           population ages 0-14   2.932657    9.191117   3.031685    -19.829276                                        New Zealand  simple-rnn  12175.0
1420          population ages 0-14   2.979613    9.503155   3.082719    -20.536430                                        New Zealand   base-lstm  12261.0
2130         population ages 15-64   0.394436    0.185871   0.431127     -0.658802                                        New Zealand    base-gru   4521.0
3552         population ages 15-64   0.722918    0.658135   0.811255     -4.873515                                        New Zealand          rf   7196.0
2841         population ages 15-64   0.969272    1.366089   1.168798    -11.191639                                        New Zealand     xgboost   8759.0
708          population ages 15-64   1.128200    1.505459   1.226972    -12.435445                                        New Zealand  simple-rnn   9105.0
1419         population ages 15-64   1.498400    2.613802   1.616726    -22.326836                                        New Zealand   base-lstm  10310.0
2843  population ages 65 and above   0.225817    0.073531   0.271166      0.920655                                        New Zealand     xgboost   2164.0
3554  population ages 65 and above   0.655604    0.552580   0.743357      0.403729                                        New Zealand          rf   5599.0
710   population ages 65 and above   0.916342    0.932847   0.965840     -0.006605                                        New Zealand  simple-rnn   6809.0
1421  population ages 65 and above   0.971627    1.042177   1.020871     -0.124580                                        New Zealand   base-lstm   7072.0
2132  population ages 65 and above   1.269288    1.761352   1.327159     -0.900619                                        New Zealand    base-gru   8366.0
3313          population ages 0-14   0.442823    0.283656   0.532593      0.876744                                          Nicaragua          rf   3967.0
2602          population ages 0-14   1.099508    1.433822   1.197423      0.376966                                          Nicaragua     xgboost   7363.0
1180          population ages 0-14   3.182152   12.081426   3.475835     -4.249705                                          Nicaragua   base-lstm  11947.0
1891          population ages 0-14   3.210644   12.269101   3.502728     -4.331254                                          Nicaragua    base-gru  11980.0
469           population ages 0-14   3.486718   14.292700   3.780569     -5.210563                                          Nicaragua  simple-rnn  12284.0
3312         population ages 15-64   0.187271    0.063695   0.252379      0.946487                                          Nicaragua          rf   1900.0
2601         population ages 15-64   0.839633    1.116389   1.056593      0.062078                                          Nicaragua     xgboost   6892.0
1890         population ages 15-64   2.329877    6.469329   2.543488     -4.435137                                          Nicaragua    base-gru  11066.0
1179         population ages 15-64   2.547256    7.897405   2.810232     -5.634919                                          Nicaragua   base-lstm  11455.0
468          population ages 15-64   2.704758    8.643249   2.939940     -6.261531                                          Nicaragua  simple-rnn  11621.0
3314  population ages 65 and above   0.191805    0.056278   0.237230      0.702992                                          Nicaragua          rf   2330.0
2603  population ages 65 and above   0.236565    0.089465   0.299108      0.527845                                          Nicaragua     xgboost   2910.0
1181  population ages 65 and above   0.906861    0.933012   0.965925     -3.923982                                          Nicaragua   base-lstm   7779.0
470   population ages 65 and above   0.981630    1.124686   1.060512     -4.935545                                          Nicaragua  simple-rnn   8175.0
1892  population ages 65 and above   1.139016    1.464954   1.210353     -6.731316                                          Nicaragua    base-gru   8837.0
3316          population ages 0-14   0.171205    0.055783   0.236184     -5.196197                                              Niger          rf   3810.0
472           population ages 0-14   0.408595    0.209923   0.458174    -22.317532                                              Niger  simple-rnn   6029.0
1183          population ages 0-14   0.645180    0.496063   0.704318    -54.100999                                              Niger   base-lstm   7562.0
2605          population ages 0-14   0.579773    0.670982   0.819135    -73.530395                                              Niger     xgboost   7889.0
1894          population ages 0-14   0.798053    0.760808   0.872243    -83.507910                                              Niger    base-gru   8413.0
3315         population ages 15-64   0.161881    0.029061   0.170473     -2.643727                                              Niger          rf   3113.0
471          population ages 15-64   0.405373    0.232340   0.482017    -28.131083                                              Niger  simple-rnn   6175.0
1182         population ages 15-64   0.540571    0.392671   0.626635    -48.233559                                              Niger   base-lstm   7081.0
2604         population ages 15-64   0.436723    0.479748   0.692639    -59.151387                                              Niger     xgboost   7153.0
1893         population ages 15-64   0.670988    0.586697   0.765962    -72.560757                                              Niger    base-gru   7863.0
2606  population ages 65 and above   0.017684    0.000405   0.020118      0.148846                                              Niger     xgboost   1105.0
3317  population ages 65 and above   0.018306    0.000463   0.021511      0.026913                                              Niger          rf   1186.0
473   population ages 65 and above   0.109507    0.016269   0.127548    -33.211534                                              Niger  simple-rnn   3675.0
1895  population ages 65 and above   0.116539    0.017460   0.132136    -35.717255                                              Niger    base-gru   3763.0
1184  population ages 65 and above   0.173555    0.036202   0.190269    -75.130703                                              Niger   base-lstm   4462.0
2608          population ages 0-14   0.099702    0.012651   0.112475      0.895714                                            Nigeria     xgboost    999.0
1897          population ages 0-14   0.401422    0.165971   0.407395     -0.368192                                            Nigeria    base-gru   4326.0
3319          population ages 0-14   0.332972    0.231810   0.481466     -0.910941                                            Nigeria          rf   4642.0
1186          population ages 0-14   0.498309    0.256385   0.506344     -1.113528                                            Nigeria   base-lstm   5159.0
475           population ages 0-14   0.773724    0.621278   0.788212     -4.121553                                            Nigeria  simple-rnn   7123.0
3318         population ages 15-64   0.099697    0.021671   0.147211      0.845751                                            Nigeria          rf   1279.0
2607         population ages 15-64   0.161289    0.035400   0.188148      0.748036                                            Nigeria     xgboost   1924.0
474          population ages 15-64   0.147924    0.042558   0.206295      0.697088                                            Nigeria  simple-rnn   2023.0
1185         population ages 15-64   0.239542    0.113215   0.336474      0.194169                                            Nigeria   base-lstm   3305.0
1896         population ages 15-64   0.462307    0.310030   0.556803     -1.206700                                            Nigeria    base-gru   5296.0
3320  population ages 65 and above   0.043707    0.003092   0.055609     -1.860910                                            Nigeria          rf   1978.0
2609  population ages 65 and above   0.111049    0.021402   0.146295    -18.800293                                            Nigeria     xgboost   3602.0
476   population ages 65 and above   0.611604    0.428418   0.654536   -395.353195                                            Nigeria  simple-rnn   7663.0
1187  population ages 65 and above   0.660229    0.492103   0.701500   -454.272004                                            Nigeria   base-lstm   7927.0
1898  population ages 65 and above   0.746291    0.626587   0.791572   -578.690440                                            Nigeria    base-gru   8383.0
3322          population ages 0-14   0.156951    0.031463   0.177378      0.870287                                      North America          rf   1600.0
2611          population ages 0-14   0.244580    0.098632   0.314057      0.593369                                      North America     xgboost   2940.0
1900          population ages 0-14   2.340305    5.894587   2.427877    -23.301660                                      North America    base-gru  11637.0
478           population ages 0-14   2.857578    8.840276   2.973260    -35.445874                                      North America  simple-rnn  12295.0
1189          population ages 0-14   2.872113    8.959186   2.993190    -35.936106                                      North America   base-lstm  12319.0
1899         population ages 15-64   0.570698    0.467400   0.683667      0.115057                                      North America    base-gru   5396.0
2610         population ages 15-64   0.871686    1.168869   1.081143     -1.213054                                      North America     xgboost   7475.0
3321         population ages 15-64   0.871859    1.234833   1.111230     -1.337945                                      North America          rf   7586.0
477          population ages 15-64   1.069960    1.701439   1.304392     -2.221383                                      North America  simple-rnn   8463.0
1188         population ages 15-64   1.464476    3.064954   1.750701     -4.802966                                      North America   base-lstm   9851.0
3323  population ages 65 and above   0.800279    1.001312   1.000656      0.323906                                      North America          rf   6575.0
479   population ages 65 and above   0.980031    0.975200   0.987522      0.341537                                      North America  simple-rnn   6747.0
1190  population ages 65 and above   0.991838    0.999608   0.999804      0.325056                                      North America   base-lstm   6806.0
2612  population ages 65 and above   0.856395    1.205361   1.097889      0.186130                                      North America     xgboost   6943.0
1901  population ages 65 and above   1.227637    1.517397   1.231827     -0.024559                                      North America    base-gru   7800.0
3325          population ages 0-14   0.034948    0.001670   0.040863      0.982083                                    North Macedonia          rf    190.0
2614          population ages 0-14   0.336046    0.168549   0.410547     -0.808566                                    North Macedonia     xgboost   4355.0
1903          population ages 0-14   1.766484    3.237042   1.799178    -33.734184                                    North Macedonia    base-gru  10857.0
1192          population ages 0-14   2.010746    4.236418   2.058256    -44.457720                                    North Macedonia   base-lstm  11401.0
481           population ages 0-14   2.098962    4.591275   2.142726    -48.265411                                    North Macedonia  simple-rnn  11529.0
2613         population ages 15-64   0.088173    0.014374   0.119892      0.856544                                    North Macedonia     xgboost   1059.0
3324         population ages 15-64   0.145443    0.048247   0.219652      0.518491                                    North Macedonia          rf   2226.0
1902         population ages 15-64   0.886742    0.873932   0.934843     -7.721919                                    North Macedonia    base-gru   7960.0
480          population ages 15-64   1.107193    1.374131   1.172233    -12.713949                                    North Macedonia  simple-rnn   8964.0
1191         population ages 15-64   1.679840    3.070313   1.752231    -29.641997                                    North Macedonia   base-lstm  10698.0
2615  population ages 65 and above   0.110970    0.017325   0.131623      0.954329                                    North Macedonia     xgboost    966.0
3326  population ages 65 and above   0.234396    0.086251   0.293685      0.772625                                    North Macedonia          rf   2618.0
1193  population ages 65 and above   0.527351    0.287413   0.536109      0.242319                                    North Macedonia   base-lstm   4761.0
482   population ages 65 and above   0.617606    0.391256   0.625505     -0.031436                                    North Macedonia  simple-rnn   5368.0
1904  population ages 65 and above   0.735288    0.550754   0.742128     -0.451905                                    North Macedonia    base-gru   6140.0
3328          population ages 0-14   0.133094    0.026271   0.162082      0.923262                                             Norway          rf   1302.0
2617          population ages 0-14   0.725103    0.673114   0.820435     -0.966216                                             Norway     xgboost   6546.0
1906          population ages 0-14   1.895353    3.734430   1.932467     -9.908541                                             Norway    base-gru  10669.0
484           population ages 0-14   2.435222    6.212108   2.492410    -17.146018                                             Norway  simple-rnn  11603.0
1195          population ages 0-14   2.453394    6.313673   2.512702    -17.442695                                             Norway   base-lstm  11625.0
1905         population ages 15-64   0.440838    0.253522   0.503510     -0.336261                                             Norway    base-gru   4770.0
3327         population ages 15-64   0.602711    0.552086   0.743025     -1.909925                                             Norway          rf   6361.0
483          population ages 15-64   0.739807    0.826230   0.908972     -3.354879                                             Norway  simple-rnn   7331.0
2616         population ages 15-64   0.876908    1.258246   1.121716     -5.631943                                             Norway     xgboost   8246.0
1194         population ages 15-64   1.053487    1.635249   1.278769     -7.619043                                             Norway   base-lstm   8949.0
3329  population ages 65 and above   0.423191    0.254331   0.504312      0.754422                                             Norway          rf   4032.0
485   population ages 65 and above   0.879841    0.809968   0.899982      0.217906                                             Norway  simple-rnn   6477.0
1196  population ages 65 and above   0.947579    0.933756   0.966311      0.098378                                             Norway   base-lstm   6790.0
2618  population ages 65 and above   0.886572    0.979292   0.989592      0.054409                                             Norway     xgboost   6801.0
1907  population ages 65 and above   1.129157    1.299428   1.139925     -0.254710                                             Norway    base-gru   7594.0
3331          population ages 0-14   0.033992    0.002058   0.045365      0.992644                                       OECD members          rf    157.0
2620          population ages 0-14   0.677365    0.663511   0.814562     -1.371754                                       OECD members     xgboost   6593.0
1909          population ages 0-14   2.669742    7.580680   2.753303    -26.097519                                       OECD members    base-gru  12051.0
1198          population ages 0-14   3.160217   10.689406   3.269466    -37.209818                                       OECD members   base-lstm  12602.0
487           population ages 0-14   3.247841   11.280270   3.358611    -39.321889                                       OECD members  simple-rnn  12714.0
3330         population ages 15-64   0.563158    0.549151   0.741047     -0.605630                                       OECD members          rf   5901.0
1908         population ages 15-64   0.710523    0.750545   0.866340     -1.194473                                       OECD members    base-gru   6738.0
2619         population ages 15-64   0.781220    0.909914   0.953894     -1.660444                                       OECD members     xgboost   7179.0
486          population ages 15-64   1.386231    2.536315   1.592581     -6.415781                                       OECD members  simple-rnn   9725.0
1197         population ages 15-64   1.854675    4.270597   2.066542    -11.486545                                       OECD members   base-lstm  10869.0
3332  population ages 65 and above   0.387458    0.233864   0.483595      0.810928                                       OECD members          rf   3790.0
1199  population ages 65 and above   0.816105    0.666928   0.816657      0.460807                                       OECD members   base-lstm   6027.0
488   population ages 65 and above   0.847353    0.719687   0.848344      0.418153                                       OECD members  simple-rnn   6204.0
1910  population ages 65 and above   1.113186    1.247376   1.116860     -0.008469                                       OECD members    base-gru   7395.0
2621  population ages 65 and above   1.731307    4.291745   2.071653     -2.469756                                       OECD members     xgboost  10147.0
3334          population ages 0-14   2.069305    5.441549   2.332713     -0.783405                                               Oman          rf  10118.0
1912          population ages 0-14   2.432824    7.844135   2.800738     -1.570825                                               Oman    base-gru  10847.0
1201          population ages 0-14   2.465556    7.817679   2.796011     -1.562154                                               Oman   base-lstm  10848.0
490           population ages 0-14   2.432823    7.854616   2.802609     -1.574260                                               Oman  simple-rnn  10853.0
2623          population ages 0-14   2.285342   10.425827   3.228905     -2.416945                                               Oman     xgboost  11215.0
1911         population ages 15-64   2.379363    6.868773   2.620834     -1.092772                                               Oman    base-gru  10554.0
489          population ages 15-64   2.386578    7.035372   2.652427     -1.143531                                               Oman  simple-rnn  10588.0
2622         population ages 15-64   2.113275    7.949563   2.819497     -1.422066                                               Oman     xgboost  10692.0
1200         population ages 15-64   2.490006    7.588832   2.754783     -1.312159                                               Oman   base-lstm  10777.0
3333         population ages 15-64   2.488701    9.432005   3.071157     -1.873735                                               Oman          rf  11109.0
3335  population ages 65 and above   0.099178    0.018491   0.135982     -0.826079                                               Oman          rf   2258.0
2624  population ages 65 and above   0.145595    0.029985   0.173161     -1.961128                                               Oman     xgboost   2950.0
1202  population ages 65 and above   1.629223    2.828714   1.681878   -278.349638                                               Oman   base-lstm  10991.0
1913  population ages 65 and above   1.831630    3.627203   1.904522   -357.204485                                               Oman    base-gru  11456.0
491   population ages 65 and above   1.865642    3.803335   1.950214   -374.598356                                               Oman  simple-rnn  11543.0
2626          population ages 0-14   0.056680    0.004434   0.066586      0.991941                                 Other small states     xgboost    287.0
3337          population ages 0-14   0.068835    0.005732   0.075707      0.989582                                 Other small states          rf    367.0
1915          population ages 0-14   1.811614    3.599014   1.897107     -5.542060                                 Other small states    base-gru  10325.0
493           population ages 0-14   1.956262    4.221736   2.054686     -6.674004                                 Other small states  simple-rnn  10707.0
1204          population ages 0-14   1.975526    4.349082   2.085445     -6.905484                                 Other small states   base-lstm  10770.0
2625         population ages 15-64   0.108508    0.016438   0.128211      0.843108                                 Other small states     xgboost   1197.0
3336         population ages 15-64   0.200094    0.050065   0.223752      0.522160                                 Other small states          rf   2413.0
1914         population ages 15-64   0.285476    0.096749   0.311046      0.076583                                 Other small states    base-gru   3373.0
492          population ages 15-64   0.512830    0.329223   0.573780     -2.142244                                 Other small states  simple-rnn   5686.0
1203         population ages 15-64   0.804032    0.813984   0.902210     -6.768996                                 Other small states   base-lstm   7712.0
3338  population ages 65 and above   0.107425    0.016771   0.129503      0.905433                                 Other small states          rf   1077.0
2627  population ages 65 and above   0.357012    0.185952   0.431222     -0.048524                                 Other small states     xgboost   4189.0
1205  population ages 65 and above   1.060096    1.235773   1.111653     -5.968135                                 Other small states   base-lstm   8466.0
494   population ages 65 and above   1.159835    1.480554   1.216780     -7.348374                                 Other small states  simple-rnn   8944.0
1916  population ages 65 and above   1.298433    1.844330   1.358061     -9.399594                                 Other small states    base-gru   9420.0
2629          population ages 0-14   0.602633    0.479604   0.692535     -2.937489                        Pacific island small states     xgboost   6378.0
3340          population ages 0-14   0.988042    1.433478   1.197279    -10.768664                        Pacific island small states          rf   8817.0
1207          population ages 0-14   1.807243    3.778324   1.943791    -30.019545                        Pacific island small states   base-lstm  11021.0
1918          population ages 0-14   1.851890    3.929184   1.982217    -31.258082                        Pacific island small states    base-gru  11122.0
496           population ages 0-14   2.063250    4.888852   2.211075    -39.136825                        Pacific island small states  simple-rnn  11525.0
2628         population ages 15-64   0.193776    0.055990   0.236621     -2.353703                        Pacific island small states     xgboost   3536.0
1917         population ages 15-64   0.926055    1.008958   1.004469    -59.435325                        Pacific island small states    base-gru   8830.0
1206         population ages 15-64   1.012541    1.250307   1.118171    -73.891831                        Pacific island small states   base-lstm   9248.0
3339         population ages 15-64   1.108344    1.825697   1.351184   -108.357003                        Pacific island small states          rf   9917.0
495          population ages 15-64   1.256137    1.837984   1.355723   -109.092947                        Pacific island small states  simple-rnn  10085.0
2630  population ages 65 and above   0.031964    0.001865   0.043186      0.961788                        Pacific island small states     xgboost    275.0
3341  population ages 65 and above   0.067875    0.007473   0.086448      0.846887                        Pacific island small states          rf    826.0
1208  population ages 65 and above   0.796619    0.685162   0.827745    -13.037806                        Pacific island small states   base-lstm   7748.0
497   population ages 65 and above   0.834238    0.758395   0.870859    -14.538239                        Pacific island small states  simple-rnn   7957.0
1919  population ages 65 and above   0.953430    0.980970   0.990439    -19.098427                        Pacific island small states    base-gru   8497.0
2632          population ages 0-14   0.571610    0.431722   0.657055      0.203996                                           Pakistan     xgboost   5254.0
3343          population ages 0-14   0.619848    0.589578   0.767840     -0.087057                                           Pakistan          rf   5885.0
1921          population ages 0-14   1.443795    2.878551   1.696629     -4.307443                                           Pakistan    base-gru   9721.0
1210          population ages 0-14   1.563242    3.381319   1.838836     -5.234442                                           Pakistan   base-lstm  10078.0
499           population ages 0-14   1.702451    3.831743   1.957484     -6.064929                                           Pakistan  simple-rnn  10383.0
1920         population ages 15-64   0.387460    0.184435   0.429459      0.441104                                           Pakistan    base-gru   3976.0
2631         population ages 15-64   0.535688    0.360643   0.600536     -0.092861                                           Pakistan     xgboost   5177.0
498          population ages 15-64   0.571926    0.452492   0.672675     -0.371191                                           Pakistan  simple-rnn   5592.0
1209         population ages 15-64   0.597521    0.502601   0.708943     -0.523035                                           Pakistan   base-lstm   5819.0
3342         population ages 15-64   0.656078    0.676372   0.822418     -1.049615                                           Pakistan          rf   6476.0
3344  population ages 65 and above   0.079743    0.009401   0.096960      0.647955                                           Pakistan          rf   1209.0
2633  population ages 65 and above   0.195346    0.059364   0.243646     -1.222979                                           Pakistan     xgboost   3311.0
1211  population ages 65 and above   0.903607    0.931769   0.965282    -33.891804                                           Pakistan   base-lstm   8575.0
500   population ages 65 and above   0.957758    1.058866   1.029012    -38.651180                                           Pakistan  simple-rnn   8804.0
1922  population ages 65 and above   1.090806    1.361977   1.167038    -50.001746                                           Pakistan    base-gru   9367.0
3346          population ages 0-14   0.081649    0.012760   0.112960      0.985443                                             Panama          rf    631.0
2635          population ages 0-14   0.290131    0.163461   0.404303      0.813519                                             Panama     xgboost   3249.0
1924          population ages 0-14   2.584826    7.326232   2.706701     -7.357979                                             Panama    base-gru  11521.0
1213          population ages 0-14   2.828648    8.883308   2.980488     -9.134338                                             Panama   base-lstm  11824.0
502           population ages 0-14   3.002690    9.909292   3.147903    -10.304811                                             Panama  simple-rnn  12047.0
2634         population ages 15-64   0.515916    0.498610   0.706123     -8.669418                                             Panama     xgboost   6735.0
3345         population ages 15-64   0.656838    0.615194   0.784343    -10.930298                                             Panama          rf   7339.0
1923         population ages 15-64   1.746104    3.246849   1.801901    -61.965344                                             Panama    base-gru  10988.0
501          population ages 15-64   2.399450    6.166152   2.483174   -118.578681                                             Panama  simple-rnn  12093.0
1212         population ages 15-64   2.457320    6.588503   2.566808   -126.769224                                             Panama   base-lstm  12190.0
3347  population ages 65 and above   0.249158    0.101681   0.318875      0.798060                                             Panama          rf   2757.0
2636  population ages 65 and above   0.434173    0.260079   0.509979      0.483481                                             Panama     xgboost   4358.0
1214  population ages 65 and above   0.510922    0.301789   0.549353      0.400645                                             Panama   base-lstm   4686.0
503   population ages 65 and above   0.627332    0.460583   0.678663      0.085278                                             Panama  simple-rnn   5497.0
1925  population ages 65 and above   0.788532    0.705466   0.839920     -0.401060                                             Panama    base-gru   6518.0
2638          population ages 0-14   0.245249    0.113853   0.337420      0.907232                                   Papua New Guinea     xgboost   2592.0
3349          population ages 0-14   0.257857    0.111872   0.334473      0.908846                                   Papua New Guinea          rf   2609.0
1216          population ages 0-14   0.800508    0.727250   0.852790      0.407433                                   Papua New Guinea   base-lstm   6171.0
1927          population ages 0-14   0.971934    1.064936   1.031958      0.132285                                   Papua New Guinea    base-gru   6949.0
505           population ages 0-14   1.132294    1.458231   1.207573     -0.188173                                   Papua New Guinea  simple-rnn   7717.0
1215         population ages 15-64   0.092316    0.017040   0.130536      0.980811                                   Papua New Guinea   base-lstm    786.0
1926         population ages 15-64   0.124437    0.018246   0.135078      0.979452                                   Papua New Guinea    base-gru    930.0
3348         population ages 15-64   0.118046    0.020237   0.142258      0.977210                                   Papua New Guinea          rf    947.0
504          population ages 15-64   0.298973    0.096886   0.311264      0.890891                                   Papua New Guinea  simple-rnn   2668.0
2637         population ages 15-64   0.992434    1.424191   1.193395     -0.603871                                   Papua New Guinea     xgboost   7693.0
3350  population ages 65 and above   0.027070    0.001124   0.033520      0.959458                                   Papua New Guinea          rf    243.0
2639  population ages 65 and above   0.190293    0.059445   0.243813     -1.144934                                   Papua New Guinea     xgboost   3276.0
506   population ages 65 and above   1.044634    1.224442   1.106545    -43.181348                                   Papua New Guinea  simple-rnn   9131.0
1217  population ages 65 and above   1.049984    1.226232   1.107354    -43.245945                                   Papua New Guinea   base-lstm   9137.0
1928  population ages 65 and above   1.148764    1.465898   1.210743    -51.893761                                   Papua New Guinea    base-gru   9538.0
2641          population ages 0-14   0.125531    0.026663   0.163289      0.986794                                           Paraguay     xgboost   1056.0
3352          population ages 0-14   0.463796    0.332328   0.576479      0.835407                                           Paraguay          rf   4220.0
1930          population ages 0-14   1.748315    4.110666   2.027478     -1.035901                                           Paraguay    base-gru   9748.0
1219          population ages 0-14   1.883598    4.752797   2.180091     -1.353931                                           Paraguay   base-lstm  10088.0
508           population ages 0-14   2.011249    5.262280   2.293966     -1.606264                                           Paraguay  simple-rnn  10326.0
2640         population ages 15-64   0.110737    0.014061   0.118581      0.986741                                           Paraguay     xgboost    759.0
3351         population ages 15-64   0.646547    0.718510   0.847650      0.322494                                           Paraguay          rf   5982.0
1929         population ages 15-64   0.697202    0.795098   0.891683      0.250277                                           Paraguay    base-gru   6181.0
507          population ages 15-64   1.042175    1.607689   1.267947     -0.515940                                           Paraguay  simple-rnn   7877.0
1218         population ages 15-64   1.117932    1.880897   1.371458     -0.773556                                           Paraguay   base-lstm   8242.0
3353  population ages 65 and above   0.079740    0.008581   0.092635      0.947863                                           Paraguay          rf    650.0
2642  population ages 65 and above   0.078855    0.008741   0.093494      0.946892                                           Paraguay     xgboost    653.0
1220  population ages 65 and above   0.761196    0.713633   0.844768     -3.335811                                           Paraguay   base-lstm   7197.0
509   population ages 65 and above   0.853038    0.904457   0.951030     -4.495200                                           Paraguay  simple-rnn   7704.0
1931  population ages 65 and above   0.991954    1.189861   1.090808     -6.229225                                           Paraguay    base-gru   8362.0
2644          population ages 0-14   0.470354    0.291935   0.540310      0.785655                                               Peru     xgboost   4232.0
3355          population ages 0-14   0.707027    0.721931   0.849665      0.469943                                               Peru          rf   5974.0
1933          population ages 0-14   2.104477    5.212058   2.282993     -2.826805                                               Peru    base-gru  10586.0
1222          population ages 0-14   2.362236    6.623695   2.573654     -3.863259                                               Peru   base-lstm  11063.0
511           population ages 0-14   2.530692    7.448727   2.729236     -4.469016                                               Peru  simple-rnn  11297.0
2643         population ages 15-64   0.371846    0.173532   0.416572      0.712640                                               Peru     xgboost   3645.0
3354         population ages 15-64   0.606667    0.460819   0.678837      0.236905                                               Peru          rf   5375.0
1932         population ages 15-64   1.545772    2.493952   1.579225     -3.129866                                               Peru    base-gru   9498.0
510          population ages 15-64   2.161854    4.897774   2.213092     -7.110483                                               Peru  simple-rnn  10980.0
1221         population ages 15-64   2.168077    4.978941   2.231354     -7.244891                                               Peru   base-lstm  11010.0
3356  population ages 65 and above   0.155434    0.032644   0.180676      0.810940                                               Peru          rf   1758.0
2645  population ages 65 and above   0.405342    0.309741   0.556543     -0.793897                                               Peru     xgboost   5030.0
1223  population ages 65 and above   0.471672    0.337368   0.580834     -0.953906                                               Peru   base-lstm   5317.0
512   population ages 65 and above   0.499177    0.392024   0.626119     -1.270454                                               Peru  simple-rnn   5605.0
1934  population ages 65 and above   0.556243    0.541193   0.735658     -2.134383                                               Peru    base-gru   6301.0
3358          population ages 0-14   0.063113    0.005075   0.071240      0.997174                                        Philippines          rf    300.0
2647          population ages 0-14   0.358218    0.143206   0.378426      0.920255                                        Philippines     xgboost   3057.0
1936          population ages 0-14   1.719448    3.623519   1.903554     -1.017762                                        Philippines    base-gru   9551.0
1225          population ages 0-14   1.877891    4.329436   2.080730     -1.410853                                        Philippines   base-lstm  10009.0
514           population ages 0-14   2.043059    5.039083   2.244790     -1.806021                                        Philippines  simple-rnn  10353.0
3357         population ages 15-64   0.255885    0.144403   0.380004      0.854202                                        Philippines          rf   2954.0
2646         population ages 15-64   0.353923    0.152104   0.390005      0.846426                                        Philippines     xgboost   3255.0
1935         population ages 15-64   0.489351    0.361620   0.601348      0.634886                                        Philippines    base-gru   4637.0
1224         population ages 15-64   0.925154    1.184968   1.088563     -0.196418                                        Philippines   base-lstm   7198.0
513          population ages 15-64   0.963537    1.189620   1.090697     -0.201115                                        Philippines  simple-rnn   7252.0
2648  population ages 65 and above   0.022940    0.000749   0.027376      0.993955                                        Philippines     xgboost     71.0
3359  population ages 65 and above   0.098486    0.016559   0.128682      0.866447                                        Philippines          rf   1124.0
1226  population ages 65 and above   0.941707    0.999258   0.999629     -7.059295                                        Philippines   base-lstm   8140.0
515   population ages 65 and above   1.018040    1.186709   1.089362     -8.571146                                        Philippines  simple-rnn   8522.0
1937  population ages 65 and above   1.190755    1.591012   1.261353    -11.831964                                        Philippines    base-gru   9231.0
3361          population ages 0-14   0.096398    0.014286   0.119526      0.388488                                             Poland          rf   1610.0
2650          population ages 0-14   0.088004    0.017142   0.130928      0.266253                                             Poland     xgboost   1712.0
1939          population ages 0-14   2.830348    8.423736   2.902367   -359.567613                                             Poland    base-gru  12672.0
1228          population ages 0-14   3.200092   10.862752   3.295869   -463.966710                                             Poland   base-lstm  13089.0
517           population ages 0-14   3.433788   12.406503   3.522287   -530.045041                                             Poland  simple-rnn  13305.0
2649         population ages 15-64   0.292594    0.153743   0.392100      0.903775                                             Poland     xgboost   3011.0
3360         population ages 15-64   0.687830    0.695967   0.834246      0.564406                                             Poland          rf   5819.0
1938         population ages 15-64   2.903084    9.525023   3.086264     -4.961553                                             Poland    base-gru  11681.0
516          population ages 15-64   3.379141   12.837022   3.582879     -7.034478                                             Poland  simple-rnn  12284.0
1227         population ages 15-64   4.038521   18.054667   4.249078    -10.300115                                             Poland   base-lstm  12789.0
2651  population ages 65 and above   0.210762    0.070259   0.265064      0.943291                                             Poland     xgboost   2037.0
1940  population ages 65 and above   0.542784    0.411772   0.641695      0.667642                                             Poland    base-gru   4818.0
518   population ages 65 and above   0.641368    0.555952   0.745622      0.551268                                             Poland  simple-rnn   5461.0
3362  population ages 65 and above   0.672766    0.567819   0.753537      0.541690                                             Poland          rf   5545.0
1229  population ages 65 and above   0.717442    0.700132   0.836739      0.434894                                             Poland   base-lstm   5978.0
2653          population ages 0-14   0.130885    0.029159   0.170761      0.928676                                           Portugal     xgboost   1333.0
3364          population ages 0-14   0.449003    0.367296   0.606050      0.101581                                           Portugal          rf   4926.0
1942          population ages 0-14   2.532699    6.536104   2.556580    -14.987546                                           Portugal    base-gru  11635.0
1231          population ages 0-14   2.915751    8.711748   2.951567    -20.309250                                           Portugal   base-lstm  12123.0
520           population ages 0-14   3.420266   12.019384   3.466898    -28.399848                                           Portugal  simple-rnn  12734.0
2652         population ages 15-64   0.201003    0.049802   0.223164      0.914600                                           Portugal     xgboost   1868.0
3363         population ages 15-64   0.378211    0.193462   0.439843      0.668254                                           Portugal          rf   3796.0
1941         population ages 15-64   1.997264    4.741313   2.177456     -7.130357                                           Portugal    base-gru  10878.0
519          population ages 15-64   2.801413    9.094564   3.015719    -14.595269                                           Portugal  simple-rnn  12018.0
1230         population ages 15-64   3.391129   12.992221   3.604472    -21.278935                                           Portugal   base-lstm  12718.0
3365  population ages 65 and above   0.273698    0.131076   0.362044      0.933349                                           Portugal          rf   2712.0
1943  population ages 65 and above   0.523805    0.377649   0.614531      0.807968                                           Portugal    base-gru   4510.0
521   population ages 65 and above   0.614154    0.587637   0.766575      0.701191                                           Portugal  simple-rnn   5346.0
1232  population ages 65 and above   0.734933    0.847450   0.920570      0.569078                                           Portugal   base-lstm   6098.0
2654  population ages 65 and above   0.753544    1.082855   1.040603      0.449376                                           Portugal     xgboost   6514.0
3367          population ages 0-14   0.202772    0.056017   0.236680      0.445622                          Post-demographic dividend          rf   2582.0
2656          population ages 0-14   0.248598    0.092530   0.304188      0.084271                          Post-demographic dividend     xgboost   3241.0
1945          population ages 0-14   2.477563    6.511852   2.551833    -63.444668                          Post-demographic dividend    base-gru  12039.0
1234          population ages 0-14   2.958156    9.333424   3.055065    -91.368415                          Post-demographic dividend   base-lstm  12612.0
523           population ages 0-14   3.195269   10.872575   3.297359   -106.600657                          Post-demographic dividend  simple-rnn  12877.0
1944         population ages 15-64   0.972495    1.451668   1.204852     -0.722272                          Post-demographic dividend    base-gru   7745.0
2655         population ages 15-64   1.013764    1.618228   1.272096     -0.919881                          Post-demographic dividend     xgboost   7989.0
3366         population ages 15-64   1.411073    2.887919   1.699388     -2.426254                          Post-demographic dividend          rf   9442.0
522          population ages 15-64   1.745336    4.083170   2.020685     -3.844311                          Post-demographic dividend  simple-rnn  10311.0
1233         population ages 15-64   2.248163    6.321707   2.514300     -6.500132                          Post-demographic dividend   base-lstm  11178.0
1235  population ages 65 and above   0.359619    0.173168   0.416135      0.885979                          Post-demographic dividend   base-lstm   3315.0
524   population ages 65 and above   0.379646    0.190660   0.436647      0.874462                          Post-demographic dividend  simple-rnn   3477.0
3368  population ages 65 and above   0.479291    0.401314   0.633493      0.735760                          Post-demographic dividend          rf   4616.0
1946  population ages 65 and above   0.614424    0.419143   0.647412      0.724020                          Post-demographic dividend    base-gru   4923.0
2657  population ages 65 and above   0.711620    1.087907   1.043028      0.283680                          Post-demographic dividend     xgboost   6554.0
2659          population ages 0-14   0.468709    0.389072   0.623756     -0.309352                           Pre-demographic dividend     xgboost   5211.0
1948          population ages 0-14   0.567419    0.354526   0.595421     -0.193096                           Pre-demographic dividend    base-gru   5269.0
3370          population ages 0-14   0.431223    0.458945   0.677455     -0.544499                           Pre-demographic dividend          rf   5404.0
1237          population ages 0-14   0.715215    0.568953   0.754290     -0.914711                           Pre-demographic dividend   base-lstm   6290.0
526           population ages 0-14   0.985918    1.073294   1.035999     -2.611980                           Pre-demographic dividend  simple-rnn   7816.0
1947         population ages 15-64   0.087443    0.010153   0.100761      0.962972                           Pre-demographic dividend    base-gru    666.0
2658         population ages 15-64   0.179255    0.053840   0.232033      0.803640                           Pre-demographic dividend     xgboost   2108.0
1236         population ages 15-64   0.257174    0.075156   0.274147      0.725894                           Pre-demographic dividend   base-lstm   2656.0
3369         population ages 15-64   0.309455    0.128112   0.357927      0.532758                           Pre-demographic dividend          rf   3390.0
525          population ages 15-64   0.479850    0.254809   0.504786      0.070675                           Pre-demographic dividend  simple-rnn   4650.0
2660  population ages 65 and above   0.038919    0.002048   0.045257     -3.037779                           Pre-demographic dividend     xgboost   2152.0
3371  population ages 65 and above   0.043378    0.003855   0.062091     -6.600481                           Pre-demographic dividend          rf   2572.0
527   population ages 65 and above   0.417525    0.197521   0.444434   -388.398284                           Pre-demographic dividend  simple-rnn   6577.0
1238  population ages 65 and above   0.467009    0.241368   0.491292   -474.837544                           Pre-demographic dividend   base-lstm   6861.0
1949  population ages 65 and above   0.506359    0.285690   0.534499   -562.214908                           Pre-demographic dividend    base-gru   7122.0
1951          population ages 0-14   0.421461    0.209139   0.457317      0.953902                                        Puerto Rico    base-gru   3426.0
529           population ages 0-14   0.487948    0.333951   0.577885      0.926391                                        Puerto Rico  simple-rnn   4065.0
2662          population ages 0-14   0.556053    0.438020   0.661831      0.903453                                        Puerto Rico     xgboost   4538.0
1240          population ages 0-14   0.607797    0.444966   0.667058      0.901922                                        Puerto Rico   base-lstm   4653.0
3373          population ages 0-14   1.143365    2.003710   1.415525      0.558348                                        Puerto Rico          rf   7681.0
2661         population ages 15-64   0.800916    0.993206   0.996597     -2.398822                                        Puerto Rico     xgboost   7479.0
3372         population ages 15-64   0.775168    1.056263   1.027747     -2.614608                                        Puerto Rico          rf   7538.0
1950         population ages 15-64   1.241788    2.288496   1.512777     -6.831396                                        Puerto Rico    base-gru   9494.0
528          population ages 15-64   1.865708    4.557144   2.134747    -14.594872                                        Puerto Rico  simple-rnn  11026.0
1239         population ages 15-64   2.315998    6.773387   2.602573    -22.179015                                        Puerto Rico   base-lstm  11748.0
2663  population ages 65 and above   0.811381    1.156260   1.075295      0.834570                                        Puerto Rico     xgboost   6245.0
1952  population ages 65 and above   2.032988    5.884518   2.425803      0.158080                                        Puerto Rico    base-gru   9761.0
1241  population ages 65 and above   2.121284    6.295744   2.509132      0.099244                                        Puerto Rico   base-lstm   9887.0
530   population ages 65 and above   2.253573    6.918443   2.630293      0.010152                                        Puerto Rico  simple-rnn  10076.0
3374  population ages 65 and above   2.637235   10.126588   3.182230     -0.448849                                        Puerto Rico          rf  10785.0
3376          population ages 0-14   0.525477    0.426717   0.653236     -0.101129                                              Qatar          rf   5316.0
2665          population ages 0-14   0.590741    0.509612   0.713871     -0.315037                                              Qatar     xgboost   5754.0
532           population ages 0-14   7.268104   63.775283   7.985943   -163.569902                                              Qatar  simple-rnn  13890.0
1243          population ages 0-14   7.442442   68.093031   8.251850   -174.711701                                              Qatar   base-lstm  13898.0
1954          population ages 0-14   8.228416   81.823693   9.045645   -210.143198                                              Qatar    base-gru  13927.0
2664         population ages 15-64   0.441414    0.557112   0.746399     -0.054876                                              Qatar     xgboost   5471.0
3375         population ages 15-64   0.718995    1.127340   1.061763     -1.134585                                              Qatar          rf   7205.0
531          population ages 15-64  12.056204  160.588000  12.672332   -303.068725                                              Qatar  simple-rnn  13994.0
1242         population ages 15-64  13.039394  190.517776  13.802818   -359.739890                                              Qatar   base-lstm  14029.0
1953         population ages 15-64  13.140027  194.178136  13.934782   -366.670676                                              Qatar    base-gru  14037.0
2666  population ages 65 and above   0.257140    0.082604   0.287409     -3.577209                                              Qatar     xgboost   4154.0
3377  population ages 65 and above   0.696810    0.572467   0.756615    -30.721313                                              Qatar          rf   7667.0
1955  population ages 65 and above   1.864662    3.626249   1.904271   -199.936365                                              Qatar    base-gru  11385.0
1244  population ages 65 and above   1.992691    4.296905   2.072898   -237.098496                                              Qatar   base-lstm  11702.0
533   population ages 65 and above   2.578031    7.222687   2.687506   -399.220806                                              Qatar  simple-rnn  12482.0
3379          population ages 0-14   0.075425    0.011904   0.109104      0.118643                                            Romania          rf   1617.0
2668          population ages 0-14   0.172802    0.041638   0.204054     -2.082924                                            Romania     xgboost   3222.0
1957          population ages 0-14   3.341974   11.563584   3.400527   -855.179115                                            Romania    base-gru  13261.0
1246          population ages 0-14   3.665657   13.979199   3.738877  -1034.033609                                            Romania   base-lstm  13546.0
535           population ages 0-14   4.010245   16.675002   4.083504  -1233.633522                                            Romania  simple-rnn  13735.0
2667         population ages 15-64   0.120175    0.019364   0.139154      0.971990                                            Romania     xgboost    969.0
3378         population ages 15-64   0.326157    0.144544   0.380190      0.790915                                            Romania          rf   3288.0
1956         population ages 15-64   2.118687    5.106754   2.259813     -6.386999                                            Romania    base-gru  10954.0
534          population ages 15-64   2.627415    7.766932   2.786922    -10.234986                                            Romania  simple-rnn  11701.0
1245         population ages 15-64   3.224070   11.437749   3.381974    -15.544880                                            Romania   base-lstm  12416.0
2669  population ages 65 and above   0.169203    0.045696   0.213766      0.911327                                            Romania     xgboost   1717.0
1247  population ages 65 and above   0.191851    0.048654   0.220576      0.905587                                            Romania   base-lstm   1841.0
536   population ages 65 and above   0.171769    0.051214   0.226304      0.900619                                            Romania  simple-rnn   1843.0
3380  population ages 65 and above   0.273402    0.091193   0.301982      0.823039                                            Romania          rf   2679.0
1958  population ages 65 and above   0.307680    0.130633   0.361431      0.746506                                            Romania    base-gru   3193.0
2671          population ages 0-14   0.269290    0.099780   0.315879     -2.303273                                 Russian Federation     xgboost   4133.0
3382          population ages 0-14   0.475960    0.267327   0.517037     -7.850057                                 Russian Federation          rf   6012.0
1960          population ages 0-14   2.791176    7.853845   2.802471   -259.006975                                 Russian Federation    base-gru  12560.0
1249          population ages 0-14   3.118410    9.865363   3.140918   -325.599680                                 Russian Federation   base-lstm  12906.0
538           population ages 0-14   3.172717   10.183550   3.191168   -336.133500                                 Russian Federation  simple-rnn  12955.0
3381         population ages 15-64   0.373886    0.195652   0.442326      0.651923                                 Russian Federation          rf   3821.0
2670         population ages 15-64   0.378159    0.267732   0.517429      0.523688                                 Russian Federation     xgboost   4226.0
1959         population ages 15-64   1.669509    2.964249   1.721700     -4.273576                                 Russian Federation    base-gru   9917.0
537          population ages 15-64   2.119208    4.795150   2.189783     -7.530858                                 Russian Federation  simple-rnn  10962.0
1248         population ages 15-64   2.516742    6.750594   2.598191    -11.009709                                 Russian Federation   base-lstm  11562.0
2672  population ages 65 and above   0.132994    0.026129   0.161645      0.925101                                 Russian Federation     xgboost   1280.0
3383  population ages 65 and above   0.342321    0.188533   0.434203      0.459569                                 Russian Federation          rf   3879.0
539   population ages 65 and above   0.646332    0.465659   0.682392     -0.334819                                 Russian Federation  simple-rnn   5757.0
1250  population ages 65 and above   0.694337    0.531523   0.729056     -0.523616                                 Russian Federation   base-lstm   6047.0
1961  population ages 65 and above   0.817826    0.718011   0.847355     -1.058187                                 Russian Federation    base-gru   6802.0
3385          population ages 0-14   0.057711    0.005386   0.073386      0.995320                                             Rwanda          rf    299.0
2674          population ages 0-14   0.904495    1.139647   1.067542      0.009591                                             Rwanda     xgboost   7036.0
1963          population ages 0-14   1.805010    3.890254   1.972373     -2.380820                                             Rwanda    base-gru  10052.0
541           population ages 0-14   1.901696    4.269483   2.066273     -2.710389                                             Rwanda  simple-rnn  10280.0
1252          population ages 0-14   1.944693    4.537240   2.130080     -2.943083                                             Rwanda   base-lstm  10400.0
3384         population ages 15-64   0.321264    0.149109   0.386146      0.871800                                             Rwanda          rf   3136.0
1962         population ages 15-64   0.486741    0.287928   0.536589      0.752448                                             Rwanda    base-gru   4286.0
540          population ages 15-64   0.559281    0.390796   0.625137      0.664004                                             Rwanda  simple-rnn   4786.0
1251         population ages 15-64   0.775982    0.743565   0.862302      0.360703                                             Rwanda   base-lstm   6186.0
2673         population ages 15-64   1.126033    2.142887   1.463860     -0.842397                                             Rwanda     xgboost   8425.0
2675  population ages 65 and above   0.118933    0.020582   0.143463     -3.655777                                             Rwanda     xgboost   2976.0
3386  population ages 65 and above   0.322500    0.157844   0.397295    -34.705664                                             Rwanda          rf   5718.0
1253  population ages 65 and above   0.949874    1.124390   1.060373   -253.347407                                             Rwanda   base-lstm   9223.0
542   population ages 65 and above   1.002766    1.264207   1.124370   -284.975329                                             Rwanda  simple-rnn   9463.0
1964  population ages 65 and above   1.140999    1.627546   1.275753   -367.166022                                             Rwanda    base-gru   9990.0
544           population ages 0-14   0.246162    0.073084   0.270340      0.268060                                              Samoa  simple-rnn   2959.0
2677          population ages 0-14   0.243031    0.108105   0.328794     -0.082689                                              Samoa     xgboost   3433.0
1255          population ages 0-14   0.277722    0.107697   0.328172     -0.078594                                              Samoa   base-lstm   3522.0
1966          population ages 0-14   0.383855    0.167568   0.409351     -0.678213                                              Samoa    base-gru   4424.0
3388          population ages 0-14   0.867325    1.033321   1.016524     -9.348835                                              Samoa          rf   8190.0
2676         population ages 15-64   0.154650    0.033004   0.181669      0.316853                                              Samoa     xgboost   2228.0
3387         population ages 15-64   0.774399    1.020463   1.010180    -20.122729                                              Samoa          rf   8336.0
543          population ages 15-64   1.018493    1.066737   1.032830    -21.080561                                              Samoa  simple-rnn   8718.0
1254         population ages 15-64   1.114844    1.264523   1.124510    -25.174570                                              Samoa   base-lstm   9114.0
1965         population ages 15-64   1.310910    1.761532   1.327227    -35.462222                                              Samoa    base-gru   9848.0
2678  population ages 65 and above   0.133598    0.021242   0.145746     -0.332898                                              Samoa     xgboost   2260.0
3389  population ages 65 and above   0.157590    0.052582   0.229307     -2.299424                                              Samoa          rf   3362.0
545   population ages 65 and above   0.562264    0.339430   0.582606    -20.298735                                              Samoa  simple-rnn   6733.0
1256  population ages 65 and above   0.564877    0.339762   0.582891    -20.319566                                              Samoa   base-lstm   6747.0
1967  population ages 65 and above   0.687442    0.505593   0.711051    -30.725222                                              Samoa    base-gru   7512.0
3391          population ages 0-14   0.518994    0.347964   0.589885      0.329150                              Sao Tome and Principe          rf   4883.0
2680          population ages 0-14   0.580313    0.399298   0.631900      0.230182                              Sao Tome and Principe     xgboost   5188.0
1969          population ages 0-14   1.790359    3.438698   1.854373     -5.629562                              Sao Tome and Principe    base-gru  10283.0
1258          population ages 0-14   1.905869    3.911696   1.977801     -6.541470                              Sao Tome and Principe   base-lstm  10586.0
547           population ages 0-14   2.279675    5.580320   2.362270     -9.758457                              Sao Tome and Principe  simple-rnn  11249.0
1968         population ages 15-64   0.689643    0.604139   0.777263     -0.514383                              Sao Tome and Principe    base-gru   6207.0
2679         population ages 15-64   0.762374    0.709379   0.842246     -0.778187                              Sao Tome and Principe     xgboost   6615.0
3390         population ages 15-64   0.751777    0.732716   0.855988     -0.836686                              Sao Tome and Principe          rf   6659.0
1257         population ages 15-64   0.940644    1.070784   1.034787     -1.684114                              Sao Tome and Principe   base-lstm   7580.0
546          population ages 15-64   1.208811    1.688505   1.299425     -3.232544                              Sao Tome and Principe  simple-rnn   8776.0
2681  population ages 65 and above   0.085218    0.009572   0.097838      0.015483                              Sao Tome and Principe     xgboost   1646.0
3392  population ages 65 and above   0.089088    0.013279   0.115233     -0.365726                              Sao Tome and Principe          rf   1937.0
1259  population ages 65 and above   0.826537    0.698388   0.835696    -70.830403                              Sao Tome and Principe   base-lstm   8318.0
548   population ages 65 and above   0.844535    0.734523   0.857043    -74.547024                              Sao Tome and Principe  simple-rnn   8419.0
1970  population ages 65 and above   0.944689    0.916996   0.957599    -93.314657                              Sao Tome and Principe    base-gru   8811.0
2683          population ages 0-14   0.298649    0.126048   0.355032      0.929683                                       Saudi Arabia     xgboost   2765.0
3394          population ages 0-14   0.452364    0.397760   0.630682      0.778105                                       Saudi Arabia          rf   4494.0
1261          population ages 0-14   2.168006    7.257830   2.694036     -3.048867                                       Saudi Arabia   base-lstm  10949.0
550           population ages 0-14   2.342323    8.311105   2.882899     -3.636449                                       Saudi Arabia  simple-rnn  11237.0
1972          population ages 0-14   2.686469   10.875633   3.297822     -5.067101                                       Saudi Arabia    base-gru  11763.0
3393         population ages 15-64   0.455009    0.354322   0.595250      0.774744                                       Saudi Arabia          rf   4397.0
2682         population ages 15-64   0.446943    0.405072   0.636452      0.742481                                       Saudi Arabia     xgboost   4555.0
549          population ages 15-64   1.823722    4.902658   2.214195     -2.116806                                       Saudi Arabia  simple-rnn  10268.0
1260         population ages 15-64   1.891399    5.330733   2.308838     -2.388949                                       Saudi Arabia   base-lstm  10433.0
1971         population ages 15-64   2.249743    7.534018   2.744817     -3.789661                                       Saudi Arabia    base-gru  11143.0
3395  population ages 65 and above   0.130646    0.028135   0.167736     -1.067276                                       Saudi Arabia          rf   2628.0
2684  population ages 65 and above   0.276465    0.111222   0.333500     -7.172214                                       Saudi Arabia     xgboost   4735.0
1262  population ages 65 and above   1.590397    2.764926   1.662807   -202.156489                                       Saudi Arabia   base-lstm  10893.0
1973  population ages 65 and above   1.672307    3.046854   1.745524   -222.871544                                       Saudi Arabia    base-gru  11079.0
551   population ages 65 and above   1.786094    3.545840   1.883040   -259.535160                                       Saudi Arabia  simple-rnn  11355.0
3397          population ages 0-14   0.368255    0.319532   0.565272     -0.318190                                            Senegal          rf   4821.0
2686          population ages 0-14   0.511755    0.354666   0.595538     -0.463129                                            Senegal     xgboost   5283.0
1975          population ages 0-14   1.452914    2.268908   1.506290     -8.360089                                            Senegal    base-gru   9745.0
1264          population ages 0-14   1.613978    2.833777   1.683383    -10.690381                                            Senegal   base-lstm  10190.0
553           population ages 0-14   1.873015    3.820936   1.954722    -14.762780                                            Senegal  simple-rnn  10828.0
2685         population ages 15-64   0.102206    0.014749   0.121446      0.943275                                            Senegal     xgboost    914.0
3396         population ages 15-64   0.386034    0.184580   0.429628      0.290102                                            Senegal          rf   4064.0
1974         population ages 15-64   0.496439    0.269920   0.519538     -0.038117                                            Senegal    base-gru   4810.0
1263         population ages 15-64   0.767797    0.633029   0.795631     -1.434641                                            Senegal   base-lstm   6665.0
552          population ages 15-64   0.959575    0.988680   0.994324     -2.802480                                            Senegal  simple-rnn   7728.0
3398  population ages 65 and above   0.030311    0.001426   0.037765     -0.231425                                            Senegal          rf   1364.0
2687  population ages 65 and above   0.263615    0.102251   0.319767    -87.287497                                            Senegal     xgboost   5395.0
554   population ages 65 and above   0.758379    0.661245   0.813170   -569.946603                                            Senegal  simple-rnn   8466.0
1265  population ages 65 and above   0.776782    0.683939   0.827006   -589.541498                                            Senegal   base-lstm   8560.0
1976  population ages 65 and above   0.886740    0.890691   0.943764   -768.059710                                            Senegal    base-gru   9022.0
3400          population ages 0-14   0.061972    0.007755   0.088061     -0.237767                                             Serbia          rf   1649.0
2689          population ages 0-14   0.105479    0.014051   0.118536     -1.242714                                             Serbia     xgboost   2314.0
1978          population ages 0-14   2.223397    5.053620   2.248026   -805.637768                                             Serbia    base-gru  12121.0
1267          population ages 0-14   2.501656    6.421324   2.534033  -1023.945077                                             Serbia   base-lstm  12438.0
556           population ages 0-14   2.936057    8.795574   2.965733  -1402.912879                                             Serbia  simple-rnn  12873.0
2688         population ages 15-64   0.018416    0.000669   0.025866      0.990077                                             Serbia     xgboost     80.0
3399         population ages 15-64   0.129869    0.032216   0.179489      0.522171                                             Serbia          rf   1981.0
1977         population ages 15-64   1.726920    3.044497   1.744849    -44.155675                                             Serbia    base-gru  10814.0
555          population ages 15-64   2.314369    5.465495   2.337840    -80.063670                                             Serbia  simple-rnn  11877.0
1266         population ages 15-64   2.862838    8.329971   2.886169   -122.549278                                             Serbia   base-lstm  12520.0
3401  population ages 65 and above   0.127307    0.026296   0.162159      0.383896                                             Serbia          rf   1960.0
2690  population ages 65 and above   0.159010    0.046490   0.215615     -0.089256                                             Serbia     xgboost   2606.0
1979  population ages 65 and above   0.493042    0.251624   0.501621     -4.895542                                             Serbia    base-gru   5759.0
557   population ages 65 and above   0.648925    0.429847   0.655627     -9.071313                                             Serbia  simple-rnn   6817.0
1268  population ages 65 and above   0.692108    0.486590   0.697560    -10.400795                                             Serbia   base-lstm   7086.0
1981          population ages 0-14   0.066199    0.005691   0.075441      0.996671                                       Sierra Leone    base-gru    324.0
1270          population ages 0-14   0.205675    0.046334   0.215253      0.972901                                       Sierra Leone   base-lstm   1648.0
559           population ages 0-14   0.324597    0.116797   0.341756      0.931689                                       Sierra Leone  simple-rnn   2753.0
2692          population ages 0-14   0.354107    0.234954   0.484720      0.862583                                       Sierra Leone     xgboost   3604.0
3403          population ages 0-14   0.541638    0.527437   0.726248      0.691519                                       Sierra Leone          rf   5084.0
558          population ages 15-64   0.553000    0.356414   0.597005      0.800682                                       Sierra Leone  simple-rnn   4547.0
1269         population ages 15-64   0.601751    0.436130   0.660401      0.756102                                       Sierra Leone   base-lstm   4898.0
3402         population ages 15-64   0.802804    0.986844   0.993400      0.448125                                       Sierra Leone          rf   6477.0
1980         population ages 15-64   0.898887    0.943319   0.971246      0.472466                                       Sierra Leone    base-gru   6549.0
2691         population ages 15-64   0.984167    1.470218   1.212525      0.177807                                       Sierra Leone     xgboost   7372.0
3404  population ages 65 and above   0.036746    0.001867   0.043210     -0.679224                                       Sierra Leone          rf   1578.0
2693  population ages 65 and above   0.079056    0.008988   0.094808     -7.083836                                       Sierra Leone     xgboost   2847.0
560   population ages 65 and above   0.607442    0.457248   0.676201   -410.228102                                       Sierra Leone  simple-rnn   7735.0
1271  population ages 65 and above   0.655220    0.517004   0.719030   -463.970238                                       Sierra Leone   base-lstm   7984.0
1982  population ages 65 and above   0.724737    0.630937   0.794316   -566.436582                                       Sierra Leone    base-gru   8357.0
3406          population ages 0-14   0.181258    0.050876   0.225558     -0.701260                                    Slovak Republic          rf   2984.0
2695          population ages 0-14   0.264336    0.125168   0.353791     -3.185526                                    Slovak Republic     xgboost   4447.0
1984          population ages 0-14   2.628257    7.197647   2.682843   -239.683314                                    Slovak Republic    base-gru  12405.0
1273          population ages 0-14   2.972043    9.275526   3.045575   -309.165850                                    Slovak Republic   base-lstm  12782.0
562           population ages 0-14   3.131713   10.240731   3.200114   -341.441512                                    Slovak Republic  simple-rnn  12945.0
2694         population ages 15-64   0.226576    0.075012   0.273883      0.929821                                    Slovak Republic     xgboost   2152.0
3405         population ages 15-64   0.501457    0.349990   0.591599      0.672560                                    Slovak Republic          rf   4582.0
1983         population ages 15-64   2.629954    7.469427   2.733025     -5.988178                                    Slovak Republic    base-gru  11461.0
561          population ages 15-64   3.046941   10.008504   3.163622     -8.363665                                    Slovak Republic  simple-rnn  11992.0
1272         population ages 15-64   3.657594   14.291860   3.780458    -12.371049                                    Slovak Republic   base-lstm  12668.0
2696  population ages 65 and above   0.203249    0.076026   0.275728      0.898060                                    Slovak Republic     xgboost   2189.0
1985  population ages 65 and above   0.278821    0.097373   0.312047      0.869436                                    Slovak Republic    base-gru   2658.0
563   population ages 65 and above   0.328879    0.144222   0.379766      0.806617                                    Slovak Republic  simple-rnn   3253.0
1274  population ages 65 and above   0.392836    0.205667   0.453506      0.724227                                    Slovak Republic   base-lstm   3829.0
3407  population ages 65 and above   0.538402    0.362484   0.602066      0.513957                                    Slovak Republic          rf   4836.0
2698          population ages 0-14   0.078991    0.008510   0.092249     -0.089823                                           Slovenia     xgboost   1659.0
3409          population ages 0-14   0.133314    0.036527   0.191121     -3.677923                                           Slovenia          rf   3329.0
1987          population ages 0-14   2.306589    5.462110   2.337116   -698.514170                                           Slovenia    base-gru  12194.0
1276          population ages 0-14   2.654636    7.284283   2.698941   -931.873834                                           Slovenia   base-lstm  12619.0
565           population ages 0-14   2.961618    9.034042   3.005668  -1155.959628                                           Slovenia  simple-rnn  12909.0
2697         population ages 15-64   0.028401    0.001229   0.035063      0.998225                                           Slovenia     xgboost     87.0
3408         population ages 15-64   0.295322    0.111784   0.334341      0.838586                                           Slovenia          rf   2852.0
1986         population ages 15-64   2.211316    5.308424   2.304002     -6.665275                                           Slovenia    base-gru  11032.0
564          population ages 15-64   2.809336    8.511818   2.917502    -11.290922                                           Slovenia  simple-rnn  11855.0
1275         population ages 15-64   3.311395   11.675823   3.416990    -15.859691                                           Slovenia   base-lstm  12460.0
2699  population ages 65 and above   0.197738    0.104431   0.323157      0.814190                                           Slovenia     xgboost   2574.0
3410  population ages 65 and above   0.437702    0.229200   0.478748      0.592192                                           Slovenia          rf   4134.0
1988  population ages 65 and above   0.517086    0.372612   0.610420      0.337024                                           Slovenia    base-gru   4934.0
566   population ages 65 and above   0.719248    0.626814   0.791716     -0.115268                                           Slovenia  simple-rnn   6121.0
1277  population ages 65 and above   0.771234    0.716745   0.846608     -0.275280                                           Slovenia   base-lstm   6458.0
3412          population ages 0-14   0.068582    0.009345   0.096670      0.982808                                       Small states          rf    504.0
2701          population ages 0-14   0.418161    0.285447   0.534273      0.474861                                       Small states     xgboost   4405.0
1990          population ages 0-14   1.751396    3.393991   1.842279     -5.243947                                       Small states    base-gru  10213.0
568           population ages 0-14   1.888289    3.962576   1.990622     -6.289977                                       Small states  simple-rnn  10575.0
1279          population ages 0-14   1.879070    3.964563   1.991121     -6.293632                                       Small states   base-lstm  10576.0
1989         population ages 15-64   0.382267    0.183275   0.428106     -0.597032                                       Small states    base-gru   4462.0
3411         population ages 15-64   0.419075    0.262235   0.512089     -1.285083                                       Small states          rf   5093.0
2700         population ages 15-64   0.478599    0.434325   0.659033     -2.784651                                       Small states     xgboost   6001.0
567          population ages 15-64   0.606256    0.467977   0.684089     -3.077893                                       Small states  simple-rnn   6376.0
1278         population ages 15-64   0.838618    0.883445   0.939917     -6.698224                                       Small states   base-lstm   7840.0
3413  population ages 65 and above   0.165313    0.036204   0.190275      0.775189                                       Small states          rf   1911.0
2702  population ages 65 and above   0.489171    0.377083   0.614071     -1.341490                                       Small states     xgboost   5574.0
1280  population ages 65 and above   0.958561    1.003641   1.001819     -5.232096                                       Small states   base-lstm   8035.0
569   population ages 65 and above   1.056680    1.222781   1.105794     -6.592844                                       Small states  simple-rnn   8498.0
1991  population ages 65 and above   1.184488    1.525552   1.235132     -8.472895                                       Small states    base-gru   9064.0
2704          population ages 0-14   0.132097    0.022173   0.148906      0.898458                                    Solomon Islands     xgboost   1303.0
3415          population ages 0-14   0.348621    0.164995   0.406195      0.244400                                    Solomon Islands          rf   3907.0
1282          population ages 0-14   0.553325    0.311876   0.558458     -0.428248                                    Solomon Islands   base-lstm   5214.0
1993          population ages 0-14   0.565070    0.325108   0.570182     -0.488844                                    Solomon Islands    base-gru   5298.0
571           population ages 0-14   0.850380    0.751109   0.866666     -2.439738                                    Solomon Islands  simple-rnn   7237.0
2703         population ages 15-64   0.130896    0.020546   0.143340      0.903073                                    Solomon Islands     xgboost   1251.0
1281         population ages 15-64   0.137346    0.031525   0.177553      0.851282                                    Solomon Islands   base-lstm   1575.0
570          population ages 15-64   0.190662    0.038609   0.196491      0.817866                                    Solomon Islands  simple-rnn   1929.0
3414         population ages 15-64   0.186422    0.043788   0.209256      0.793432                                    Solomon Islands          rf   2018.0
1992         population ages 15-64   0.167879    0.049203   0.221818      0.767887                                    Solomon Islands    base-gru   2056.0
2705  population ages 65 and above   0.005557    0.000046   0.006763      0.247055                                    Solomon Islands     xgboost   1033.0
3416  population ages 65 and above   0.029345    0.001584   0.039794    -25.070103                                    Solomon Islands          rf   2958.0
572   population ages 65 and above   0.640896    0.444505   0.666712  -7316.688640                                    Solomon Islands  simple-rnn   7914.0
1283  population ages 65 and above   0.672335    0.483717   0.695498  -7962.212376                                    Solomon Islands   base-lstm   8077.0
1994  population ages 65 and above   0.771742    0.635925   0.797449 -10467.935009                                    Solomon Islands    base-gru   8557.0
3418          population ages 0-14   0.735700    0.648171   0.805091     -4.549493                                            Somalia          rf   7164.0
574           population ages 0-14   0.852135    0.787991   0.887689     -5.746599                                            Somalia  simple-rnn   7657.0
2707          population ages 0-14   0.838269    0.804309   0.896833     -5.886309                                            Somalia     xgboost   7677.0
1285          population ages 0-14   1.119256    1.355381   1.164208    -10.604461                                            Somalia   base-lstm   8889.0
1996          population ages 0-14   1.386490    2.095205   1.447482    -16.938673                                            Somalia    base-gru   9849.0
3417         population ages 15-64   0.459527    0.276105   0.525457     -2.323543                                            Somalia          rf   5465.0
2706         population ages 15-64   0.567212    0.391764   0.625911     -3.715754                                            Somalia     xgboost   6210.0
573          population ages 15-64   0.706597    0.548141   0.740366     -5.598101                                            Somalia  simple-rnn   6988.0
1284         population ages 15-64   0.914218    0.911124   0.954528     -9.967405                                            Somalia   base-lstm   8124.0
1995         population ages 15-64   1.199845    1.553768   1.246502    -17.703054                                            Somalia    base-gru   9348.0
3419  population ages 65 and above   0.057113    0.005904   0.076837     -0.527303                                            Somalia          rf   1696.0
1286  population ages 65 and above   0.062410    0.005989   0.077391     -0.549416                                            Somalia   base-lstm   1726.0
1997  population ages 65 and above   0.087271    0.014777   0.121559     -2.822597                                            Somalia    base-gru   2600.0
575   population ages 65 and above   0.135785    0.026472   0.162703     -5.848239                                            Somalia  simple-rnn   3348.0
2708  population ages 65 and above   0.128690    0.027757   0.166605     -6.180621                                            Somalia     xgboost   3369.0
2710          population ages 0-14   0.199863    0.066929   0.258707     -0.698084                                       South Africa     xgboost   3199.0
3421          population ages 0-14   1.439996    3.283214   1.811964    -82.299596                                       South Africa          rf  10841.0
1288          population ages 0-14   4.705779   27.653948   5.258702   -700.618161                                       South Africa   base-lstm  13936.0
577           population ages 0-14   4.905716   29.619928   5.442419   -750.497749                                       South Africa  simple-rnn  13979.0
1999          population ages 0-14   5.048436   31.401339   5.603690   -795.694569                                       South Africa    base-gru  14003.0
2709         population ages 15-64   0.564033    0.533065   0.730113     -1.548433                                       South Africa     xgboost   6147.0
3420         population ages 15-64   0.952863    1.395236   1.181201     -5.670230                                       South Africa          rf   8488.0
576          population ages 15-64   4.079828   20.285783   4.503974    -95.980591                                       South Africa  simple-rnn  13494.0
1998         population ages 15-64   4.197417   21.533383   4.640408   -101.945015                                       South Africa    base-gru  13554.0
1287         population ages 15-64   4.289900   22.854977   4.780688   -108.263181                                       South Africa   base-lstm  13596.0
2711  population ages 65 and above   0.090542    0.016729   0.129339      0.901552                                       South Africa     xgboost   1027.0
3422  population ages 65 and above   0.153094    0.035506   0.188431      0.791045                                       South Africa          rf   1836.0
1289  population ages 65 and above   0.969179    1.131968   1.063940     -5.661664                                       South Africa   base-lstm   8225.0
578   population ages 65 and above   1.217058    1.804321   1.343250     -9.618478                                       South Africa  simple-rnn   9328.0
2000  population ages 65 and above   1.248111    1.853195   1.361321     -9.906104                                       South Africa    base-gru   9397.0
3424          population ages 0-14   0.262567    0.127486   0.357052      0.955844                                         South Asia          rf   2606.0
2713          population ages 0-14   0.482748    0.451912   0.672244      0.843476                                         South Asia     xgboost   4539.0
2002          population ages 0-14   3.077713   10.696546   3.270557     -2.704839                                         South Asia    base-gru  11559.0
1291          population ages 0-14   3.218711   11.765001   3.430015     -3.074908                                         South Asia   base-lstm  11778.0
580           population ages 0-14   3.291575   12.176789   3.489526     -3.217534                                         South Asia  simple-rnn  11861.0
3423         population ages 15-64   0.158915    0.037789   0.194394      0.972943                                         South Asia          rf   1426.0
2712         population ages 15-64   0.208571    0.082447   0.287136      0.940967                                         South Asia     xgboost   2134.0
2001         population ages 15-64   1.295410    2.007056   1.416706     -0.437062                                         South Asia    base-gru   8381.0
579          population ages 15-64   1.580126    2.953469   1.718566     -1.114699                                         South Asia  simple-rnn   9270.0
1290         population ages 15-64   1.889084    4.356387   2.087196     -2.119196                                         South Asia   base-lstm  10191.0
2714  population ages 65 and above   0.159802    0.046740   0.216194      0.830723                                         South Asia     xgboost   1892.0
3425  population ages 65 and above   0.235130    0.105119   0.324220      0.619293                                         South Asia          rf   2945.0
1292  population ages 65 and above   1.249233    1.715397   1.309732     -5.212626                                         South Asia   base-lstm   9044.0
581   population ages 65 and above   1.462786    2.384606   1.544217     -7.636293                                         South Asia  simple-rnn   9784.0
2003  population ages 65 and above   1.631474    2.933000   1.712600     -9.622402                                         South Asia    base-gru  10214.0
3427          population ages 0-14   0.262567    0.127486   0.357052      0.955844                            South Asia (IDA & IBRD)          rf   2606.0
2716          population ages 0-14   0.482748    0.451912   0.672244      0.843476                            South Asia (IDA & IBRD)     xgboost   4539.0
2005          population ages 0-14   3.077713   10.696546   3.270557     -2.704839                            South Asia (IDA & IBRD)    base-gru  11559.0
1294          population ages 0-14   3.218711   11.765001   3.430015     -3.074908                            South Asia (IDA & IBRD)   base-lstm  11778.0
583           population ages 0-14   3.291575   12.176789   3.489526     -3.217534                            South Asia (IDA & IBRD)  simple-rnn  11861.0
3426         population ages 15-64   0.158915    0.037789   0.194394      0.972943                            South Asia (IDA & IBRD)          rf   1430.0
2715         population ages 15-64   0.208571    0.082447   0.287136      0.940967                            South Asia (IDA & IBRD)     xgboost   2130.0
2004         population ages 15-64   1.295410    2.007056   1.416706     -0.437062                            South Asia (IDA & IBRD)    base-gru   8385.0
582          population ages 15-64   1.580126    2.953469   1.718566     -1.114699                            South Asia (IDA & IBRD)  simple-rnn   9274.0
1293         population ages 15-64   1.889084    4.356387   2.087196     -2.119196                            South Asia (IDA & IBRD)   base-lstm  10195.0
2717  population ages 65 and above   0.159802    0.046740   0.216194      0.830723                            South Asia (IDA & IBRD)     xgboost   1892.0
3428  population ages 65 and above   0.235130    0.105119   0.324220      0.619293                            South Asia (IDA & IBRD)          rf   2945.0
1295  population ages 65 and above   1.249233    1.715397   1.309732     -5.212626                            South Asia (IDA & IBRD)   base-lstm   9044.0
584   population ages 65 and above   1.462786    2.384606   1.544217     -7.636293                            South Asia (IDA & IBRD)  simple-rnn   9784.0
2006  population ages 65 and above   1.631474    2.933000   1.712600     -9.622402                            South Asia (IDA & IBRD)    base-gru  10214.0
3430          population ages 0-14   0.181998    0.105245   0.324415     -0.026223                                              Spain          rf   3197.0
2719          population ages 0-14   0.667182    0.522883   0.723107     -4.098542                                              Spain     xgboost   6760.0
2008          population ages 0-14   3.653721   13.620023   3.690532   -131.806404                                              Spain    base-gru  13261.0
1297          population ages 0-14   4.049225   16.812724   4.100332   -162.937854                                              Spain   base-lstm  13504.0
586           population ages 0-14   4.560895   21.328996   4.618333   -206.975209                                              Spain  simple-rnn  13673.0
2718         population ages 15-64   0.338113    0.289765   0.538298      0.632266                                              Spain     xgboost   4107.0
3429         population ages 15-64   0.477509    0.305834   0.553023      0.611873                                              Spain          rf   4464.0
2007         population ages 15-64   1.253532    1.934971   1.391032     -1.455624                                              Spain    base-gru   8632.0
585          population ages 15-64   2.104292    5.278326   2.297461     -5.698595                                              Spain  simple-rnn  10921.0
1296         population ages 15-64   2.736443    8.584742   2.929973     -9.894686                                              Spain   base-lstm  11802.0
3431  population ages 65 and above   0.186618    0.051091   0.226033      0.952733                                              Spain          rf   1736.0
2720  population ages 65 and above   0.327937    0.170298   0.412672      0.842448                                              Spain     xgboost   3313.0
1298  population ages 65 and above   1.083437    1.238343   1.112809     -0.145657                                              Spain   base-lstm   7415.0
587   population ages 65 and above   1.234737    1.559437   1.248774     -0.442719                                              Spain  simple-rnn   8034.0
2009  population ages 65 and above   1.507363    2.306648   1.518765     -1.134003                                              Spain    base-gru   8947.0
3433          population ages 0-14   0.299370    0.119413   0.345562      0.778813                                          Sri Lanka          rf   3047.0
2722          population ages 0-14   0.443313    0.329077   0.573652      0.390456                                          Sri Lanka     xgboost   4663.0
2011          population ages 0-14   4.104653   18.175111   4.263228    -32.665533                                          Sri Lanka    base-gru  13206.0
589           population ages 0-14   4.346399   20.440690   4.521138    -36.862035                                          Sri Lanka  simple-rnn  13318.0
1300          population ages 0-14   4.391402   20.947295   4.576821    -37.800413                                          Sri Lanka   base-lstm  13336.0
2721         population ages 15-64   0.442880    0.291117   0.539553     -0.415401                                          Sri Lanka     xgboost   4949.0
3432         population ages 15-64   0.572881    0.481680   0.694032     -1.341914                                          Sri Lanka          rf   6002.0
2010         population ages 15-64   2.735889    8.603122   2.933108    -40.828106                                          Sri Lanka    base-gru  12278.0
588          population ages 15-64   2.978425   10.113465   3.180168    -48.171350                                          Sri Lanka  simple-rnn  12555.0
1299         population ages 15-64   3.518878   14.268686   3.777391    -68.373904                                          Sri Lanka   base-lstm  13145.0
1301  population ages 65 and above   1.017014    1.035465   1.017578      0.223540                                          Sri Lanka   base-lstm   6932.0
590   population ages 65 and above   1.226353    1.510395   1.228981     -0.132595                                          Sri Lanka  simple-rnn   7845.0
3434  population ages 65 and above   1.050516    1.841975   1.357194     -0.381235                                          Sri Lanka          rf   7997.0
2012  population ages 65 and above   1.322613    1.756015   1.325147     -0.316777                                          Sri Lanka    base-gru   8200.0
2723  population ages 65 and above   1.329471    2.771347   1.664736     -1.078140                                          Sri Lanka     xgboost   9002.0
2725          population ages 0-14   0.245327    0.091760   0.302919      0.866233                                St. Kitts and Nevis     xgboost   2522.0
3436          population ages 0-14   0.274934    0.150464   0.387897      0.780654                                St. Kitts and Nevis          rf   3200.0
2014          population ages 0-14   2.103423    4.859786   2.204492     -6.084590                                St. Kitts and Nevis    base-gru  10871.0
592           population ages 0-14   2.266448    5.653556   2.377721     -7.241747                                St. Kitts and Nevis  simple-rnn  11139.0
1303          population ages 0-14   2.303374    5.888662   2.426657     -7.584483                                St. Kitts and Nevis   base-lstm  11208.0
3435         population ages 15-64   0.428181    0.354989   0.595810     -2.193073                                St. Kitts and Nevis          rf   5614.0
2013         population ages 15-64   0.521613    0.441293   0.664299     -2.969362                                St. Kitts and Nevis    base-gru   6129.0
591          population ages 15-64   0.559423    0.516373   0.718591     -3.644692                                St. Kitts and Nevis  simple-rnn   6501.0
2724         population ages 15-64   0.568851    0.817255   0.904022     -6.351084                                St. Kitts and Nevis     xgboost   7312.0
1302         population ages 15-64   0.891433    1.443412   1.201421    -11.983265                                St. Kitts and Nevis   base-lstm   8751.0
2726  population ages 65 and above   0.509230    0.519982   0.721098      0.071127                                St. Kitts and Nevis     xgboost   5429.0
3437  population ages 65 and above   0.774836    1.151350   1.073010     -1.056719                                St. Kitts and Nevis          rf   7283.0
1304  population ages 65 and above   1.986983    3.969491   1.992358     -6.090919                                St. Kitts and Nevis   base-lstm  10626.0
593   population ages 65 and above   2.126168    4.555033   2.134252     -7.136905                                St. Kitts and Nevis  simple-rnn  10891.0
2015  population ages 65 and above   2.146225    4.629417   2.151608     -7.269780                                St. Kitts and Nevis    base-gru  10932.0
2728          population ages 0-14   0.066437    0.009057   0.095169      0.993104                                          St. Lucia     xgboost    441.0
3439          population ages 0-14   0.309926    0.119729   0.346019      0.908838                                          St. Lucia          rf   2798.0
2017          population ages 0-14   0.918968    0.996617   0.998307      0.241173                                          St. Lucia    base-gru   6764.0
595           population ages 0-14   1.017097    1.210661   1.100301      0.078198                                          St. Lucia  simple-rnn   7204.0
1306          population ages 0-14   1.091967    1.415970   1.189945     -0.078124                                          St. Lucia   base-lstm   7580.0
2727         population ages 15-64   0.126417    0.027616   0.166180      0.960047                                          St. Lucia     xgboost   1193.0
1305         population ages 15-64   0.357389    0.133756   0.365727      0.806490                                          St. Lucia   base-lstm   3233.0
3438         population ages 15-64   0.468575    0.295379   0.543488      0.572666                                          St. Lucia          rf   4455.0
2016         population ages 15-64   0.794321    0.648666   0.805398      0.061554                                          St. Lucia    base-gru   6189.0
594          population ages 15-64   0.888314    0.803729   0.896509     -0.162780                                          St. Lucia  simple-rnn   6680.0
3440  population ages 65 and above   0.157784    0.046201   0.214945      0.581879                                          St. Lucia          rf   2174.0
2729  population ages 65 and above   0.323522    0.161618   0.402017     -0.462639                                          St. Lucia     xgboost   4182.0
1307  population ages 65 and above   2.036458    4.480160   2.116639    -39.545341                                          St. Lucia   base-lstm  11437.0
596   population ages 65 and above   2.135103    4.939547   2.222509    -43.702782                                          St. Lucia  simple-rnn  11595.0
2018  population ages 65 and above   2.153774    4.980921   2.231798    -44.077216                                          St. Lucia    base-gru  11618.0
3442          population ages 0-14   0.051041    0.006791   0.082405      0.995978                     St. Vincent and the Grenadines          rf    329.0
2731          population ages 0-14   0.208931    0.070005   0.264585      0.958541                     St. Vincent and the Grenadines     xgboost   1973.0
2020          population ages 0-14   1.053280    1.480326   1.216687      0.123304                     St. Vincent and the Grenadines    base-gru   7503.0
598           population ages 0-14   1.214946    1.921913   1.386331     -0.138218                     St. Vincent and the Grenadines  simple-rnn   8132.0
1309          population ages 0-14   1.258143    2.104848   1.450809     -0.246558                     St. Vincent and the Grenadines   base-lstm   8324.0
597          population ages 15-64   0.249572    0.077409   0.278225      0.710590                     St. Vincent and the Grenadines  simple-rnn   2669.0
2019         population ages 15-64   0.280473    0.098907   0.314496      0.630215                     St. Vincent and the Grenadines    base-gru   3023.0
1308         population ages 15-64   0.389127    0.257604   0.507547      0.036897                     St. Vincent and the Grenadines   base-lstm   4520.0
2730         population ages 15-64   0.589878    0.554466   0.744625     -1.072981                     St. Vincent and the Grenadines     xgboost   6119.0
3441         population ages 15-64   0.887837    1.184345   1.088276     -3.427910                     St. Vincent and the Grenadines          rf   7962.0
2732  population ages 65 and above   0.293551    0.136368   0.369280      0.792778                     St. Vincent and the Grenadines     xgboost   3146.0
3443  population ages 65 and above   0.405895    0.240109   0.490009      0.635135                     St. Vincent and the Grenadines          rf   4061.0
1310  population ages 65 and above   0.800952    0.740090   0.860285     -0.124628                     St. Vincent and the Grenadines   base-lstm   6474.0
599   population ages 65 and above   0.891379    0.917131   0.957670     -0.393657                     St. Vincent and the Grenadines  simple-rnn   6933.0
2021  population ages 65 and above   1.010683    1.159845   1.076961     -0.762480                     St. Vincent and the Grenadines    base-gru   7490.0
2734          population ages 0-14   0.135554    0.027408   0.165555      0.904066                                 Sub-Saharan Africa     xgboost   1402.0
3445          population ages 0-14   0.286021    0.184777   0.429857      0.353250                                 Sub-Saharan Africa          rf   3789.0
2023          population ages 0-14   0.763863    0.654379   0.808937     -1.290431                                 Sub-Saharan Africa    base-gru   6666.0
1312          population ages 0-14   0.882065    0.878169   0.937107     -2.073734                                 Sub-Saharan Africa   base-lstm   7353.0
601           population ages 0-14   1.148920    1.477799   1.215648     -4.172534                                 Sub-Saharan Africa  simple-rnn   8681.0
2022         population ages 15-64   0.202458    0.049456   0.222387      0.792747                                 Sub-Saharan Africa    base-gru   2129.0
2733         population ages 15-64   0.252996    0.088309   0.297169      0.629926                                 Sub-Saharan Africa     xgboost   2858.0
1311         population ages 15-64   0.376897    0.167036   0.408700      0.300011                                 Sub-Saharan Africa   base-lstm   3953.0
3444         population ages 15-64   0.417331    0.256838   0.506792     -0.076316                                 Sub-Saharan Africa          rf   4616.0
600          population ages 15-64   0.601910    0.413408   0.642968     -0.732448                                 Sub-Saharan Africa  simple-rnn   5685.0
2735  population ages 65 and above   0.037405    0.002092   0.045738      0.043299                                 Sub-Saharan Africa     xgboost   1296.0
3446  population ages 65 and above   0.043017    0.003466   0.058872     -0.585061                                 Sub-Saharan Africa          rf   1615.0
602   population ages 65 and above   0.431864    0.212829   0.461334    -96.331783                                 Sub-Saharan Africa  simple-rnn   6463.0
1313  population ages 65 and above   0.465878    0.241649   0.491578   -109.511945                                 Sub-Saharan Africa   base-lstm   6655.0
2024  population ages 65 and above   0.535147    0.320955   0.566529   -145.780573                                 Sub-Saharan Africa    base-gru   7104.0
2737          population ages 0-14   0.135554    0.027408   0.165555      0.904066          Sub-Saharan Africa (IDA & IBRD countries)     xgboost   1398.0
3448          population ages 0-14   0.286021    0.184777   0.429857      0.353250          Sub-Saharan Africa (IDA & IBRD countries)          rf   3785.0
2026          population ages 0-14   0.763863    0.654379   0.808937     -1.290431          Sub-Saharan Africa (IDA & IBRD countries)    base-gru   6670.0
1315          population ages 0-14   0.882065    0.878169   0.937107     -2.073734          Sub-Saharan Africa (IDA & IBRD countries)   base-lstm   7357.0
604           population ages 0-14   1.148920    1.477799   1.215648     -4.172534          Sub-Saharan Africa (IDA & IBRD countries)  simple-rnn   8685.0
2025         population ages 15-64   0.202458    0.049456   0.222387      0.792747          Sub-Saharan Africa (IDA & IBRD countries)    base-gru   2129.0
2736         population ages 15-64   0.252996    0.088309   0.297169      0.629926          Sub-Saharan Africa (IDA & IBRD countries)     xgboost   2858.0
1314         population ages 15-64   0.376897    0.167036   0.408700      0.300011          Sub-Saharan Africa (IDA & IBRD countries)   base-lstm   3953.0
3447         population ages 15-64   0.417331    0.256838   0.506792     -0.076316          Sub-Saharan Africa (IDA & IBRD countries)          rf   4616.0
603          population ages 15-64   0.601910    0.413408   0.642968     -0.732448          Sub-Saharan Africa (IDA & IBRD countries)  simple-rnn   5685.0
2738  population ages 65 and above   0.037405    0.002092   0.045738      0.043299          Sub-Saharan Africa (IDA & IBRD countries)     xgboost   1296.0
3449  population ages 65 and above   0.043017    0.003466   0.058872     -0.585061          Sub-Saharan Africa (IDA & IBRD countries)          rf   1615.0
605   population ages 65 and above   0.431864    0.212829   0.461334    -96.331783          Sub-Saharan Africa (IDA & IBRD countries)  simple-rnn   6463.0
1316  population ages 65 and above   0.465878    0.241649   0.491578   -109.511945          Sub-Saharan Africa (IDA & IBRD countries)   base-lstm   6655.0
2027  population ages 65 and above   0.535147    0.320955   0.566529   -145.780573          Sub-Saharan Africa (IDA & IBRD countries)    base-gru   7104.0
2740          population ages 0-14   0.099736    0.013436   0.115916      0.952991         Sub-Saharan Africa (excluding high income)     xgboost    851.0
3451          population ages 0-14   0.287628    0.185115   0.430250      0.352357         Sub-Saharan Africa (excluding high income)          rf   3799.0
2029          population ages 0-14   0.764080    0.654767   0.809177     -1.290770         Sub-Saharan Africa (excluding high income)    base-gru   6675.0
1318          population ages 0-14   0.882157    0.878261   0.937156     -2.072689         Sub-Saharan Africa (excluding high income)   base-lstm   7358.0
607           population ages 0-14   1.148458    1.476847   1.215256     -4.166901         Sub-Saharan Africa (excluding high income)  simple-rnn   8676.0
2028         population ages 15-64   0.202825    0.049631   0.222780      0.792128         Sub-Saharan Africa (excluding high income)    base-gru   2144.0
2739         population ages 15-64   0.278168    0.115627   0.340040      0.515711         Sub-Saharan Africa (excluding high income)     xgboost   3232.0
1317         population ages 15-64   0.377061    0.167137   0.408823      0.299971         Sub-Saharan Africa (excluding high income)   base-lstm   3960.0
3450         population ages 15-64   0.415574    0.255149   0.505122     -0.068656         Sub-Saharan Africa (excluding high income)          rf   4591.0
606          population ages 15-64   0.601653    0.413150   0.642767     -0.730424         Sub-Saharan Africa (excluding high income)  simple-rnn   5676.0
2741  population ages 65 and above   0.037293    0.002081   0.045623      0.047555         Sub-Saharan Africa (excluding high income)     xgboost   1288.0
3452  population ages 65 and above   0.042595    0.003430   0.058568     -0.569609         Sub-Saharan Africa (excluding high income)          rf   1604.0
608   population ages 65 and above   0.431756    0.212735   0.461233    -96.343417         Sub-Saharan Africa (excluding high income)  simple-rnn   6458.0
1319  population ages 65 and above   0.465830    0.241596   0.491524   -109.549369         Sub-Saharan Africa (excluding high income)   base-lstm   6652.0
2030  population ages 65 and above   0.535050    0.320861   0.566446   -145.819259         Sub-Saharan Africa (excluding high income)    base-gru   7101.0
2743          population ages 0-14   0.394380    0.191903   0.438067      0.067043                                              Sudan     xgboost   4239.0
3454          population ages 0-14   0.509639    0.301021   0.548654     -0.463447                                              Sudan          rf   5107.0
1321          population ages 0-14   0.946538    0.976939   0.988402     -3.749496                                              Sudan   base-lstm   7849.0
2032          population ages 0-14   1.031421    1.157215   1.075739     -4.625927                                              Sudan    base-gru   8239.0
610           population ages 0-14   1.158140    1.462588   1.209375     -6.110536                                              Sudan  simple-rnn   8823.0
2742         population ages 15-64   0.303995    0.131980   0.363290     -3.406780                                              Sudan     xgboost   4623.0
2031         population ages 15-64   0.693720    0.557570   0.746706    -17.617172                                              Sudan    base-gru   7440.0
1320         population ages 15-64   0.706084    0.593812   0.770592    -18.827313                                              Sudan   base-lstm   7564.0
609          population ages 15-64   0.899090    0.960103   0.979848    -31.057701                                              Sudan  simple-rnn   8588.0
3453         population ages 15-64   1.795934    4.197180   2.048702   -139.143258                                              Sudan          rf  11485.0
611   population ages 65 and above   0.285790    0.088721   0.297861      0.277505                                              Sudan  simple-rnn   3201.0
1322  population ages 65 and above   0.294119    0.097196   0.311763      0.208491                                              Sudan   base-lstm   3335.0
3455  population ages 65 and above   0.315924    0.159810   0.399762     -0.301400                                              Sudan          rf   4076.0
2033  population ages 65 and above   0.396860    0.160187   0.400234     -0.304471                                              Sudan    base-gru   4261.0
2744  population ages 65 and above   0.373869    0.235911   0.485707     -0.921129                                              Sudan     xgboost   4756.0
3457          population ages 0-14   0.036113    0.003681   0.060670      0.997299                                           Suriname          rf    187.0
2746          population ages 0-14   1.443459    3.281901   1.811602     -1.408523                                           Suriname     xgboost   9375.0
2035          population ages 0-14   2.343170    5.669775   2.381129     -3.160936                                           Suriname    base-gru  10810.0
1324          population ages 0-14   2.541668    6.741860   2.596509     -3.947719                                           Suriname   base-lstm  11155.0
613           population ages 0-14   2.699854    7.582287   2.753595     -4.564492                                           Suriname  simple-rnn  11388.0
3456         population ages 15-64   0.073741    0.007430   0.086197      0.986683                                           Suriname          rf    453.0
2034         population ages 15-64   0.967281    0.959780   0.979683     -0.720306                                           Suriname    base-gru   7201.0
2745         population ages 15-64   0.806571    1.169602   1.081482     -1.096392                                           Suriname     xgboost   7365.0
1323         population ages 15-64   1.489658    2.236086   1.495355     -3.007954                                           Suriname   base-lstm   9289.0
612          population ages 15-64   1.571171    2.481790   1.575370     -3.448354                                           Suriname  simple-rnn   9564.0
2747  population ages 65 and above   0.082304    0.010272   0.101353      0.945739                                           Suriname     xgboost    712.0
3458  population ages 65 and above   0.086503    0.010597   0.102943      0.944024                                           Suriname          rf    734.0
1325  population ages 65 and above   1.244687    1.721285   1.311978     -8.092134                                           Suriname   base-lstm   9237.0
614   population ages 65 and above   1.253373    1.760300   1.326763     -8.298216                                           Suriname  simple-rnn   9280.0
2036  population ages 65 and above   1.421672    2.237967   1.495984    -10.821340                                           Suriname    base-gru   9790.0
2749          population ages 0-14   0.570523    0.499394   0.706678     -1.696805                                             Sweden     xgboost   6131.0
3460          population ages 0-14   1.246041    2.277724   1.509213    -11.300066                                             Sweden          rf   9685.0
2038          population ages 0-14   2.809757    8.968958   2.994822    -47.433771                                             Sweden    base-gru  12373.0
1327          population ages 0-14   3.369272   12.772117   3.573810    -67.971423                                             Sweden   base-lstm  12991.0
616           population ages 0-14   3.497452   13.712572   3.703049    -73.050026                                             Sweden  simple-rnn  13121.0
2748         population ages 15-64   1.028417    1.251124   1.118537     -0.024942                                             Sweden     xgboost   7319.0
3459         population ages 15-64   1.160657    1.633173   1.277957     -0.337923                                             Sweden          rf   7978.0
2037         population ages 15-64   1.697921    3.825801   1.955965     -2.134160                                             Sweden    base-gru   9908.0
615          population ages 15-64   2.513648    8.033209   2.834292     -5.580940                                             Sweden  simple-rnn  11448.0
1326         population ages 15-64   2.936121   10.567975   3.250842     -7.657462                                             Sweden   base-lstm  11973.0
2039  population ages 65 and above   0.184165    0.050723   0.225218      0.891209                                             Sweden    base-gru   1900.0
1328  population ages 65 and above   0.215917    0.064239   0.253455      0.862220                                             Sweden   base-lstm   2212.0
617   population ages 65 and above   0.241277    0.077838   0.278994      0.833054                                             Sweden  simple-rnn   2450.0
2750  population ages 65 and above   0.471088    0.311651   0.558257      0.331573                                             Sweden     xgboost   4690.0
3461  population ages 65 and above   0.751656    0.895054   0.946073     -0.919707                                             Sweden          rf   6885.0
2752          population ages 0-14   0.097863    0.012596   0.112230      0.020603                                        Switzerland     xgboost   1776.0
3463          population ages 0-14   0.224132    0.081383   0.285276     -5.328078                                        Switzerland          rf   4198.0
2041          population ages 0-14   2.397988    6.328493   2.515650   -491.086159                                        Switzerland    base-gru  12310.0
1330          population ages 0-14   2.902982    9.239087   3.039587   -717.405877                                        Switzerland   base-lstm  12870.0
619           population ages 0-14   3.211656   11.255993   3.354995   -874.234899                                        Switzerland  simple-rnn  13202.0
3462         population ages 15-64   0.129263    0.026820   0.163768      0.943512                                        Switzerland          rf   1249.0
2040         population ages 15-64   0.370199    0.210693   0.459014      0.556234                                        Switzerland    base-gru   3960.0
2751         population ages 15-64   1.034025    1.559621   1.248848     -2.284903                                        Switzerland     xgboost   8338.0
618          population ages 15-64   1.089841    1.668739   1.291797     -2.514730                                        Switzerland  simple-rnn   8512.0
1329         population ages 15-64   1.647621    3.362389   1.833682     -6.081928                                        Switzerland   base-lstm  10200.0
2753  population ages 65 and above   0.164259    0.047549   0.218058      0.909206                                        Switzerland     xgboost   1738.0
3464  population ages 65 and above   0.184163    0.049224   0.221864      0.906009                                        Switzerland          rf   1822.0
1331  population ages 65 and above   1.087510    1.228857   1.108538     -1.346468                                        Switzerland   base-lstm   7839.0
620   population ages 65 and above   1.132408    1.347952   1.161013     -1.573877                                        Switzerland  simple-rnn   8078.0
2042  population ages 65 and above   1.422156    2.118093   1.455367     -3.044440                                        Switzerland    base-gru   9193.0
2755          population ages 0-14   1.983162    6.732029   2.594615     -0.223826                               Syrian Arab Republic     xgboost  10062.0
3466          population ages 0-14   3.157051   15.345288   3.917306     -1.789644                               Syrian Arab Republic          rf  11751.0
1333          population ages 0-14   5.464492   37.594741   6.131455     -5.834408                               Syrian Arab Republic   base-lstm  12869.0
622           population ages 0-14   5.606067   39.135152   6.255809     -6.114442                               Syrian Arab Republic  simple-rnn  12902.0
2044          population ages 0-14   5.624024   39.843133   6.312142     -6.243147                               Syrian Arab Republic    base-gru  12919.0
2754         population ages 15-64   3.152688   15.936710   3.992081     -1.866681                               Syrian Arab Republic     xgboost  11801.0
3465         population ages 15-64   3.811208   21.370506   4.622824     -2.844107                               Syrian Arab Republic          rf  12281.0
2043         population ages 15-64   4.504764   28.247877   5.314873     -4.081202                               Syrian Arab Republic    base-gru  12618.0
621          population ages 15-64   4.639095   29.345884   5.417184     -4.278711                               Syrian Arab Republic  simple-rnn  12651.0
1332         population ages 15-64   4.660982   29.691477   5.448989     -4.340876                               Syrian Arab Republic   base-lstm  12663.0
2756  population ages 65 and above   0.171529    0.042011   0.204967      0.798195                               Syrian Arab Republic     xgboost   1948.0
3467  population ages 65 and above   0.645043    0.630469   0.794021     -2.028507                               Syrian Arab Republic          rf   6616.0
1334  population ages 65 and above   0.782108    0.667800   0.817190     -2.207825                               Syrian Arab Republic   base-lstm   6950.0
623   population ages 65 and above   0.891505    0.889181   0.942964     -3.271247                               Syrian Arab Republic  simple-rnn   7607.0
2045  population ages 65 and above   1.072251    1.261943   1.123362     -5.061838                               Syrian Arab Republic    base-gru   8431.0
3469          population ages 0-14   0.042940    0.004809   0.069348     -0.012024                                         Tajikistan          rf   1412.0
2758          population ages 0-14   0.193659    0.067997   0.260762    -13.308959                                         Tajikistan     xgboost   4353.0
1336          population ages 0-14   2.764179    8.083533   2.843155  -1700.057887                                         Tajikistan   base-lstm  12769.0
2047          population ages 0-14   2.790799    8.241885   2.870868  -1733.380661                                         Tajikistan    base-gru  12782.0
625           population ages 0-14   2.976228    9.400422   3.066011  -1977.177373                                         Tajikistan  simple-rnn  12988.0
2757         population ages 15-64   0.077896    0.010761   0.103736     -4.006874                                         Tajikistan     xgboost   2640.0
3468         population ages 15-64   0.087995    0.011290   0.106254     -4.252857                                         Tajikistan          rf   2710.0
2046         population ages 15-64   1.901899    4.022131   2.005525  -1870.370468                                         Tajikistan    base-gru  11778.0
1335         population ages 15-64   1.928198    4.187510   2.046341  -1947.316260                                         Tajikistan   base-lstm  11840.0
624          population ages 15-64   2.216755    5.395543   2.322831  -2509.376092                                         Tajikistan  simple-rnn  12215.0
2759  population ages 65 and above   0.019065    0.000453   0.021285      0.964681                                         Tajikistan     xgboost    184.0
3470  population ages 65 and above   0.144555    0.033075   0.181865     -1.578556                                         Tajikistan          rf   2933.0
1337  population ages 65 and above   0.992056    0.987177   0.993568    -75.961605                                         Tajikistan   base-lstm   8932.0
626   population ages 65 and above   1.009609    1.023574   1.011718    -78.799116                                         Tajikistan  simple-rnn   8999.0
2048  population ages 65 and above   1.142099    1.307938   1.143651   -100.968471                                         Tajikistan    base-gru   9512.0
3472          population ages 0-14   0.163186    0.035274   0.187814      0.836512                                           Tanzania          rf   1758.0
2761          population ages 0-14   0.672075    0.539250   0.734337     -1.499311                                           Tanzania     xgboost   6347.0
2050          population ages 0-14   1.464709    2.303731   1.517805     -9.677305                                           Tanzania    base-gru   9826.0
1339          population ages 0-14   1.627004    2.855973   1.689962    -12.236830                                           Tanzania   base-lstm  10261.0
628           population ages 0-14   1.899069    3.904303   1.975931    -17.095616                                           Tanzania  simple-rnn  10925.0
3471         population ages 15-64   0.200780    0.046636   0.215954      0.718146                                           Tanzania          rf   2185.0
2760         population ages 15-64   0.976274    1.109864   1.053501     -5.707697                                           Tanzania     xgboost   8221.0
2049         population ages 15-64   1.051566    1.175332   1.084127     -6.103365                                           Tanzania    base-gru   8393.0
1338         population ages 15-64   1.268310    1.713795   1.309120     -9.357681                                           Tanzania   base-lstm   9309.0
627          population ages 15-64   1.487679    2.358056   1.535596    -13.251408                                           Tanzania  simple-rnn   9977.0
3473  population ages 65 and above   0.080250    0.012131   0.110140     -0.642328                                           Tanzania          rf   1970.0
2762  population ages 65 and above   0.205653    0.074962   0.273793     -9.148820                                           Tanzania     xgboost   4317.0
629   population ages 65 and above   0.335988    0.139036   0.372876    -17.823487                                           Tanzania  simple-rnn   5426.0
1340  population ages 65 and above   0.369017    0.160196   0.400244    -20.688155                                           Tanzania   base-lstm   5690.0
2051  population ages 65 and above   0.439894    0.226107   0.475507    -29.611595                                           Tanzania    base-gru   6248.0
3475          population ages 0-14   0.575867    0.424237   0.651335      0.704645                                           Thailand          rf   4884.0
2764          population ages 0-14   0.867643    1.001631   1.000815      0.302661                                           Thailand     xgboost   6668.0
2053          population ages 0-14   2.454119    6.509337   2.551340     -3.531824                                           Thailand    base-gru  11028.0
1342          population ages 0-14   2.666108    7.892578   2.809373     -4.494841                                           Thailand   base-lstm  11398.0
631           population ages 0-14   2.731174    8.159589   2.856499     -4.680735                                           Thailand  simple-rnn  11463.0
2763         population ages 15-64   0.435874    0.353901   0.594896      0.383642                                           Thailand     xgboost   4719.0
3474         population ages 15-64   0.610753    0.715081   0.845624     -0.245391                                           Thailand          rf   6194.0
2052         population ages 15-64   2.563279    7.396569   2.719663    -11.881936                                           Thailand    base-gru  11694.0
630          population ages 15-64   2.767161    8.577249   2.928694    -13.938219                                           Thailand  simple-rnn  11929.0
1341         population ages 15-64   3.653620   15.229958   3.902558    -25.524640                                           Thailand   base-lstm  12967.0
2765  population ages 65 and above   0.237044    0.130878   0.361771      0.963073                                           Thailand     xgboost   2516.0
1343  population ages 65 and above   0.380981    0.182136   0.426774      0.948610                                           Thailand   base-lstm   3229.0
2054  population ages 65 and above   0.517024    0.324953   0.570046      0.908315                                           Thailand    base-gru   4139.0
632   population ages 65 and above   0.524507    0.328484   0.573135      0.907318                                           Thailand  simple-rnn   4167.0
3476  population ages 65 and above   1.254351    2.701233   1.643543      0.237847                                           Thailand          rf   8369.0
3478          population ages 0-14   0.253840    0.117006   0.342061      0.926289                                        Timor-Leste          rf   2579.0
2767          population ages 0-14   0.582626    0.468419   0.684411      0.704907                                        Timor-Leste     xgboost   5007.0
634           population ages 0-14   0.821030    0.682112   0.825901      0.570285                                        Timor-Leste  simple-rnn   5971.0
1345          population ages 0-14   0.892747    0.804685   0.897042      0.493066                                        Timor-Leste   base-lstm   6323.0
2056          population ages 0-14   1.080916    1.187728   1.089829      0.251757                                        Timor-Leste    base-gru   7142.0
2766         population ages 15-64   0.130261    0.033975   0.184322      0.980653                                        Timor-Leste     xgboost   1254.0
3477         population ages 15-64   0.454029    0.290064   0.538576      0.834822                                        Timor-Leste          rf   4079.0
1344         population ages 15-64   1.110058    1.472870   1.213619      0.161268                                        Timor-Leste   base-lstm   7524.0
633          population ages 15-64   1.160988    1.573964   1.254577      0.103700                                        Timor-Leste  simple-rnn   7728.0
2055         population ages 15-64   1.319868    2.031157   1.425187     -0.156651                                        Timor-Leste    base-gru   8293.0
2768  population ages 65 and above   0.063833    0.009518   0.097558     -0.848638                                        Timor-Leste     xgboost   1924.0
1346  population ages 65 and above   0.298003    0.119418   0.345568    -22.195069                                        Timor-Leste   base-lstm   5266.0
635   population ages 65 and above   0.304515    0.124397   0.352700    -23.162339                                        Timor-Leste  simple-rnn   5337.0
2057  population ages 65 and above   0.318307    0.137161   0.370352    -25.641445                                        Timor-Leste    base-gru   5517.0
3479  population ages 65 and above   0.410807    0.250036   0.500036    -47.565700                                        Timor-Leste          rf   6384.0
2770          population ages 0-14   0.341928    0.158693   0.398362      0.412140                                               Togo     xgboost   3749.0
3481          population ages 0-14   0.575270    0.410019   0.640327     -0.518871                                               Togo          rf   5547.0
2059          population ages 0-14   1.780110    3.602533   1.898034    -12.345209                                               Togo    base-gru  10624.0
1348          population ages 0-14   1.829773    3.874013   1.968251    -13.350880                                               Togo   base-lstm  10787.0
637           population ages 0-14   1.937462    4.285692   2.070191    -14.875901                                               Togo  simple-rnn  11019.0
2058         population ages 15-64   0.505931    0.312359   0.558891     -0.860547                                               Togo    base-gru   5268.0
3480         population ages 15-64   0.629621    0.534206   0.730894     -2.181964                                               Togo          rf   6431.0
636          population ages 15-64   0.751347    0.700385   0.836890     -3.171804                                               Togo  simple-rnn   7143.0
1347         population ages 15-64   0.766458    0.747093   0.864346     -3.450017                                               Togo   base-lstm   7279.0
2769         population ages 15-64   0.820404    0.831932   0.912103     -3.955355                                               Togo     xgboost   7534.0
3482  population ages 65 and above   0.026670    0.001078   0.032828      0.913294                                               Togo          rf    377.0
2771  population ages 65 and above   0.129148    0.024391   0.156177     -0.962486                                               Togo     xgboost   2500.0
1349  population ages 65 and above   0.930989    0.942827   0.970993    -74.858087                                               Togo   base-lstm   8804.0
638   population ages 65 and above   0.971288    1.031877   1.015814    -82.022906                                               Togo  simple-rnn   8973.0
2060  population ages 65 and above   1.153194    1.451510   1.204786   -115.785733                                               Togo    base-gru   9701.0
3484          population ages 0-14   0.095421    0.012220   0.110545      0.979712                                              Tonga          rf    689.0
2773          population ages 0-14   0.390688    0.225476   0.474843      0.625660                                              Tonga     xgboost   4002.0
2062          population ages 0-14   1.394313    2.020084   1.421297     -2.353782                                              Tonga    base-gru   9006.0
640           population ages 0-14   1.549653    2.496450   1.580016     -3.144654                                              Tonga  simple-rnn   9505.0
1351          population ages 0-14   1.580331    2.617102   1.617746     -3.344962                                              Tonga   base-lstm   9624.0
2061         population ages 15-64   0.063660    0.005822   0.076303      0.987897                                              Tonga    base-gru    363.0
2772         population ages 15-64   0.143237    0.031911   0.178637      0.933665                                              Tonga     xgboost   1412.0
639          population ages 15-64   0.173899    0.033956   0.184272      0.929414                                              Tonga  simple-rnn   1564.0
3483         population ages 15-64   0.298002    0.113680   0.337165      0.763688                                              Tonga          rf   3026.0
1350         population ages 15-64   0.342429    0.126726   0.355986      0.736569                                              Tonga   base-lstm   3269.0
3485  population ages 65 and above   0.125763    0.019262   0.138789     -1.758549                                              Tonga          rf   2635.0
2774  population ages 65 and above   0.119535    0.026476   0.162713     -2.791551                                              Tonga     xgboost   2937.0
1352  population ages 65 and above   0.661991    0.495027   0.703582    -69.892453                                              Tonga   base-lstm   7639.0
641   population ages 65 and above   0.728517    0.596593   0.772395    -84.437618                                              Tonga  simple-rnn   8021.0
2063  population ages 65 and above   0.853874    0.804696   0.897049   -114.239870                                              Tonga    base-gru   8609.0
3487          population ages 0-14   0.384589    0.217836   0.466729      0.171523                                Trinidad and Tobago          rf   4284.0
2776          population ages 0-14   0.509341    0.535388   0.731702     -1.036196                                Trinidad and Tobago     xgboost   5906.0
2065          population ages 0-14   3.342530   12.291197   3.505880    -45.746077                                Trinidad and Tobago    base-gru  12868.0
1354          population ages 0-14   3.354045   12.617306   3.552085    -46.986342                                Trinidad and Tobago   base-lstm  12897.0
643           population ages 0-14   3.431843   13.015440   3.607692    -48.500531                                Trinidad and Tobago  simple-rnn  12963.0
2775         population ages 15-64   3.103552   15.391052   3.923143    -15.604598                                Trinidad and Tobago     xgboost  12634.0
2064         population ages 15-64   3.758313   15.455193   3.931309    -15.673796                                Trinidad and Tobago    base-gru  12819.0
642          population ages 15-64   3.837161   16.007601   4.000950    -16.269760                                Trinidad and Tobago  simple-rnn  12871.0
3486         population ages 15-64   3.272364   23.563501   4.854225    -24.421424                                Trinidad and Tobago          rf  13070.0
1353         population ages 15-64   4.220341   19.764833   4.445766    -20.323241                                Trinidad and Tobago   base-lstm  13109.0
2777  population ages 65 and above   0.537585    0.533166   0.730182      0.751269                                Trinidad and Tobago     xgboost   5019.0
1355  population ages 65 and above   0.728782    0.558032   0.747015      0.739668                                Trinidad and Tobago   base-lstm   5434.0
2066  population ages 65 and above   0.779055    0.630105   0.793792      0.706045                                Trinidad and Tobago    base-gru   5676.0
644   population ages 65 and above   0.883745    0.789005   0.888260      0.631915                                Trinidad and Tobago  simple-rnn   6161.0
3488  population ages 65 and above   1.096125    1.918700   1.385171      0.104893                                Trinidad and Tobago          rf   7875.0
2779          population ages 0-14   0.698547    0.794335   0.891255     -1.991526                                            Tunisia     xgboost   6965.0
3490          population ages 0-14   1.112269    1.750416   1.323033     -5.592199                                            Tunisia          rf   8945.0
2068          population ages 0-14   4.753181   26.762995   5.173296    -99.791478                                            Tunisia    base-gru  13652.0
1357          population ages 0-14   4.743800   27.015996   5.197691   -100.744298                                            Tunisia   base-lstm  13654.0
646           population ages 0-14   4.869138   28.150254   5.305681   -105.015998                                            Tunisia  simple-rnn  13687.0
2778         population ages 15-64   0.169327    0.043525   0.208627      0.966055                                            Tunisia     xgboost   1533.0
3489         population ages 15-64   1.572371    3.778706   1.943889     -1.946979                                            Tunisia          rf   9766.0
2067         population ages 15-64   3.476701   14.120037   3.757664    -10.012089                                            Tunisia    base-gru  12535.0
645          population ages 15-64   3.743897   16.354175   4.044029    -11.754472                                            Tunisia  simple-rnn  12748.0
1356         population ages 15-64   4.155350   20.339352   4.509917    -14.862476                                            Tunisia   base-lstm  12985.0
3491  population ages 65 and above   0.158057    0.031843   0.178446      0.920289                                            Tunisia          rf   1498.0
2780  population ages 65 and above   0.527044    0.420227   0.648249     -0.051930                                            Tunisia     xgboost   5282.0
1358  population ages 65 and above   1.266134    1.781528   1.334739     -3.459594                                            Tunisia   base-lstm   8943.0
647   population ages 65 and above   1.526673    2.615394   1.617218     -5.546961                                            Tunisia  simple-rnn   9792.0
2069  population ages 65 and above   1.610288    2.883365   1.698048     -6.217758                                            Tunisia    base-gru  10007.0
3493          population ages 0-14   0.071458    0.008414   0.091729      0.993772                                            Turkiye          rf    433.0
2782          population ages 0-14   0.641272    0.624123   0.790015      0.538062                                            Turkiye     xgboost   5602.0
2071          population ages 0-14   3.243562   11.715103   3.422733     -7.670806                                            Turkiye    base-gru  12195.0
1360          population ages 0-14   3.340895   12.630681   3.553967     -8.348461                                            Turkiye   base-lstm  12323.0
649           population ages 0-14   3.580188   14.290599   3.780291     -9.577032                                            Turkiye  simple-rnn  12558.0
3492         population ages 15-64   0.227583    0.079084   0.281220      0.770221                                            Turkiye          rf   2541.0
2781         population ages 15-64   0.325698    0.146750   0.383080      0.573619                                            Turkiye     xgboost   3523.0
2070         population ages 15-64   1.586293    2.775742   1.666056     -7.064878                                            Turkiye    base-gru  10002.0
648          population ages 15-64   2.140970    5.077398   2.253308    -13.752302                                            Turkiye  simple-rnn  11250.0
1359         population ages 15-64   2.385849    6.528428   2.555079    -17.968250                                            Turkiye   base-lstm  11640.0
3494  population ages 65 and above   0.059097    0.005568   0.074619      0.987663                                            Turkiye          rf    346.0
2783  population ages 65 and above   0.206266    0.076109   0.275878      0.831365                                            Turkiye     xgboost   2334.0
1361  population ages 65 and above   1.302071    1.873542   1.368774     -3.151227                                            Turkiye   base-lstm   8976.0
650   population ages 65 and above   1.543683    2.666084   1.632815     -4.907269                                            Turkiye  simple-rnn   9771.0
2072  population ages 65 and above   1.707768    3.220602   1.794604     -6.135921                                            Turkiye    base-gru  10194.0
2785          population ages 0-14   0.311796    0.135356   0.367907     -1.824935                                       Turkmenistan     xgboost   4397.0
3496          population ages 0-14   0.440443    0.257979   0.507916     -4.384134                                       Turkmenistan          rf   5663.0
1363          population ages 0-14   3.851901   15.800552   3.974991   -328.764304                                       Turkmenistan   base-lstm  13520.0
2074          population ages 0-14   4.015105   17.146212   4.140798   -356.848797                                       Turkmenistan    base-gru  13606.0
652           population ages 0-14   4.078505   17.714684   4.208882   -368.713051                                       Turkmenistan  simple-rnn  13640.0
2784         population ages 15-64   0.152345    0.038286   0.195668      0.832749                                       Turkmenistan     xgboost   1775.0
3495         population ages 15-64   0.402099    0.212835   0.461341      0.070241                                       Turkmenistan          rf   4348.0
2073         population ages 15-64   3.583066   13.996657   3.741211    -60.143589                                       Turkmenistan    base-gru  13110.0
1362         population ages 15-64   3.638394   14.489706   3.806535    -62.297444                                       Turkmenistan   base-lstm  13163.0
651          population ages 15-64   3.835562   15.866098   3.983227    -68.310132                                       Turkmenistan  simple-rnn  13266.0
2786  population ages 65 and above   0.095061    0.018280   0.135204      0.750485                                       Turkmenistan     xgboost   1382.0
3497  population ages 65 and above   0.289520    0.123370   0.351241     -0.683957                                       Turkmenistan          rf   3908.0
1364  population ages 65 and above   0.718685    0.521692   0.722282     -6.120880                                       Turkmenistan   base-lstm   6995.0
653   population ages 65 and above   0.830387    0.691991   0.831860     -8.445400                                       Turkmenistan  simple-rnn   7655.0
2075  population ages 65 and above   0.889699    0.795491   0.891903     -9.858133                                       Turkmenistan    base-gru   7940.0
1366          population ages 0-14   0.507149    0.265112   0.514890      0.832476                                             Uganda   base-lstm   4109.0
2788          population ages 0-14   0.445785    0.301242   0.548855      0.809645                                             Uganda     xgboost   4151.0
2077          population ages 0-14   0.531370    0.288302   0.536938      0.817822                                             Uganda    base-gru   4251.0
655           population ages 0-14   0.675855    0.461695   0.679482      0.708254                                             Uganda  simple-rnn   5146.0
3499          population ages 0-14   0.752025    0.850859   0.922420      0.462341                                             Uganda          rf   6225.0
2076         population ages 15-64   0.185461    0.040418   0.201043      0.972355                                             Uganda    base-gru   1542.0
654          population ages 15-64   0.185720    0.040637   0.201587      0.972205                                             Uganda  simple-rnn   1546.0
1365         population ages 15-64   0.193084    0.044673   0.211360      0.969445                                             Uganda   base-lstm   1608.0
2787         population ages 15-64   0.428203    0.257112   0.507062      0.824143                                             Uganda     xgboost   3946.0
3498         population ages 15-64   0.713125    0.883663   0.940034      0.395600                                             Uganda          rf   6245.0
2789  population ages 65 and above   0.017936    0.000426   0.020651      0.821632                                             Uganda     xgboost    533.0
3500  population ages 65 and above   0.047108    0.003376   0.058101     -0.411862                                             Uganda          rf   1558.0
656   population ages 65 and above   0.757228    0.587465   0.766463   -244.701405                                             Uganda  simple-rnn   8180.0
1367  population ages 65 and above   0.798763    0.650410   0.806480   -271.027373                                             Uganda   base-lstm   8394.0
2078  population ages 65 and above   0.815954    0.679985   0.824612   -283.396924                                             Uganda    base-gru   8487.0
3502          population ages 0-14   0.031008    0.001583   0.039787      0.752689                                            Ukraine          rf    730.0
2791          population ages 0-14   0.435234    0.337625   0.581055    -51.747556                                            Ukraine     xgboost   6762.0
2080          population ages 0-14   3.638204   13.344765   3.653049  -2083.867155                                            Ukraine    base-gru  13518.0
1369          population ages 0-14   3.866941   15.139689   3.890975  -2364.290031                                            Ukraine   base-lstm  13665.0
658           population ages 0-14   4.175852   17.622525   4.197919  -2752.186277                                            Ukraine  simple-rnn  13801.0
3501         population ages 15-64   0.094472    0.012787   0.113082      0.932662                                            Ukraine          rf    863.0
2790         population ages 15-64   0.178061    0.061645   0.248285      0.675378                                            Ukraine     xgboost   2345.0
2079         population ages 15-64   1.553319    2.646127   1.626692    -12.934396                                            Ukraine    base-gru  10158.0
657          population ages 15-64   1.925627    4.077722   2.019337    -20.473118                                            Ukraine  simple-rnn  11072.0
1368         population ages 15-64   2.634410    7.484636   2.735806    -38.413787                                            Ukraine   base-lstm  12124.0
3503  population ages 65 and above   0.140424    0.024464   0.156410      0.890864                                            Ukraine          rf   1390.0
2792  population ages 65 and above   0.302018    0.161343   0.401676      0.280239                                            Ukraine     xgboost   3758.0
1370  population ages 65 and above   0.743274    0.580748   0.762068     -1.590743                                            Ukraine   base-lstm   6563.0
659   population ages 65 and above   0.852710    0.747609   0.864644     -2.335121                                            Ukraine  simple-rnn   7213.0
2081  population ages 65 and above   1.062296    1.151293   1.072983     -4.135971                                            Ukraine    base-gru   8219.0
3505          population ages 0-14   0.254308    0.122535   0.350050     -0.321401                               United Arab Emirates          rf   3672.0
2794          population ages 0-14   0.754350    0.940284   0.969682     -9.139882                               United Arab Emirates     xgboost   7926.0
661           population ages 0-14   8.370677   82.231430   9.068155   -885.771510                               United Arab Emirates  simple-rnn  14126.0
1372          population ages 0-14   8.393932   83.311898   9.127535   -897.423110                               United Arab Emirates   base-lstm  14130.0
2083          population ages 0-14   9.375350  103.519895  10.174473  -1115.343148                               United Arab Emirates    base-gru  14150.0
2793         population ages 15-64   0.765651    1.102761   1.050125     -0.698359                               United Arab Emirates     xgboost   7108.0
3504         population ages 15-64   0.845417    1.280074   1.131403     -0.971437                               United Arab Emirates          rf   7487.0
660          population ages 15-64  13.984561  213.281503  14.604160   -327.474113                               United Arab Emirates  simple-rnn  14015.0
1371         population ages 15-64  14.274097  225.220890  15.007361   -345.861921                               United Arab Emirates   base-lstm  14030.0
2082         population ages 15-64  15.057958  252.509612  15.890551   -387.889189                               United Arab Emirates    base-gru  14058.0
2795  population ages 65 and above   0.583419    0.557160   0.746432     -1.116825                               United Arab Emirates     xgboost   6135.0
3506  population ages 65 and above   0.640638    0.675397   0.821825     -1.566044                               United Arab Emirates          rf   6599.0
2084  population ages 65 and above   1.895169    3.622194   1.903206    -12.761845                               United Arab Emirates    base-gru  10715.0
1373  population ages 65 and above   2.281481    5.452986   2.335163    -19.717592                               United Arab Emirates   base-lstm  11477.0
662   population ages 65 and above   2.601956    7.165313   2.676810    -26.223256                               United Arab Emirates  simple-rnn  11952.0
3508          population ages 0-14   0.099672    0.020353   0.142665     -1.019867                                     United Kingdom          rf   2351.0
2797          population ages 0-14   0.196682    0.051945   0.227914     -4.155014                                     United Kingdom     xgboost   3740.0
2086          population ages 0-14   4.185854   18.860671   4.342887  -1870.733926                                     United Kingdom    base-gru  13827.0
1375          population ages 0-14   4.514330   21.934108   4.683386  -2175.741947                                     United Kingdom   base-lstm  13926.0
664           population ages 0-14   4.885878   25.715838   5.071079  -2551.041041                                     United Kingdom  simple-rnn  14004.0
3507         population ages 15-64   0.864004    0.945761   0.972503     -0.052657                                     United Kingdom          rf   6793.0
2085         population ages 15-64   1.446417    2.891290   1.700379     -2.218082                                     United Kingdom    base-gru   9432.0
2796         population ages 15-64   1.590837    3.538805   1.881171     -2.938783                                     United Kingdom     xgboost   9883.0
663          population ages 15-64   2.238769    6.606944   2.570398     -6.353703                                     United Kingdom  simple-rnn  11213.0
1374         population ages 15-64   2.718435    9.267606   3.044274     -9.315090                                     United Kingdom   base-lstm  11852.0
3509  population ages 65 and above   0.605450    0.478646   0.691843      0.378213                                     United Kingdom          rf   5355.0
2798  population ages 65 and above   0.806830    1.037074   1.018368     -0.347214                                     United Kingdom     xgboost   6964.0
1376  population ages 65 and above   1.024028    1.068567   1.033715     -0.388126                                     United Kingdom   base-lstm   7264.0
665   population ages 65 and above   1.147924    1.349245   1.161570     -0.752741                                     United Kingdom  simple-rnn   7850.0
2087  population ages 65 and above   1.513211    2.339906   1.529675     -2.039662                                     United Kingdom    base-gru   9195.0
3511          population ages 0-14   0.264133    0.089496   0.299159      0.662477                                      United States          rf   2872.0
2800          population ages 0-14   0.344899    0.251259   0.501258      0.052408                                      United States     xgboost   4365.0
2089          population ages 0-14   2.882574    8.912434   2.985370    -32.612088                                      United States    base-gru  12289.0
1378          population ages 0-14   3.354876   12.163440   3.487612    -44.872833                                      United States   base-lstm  12854.0
667           population ages 0-14   3.398751   12.460968   3.530010    -45.994923                                      United States  simple-rnn  12895.0
2088         population ages 15-64   0.669944    0.647192   0.804482     -0.428693                                      United States    base-gru   6229.0
2799         population ages 15-64   0.888678    1.212104   1.100956     -1.675752                                      United States     xgboost   7686.0
3510         population ages 15-64   0.961920    1.374083   1.172213     -2.033323                                      United States          rf   8019.0
666          population ages 15-64   1.215192    2.146766   1.465185     -3.739042                                      United States  simple-rnn   9146.0
1377         population ages 15-64   1.645385    3.766226   1.940677     -7.314042                                      United States   base-lstm  10408.0
2801  population ages 65 and above   0.833575    1.103723   1.050582      0.215116                                      United States     xgboost   6789.0
3512  population ages 65 and above   0.871477    1.264523   1.124510      0.100768                                      United States          rf   7075.0
1379  population ages 65 and above   1.174831    1.392305   1.179960      0.009899                                      United States   base-lstm   7614.0
668   population ages 65 and above   1.208945    1.474053   1.214106     -0.048234                                      United States  simple-rnn   7753.0
2090  population ages 65 and above   1.490503    2.241072   1.497021     -0.593679                                      United States    base-gru   8718.0
2803          population ages 0-14   0.277125    0.093544   0.305850      0.472912                                Upper middle income     xgboost   3104.0
3514          population ages 0-14   0.722842    0.683371   0.826663     -2.850555                                Upper middle income          rf   7000.0
2092          population ages 0-14   3.006949   10.279033   3.206093    -56.918706                                Upper middle income    base-gru  12612.0
1381          population ages 0-14   3.130314   11.358069   3.370173    -62.998694                                Upper middle income   base-lstm  12785.0
670           population ages 0-14   3.220902   11.857510   3.443474    -65.812865                                Upper middle income  simple-rnn  12872.0
3513         population ages 15-64   0.650800    0.627884   0.792391     -0.468001                                Upper middle income          rf   6175.0
2802         population ages 15-64   0.651307    0.691003   0.831266     -0.615575                                Upper middle income     xgboost   6371.0
2091         population ages 15-64   1.587266    3.012891   1.735768     -6.044179                                Upper middle income    base-gru  10016.0
669          population ages 15-64   1.797556    3.893318   1.973149     -8.102630                                Upper middle income  simple-rnn  10593.0
1380         population ages 15-64   2.306955    6.415583   2.532900    -13.999719                                Upper middle income   base-lstm  11507.0
2804  population ages 65 and above   0.335821    0.230855   0.480474      0.792589                                Upper middle income     xgboost   3700.0
3515  population ages 65 and above   0.411105    0.363305   0.602748      0.673591                                Upper middle income          rf   4434.0
1382  population ages 65 and above   1.397599    2.012222   1.418528     -0.807868                                Upper middle income   base-lstm   8594.0
671   population ages 65 and above   1.652663    2.842487   1.685968     -1.553814                                Upper middle income  simple-rnn   9392.0
2093  population ages 65 and above   1.658497    2.853138   1.689123     -1.563384                                Upper middle income    base-gru   9406.0
3517          population ages 0-14   0.421136    0.249123   0.499123      0.708044                                            Uruguay          rf   4057.0
2806          population ages 0-14   0.582026    0.432973   0.658007      0.492584                                            Uruguay     xgboost   5097.0
2095          population ages 0-14   3.908298   16.210993   4.026288    -17.998224                                            Uruguay    base-gru  12929.0
1384          population ages 0-14   4.176877   18.505288   4.301777    -20.686988                                            Uruguay   base-lstm  13082.0
673           population ages 0-14   4.417259   20.752476   4.555489    -23.320545                                            Uruguay  simple-rnn  13193.0
2094         population ages 15-64   0.073674    0.006907   0.083109      0.978084                                            Uruguay    base-gru    478.0
2805         population ages 15-64   0.318212    0.140500   0.374834      0.554193                                            Uruguay     xgboost   3496.0
3516         population ages 15-64   0.626218    0.676607   0.822561     -1.146870                                            Uruguay          rf   6474.0
672          population ages 15-64   0.770158    0.671906   0.819699     -1.131955                                            Uruguay  simple-rnn   6673.0
1383         population ages 15-64   1.004017    1.147186   1.071068     -2.640017                                            Uruguay   base-lstm   7933.0
3518  population ages 65 and above   0.334540    0.162603   0.403241     -0.228618                                            Uruguay          rf   4108.0
2807  population ages 65 and above   0.323958    0.175800   0.419285     -0.328330                                            Uruguay     xgboost   4195.0
1385  population ages 65 and above   2.044432    4.812969   2.193848    -35.366386                                            Uruguay   base-lstm  11478.0
674   population ages 65 and above   2.114125    5.151772   2.269751    -37.926348                                            Uruguay  simple-rnn  11586.0
2096  population ages 65 and above   2.537543    7.315219   2.704666    -54.273174                                            Uruguay    base-gru  12153.0
2809          population ages 0-14   0.121347    0.026262   0.162055      0.893155                                         Uzbekistan     xgboost   1342.0
3520          population ages 0-14   0.621384    0.586989   0.766152     -1.388133                                         Uzbekistan          rf   6342.0
1387          population ages 0-14   4.610377   22.901297   4.785530    -92.172616                                         Uzbekistan   base-lstm  13583.0
2098          population ages 0-14   4.742136   24.186927   4.918021    -97.403127                                         Uzbekistan    base-gru  13621.0
676           population ages 0-14   4.810987   24.905484   4.990539   -100.326534                                         Uzbekistan  simple-rnn  13645.0
2808         population ages 15-64   0.681546    0.666435   0.816355     -0.402135                                         Uzbekistan     xgboost   6276.0
3519         population ages 15-64   0.853353    1.037395   1.018526     -1.182610                                         Uzbekistan          rf   7307.0
2097         population ages 15-64   3.845806   16.455347   4.056519    -33.620964                                         Uzbekistan    base-gru  13141.0
1386         population ages 15-64   3.983315   17.755131   4.213684    -36.355624                                         Uzbekistan   base-lstm  13209.0
675          population ages 15-64   4.075658   18.407574   4.290405    -37.728321                                         Uzbekistan  simple-rnn  13246.0
2810  population ages 65 and above   0.126166    0.018596   0.136367      0.507007                                         Uzbekistan     xgboost   1721.0
3521  population ages 65 and above   0.232393    0.077822   0.278966     -1.063115                                         Uzbekistan          rf   3509.0
1388  population ages 65 and above   1.286399    1.660724   1.288691    -43.026966                                         Uzbekistan   base-lstm   9790.0
677   population ages 65 and above   1.403042    1.979178   1.406833    -51.469407                                         Uzbekistan  simple-rnn  10136.0
2099  population ages 65 and above   1.492036    2.233880   1.494617    -58.221732                                         Uzbekistan    base-gru  10357.0
2812          population ages 0-14   0.246417    0.088973   0.298283     -1.502267                                            Vanuatu     xgboost   3785.0
3523          population ages 0-14   0.510146    0.373250   0.610942     -9.497247                                            Vanuatu          rf   6442.0
1390          population ages 0-14   2.394719    6.337727   2.517484   -177.241750                                            Vanuatu   base-lstm  12165.0
2101          population ages 0-14   2.400824    6.342378   2.518408   -177.372533                                            Vanuatu    base-gru  12174.0
679           population ages 0-14   2.619553    7.584178   2.753939   -212.296829                                            Vanuatu  simple-rnn  12455.0
2811         population ages 15-64   0.036520    0.001648   0.040593      0.942889                                            Vanuatu     xgboost    340.0
3522         population ages 15-64   0.701704    0.700088   0.836713    -23.264790                                            Vanuatu          rf   7857.0
2100         population ages 15-64   1.462712    2.360433   1.536370    -80.811679                                            Vanuatu    base-gru  10489.0
1389         population ages 15-64   1.537070    2.653284   1.628890    -90.961778                                            Vanuatu   base-lstm  10700.0
678          population ages 15-64   1.767038    3.470466   1.862919   -119.285009                                            Vanuatu  simple-rnn  11212.0
3524  population ages 65 and above   0.033429    0.001578   0.039722     -2.688271                                            Vanuatu          rf   2046.0
2813  population ages 65 and above   0.186110    0.054158   0.232718   -125.597823                                            Vanuatu     xgboost   4818.0
1391  population ages 65 and above   0.793867    0.687684   0.829267  -1606.513994                                            Vanuatu   base-lstm   8648.0
680   population ages 65 and above   0.795704    0.696257   0.834420  -1626.555027                                            Vanuatu  simple-rnn   8676.0
2102  population ages 65 and above   0.926910    0.932983   0.965910  -2179.920278                                            Vanuatu    base-gru   9186.0
2815          population ages 0-14   0.227570    0.061977   0.248952      0.945431                                      Venezuela, RB     xgboost   2006.0
3526          population ages 0-14   0.241373    0.094206   0.306929      0.917055                                      Venezuela, RB          rf   2411.0
2104          population ages 0-14   2.492871    7.211647   2.685451     -5.349612                                      Venezuela, RB    base-gru  11320.0
1393          population ages 0-14   2.692340    8.502212   2.915855     -6.485910                                      Venezuela, RB   base-lstm  11620.0
682           population ages 0-14   2.957381   10.092204   3.176823     -7.885845                                      Venezuela, RB  simple-rnn  11950.0
2814         population ages 15-64   0.303248    0.133717   0.365674      0.713297                                      Venezuela, RB     xgboost   3239.0
3525         population ages 15-64   0.520513    0.433396   0.658328      0.070756                                      Venezuela, RB          rf   5232.0
2103         population ages 15-64   1.737019    3.534380   1.879995     -6.578062                                      Venezuela, RB    base-gru  10345.0
1392         population ages 15-64   2.274438    6.096988   2.469208    -12.072551                                      Venezuela, RB   base-lstm  11385.0
681          population ages 15-64   2.407242    6.687990   2.586115    -13.339717                                      Venezuela, RB  simple-rnn  11582.0
3527  population ages 65 and above   0.033933    0.001805   0.042487      0.988015                                      Venezuela, RB          rf    165.0
2816  population ages 65 and above   0.062220    0.004940   0.070286      0.967201                                      Venezuela, RB     xgboost    432.0
1394  population ages 65 and above   0.666586    0.507519   0.712404     -2.369625                                      Venezuela, RB   base-lstm   6471.0
683   population ages 65 and above   0.702599    0.565580   0.752051     -2.755117                                      Venezuela, RB  simple-rnn   6703.0
2105  population ages 65 and above   0.872444    0.847266   0.920470     -4.625343                                      Venezuela, RB    base-gru   7675.0
2818          population ages 0-14   0.321318    0.127795   0.357485     -0.202540                                           Viet Nam     xgboost   3850.0
3529          population ages 0-14   0.614159    0.489476   0.699626     -3.605924                                           Viet Nam          rf   6542.0
2107          population ages 0-14   3.495846   13.015727   3.607732   -121.476712                                           Viet Nam    base-gru  13170.0
1396          population ages 0-14   3.548736   13.517538   3.676620   -126.198701                                           Viet Nam   base-lstm  13221.0
685           population ages 0-14   3.618780   13.975677   3.738406   -130.509747                                           Viet Nam  simple-rnn  13265.0
2817         population ages 15-64   0.373845    0.183146   0.427955     -0.784805                                           Viet Nam     xgboost   4493.0
3528         population ages 15-64   0.449756    0.303954   0.551320     -1.962110                                           Viet Nam          rf   5440.0
2106         population ages 15-64   2.435201    6.665867   2.581834    -63.960612                                           Viet Nam    base-gru  12059.0
684          population ages 15-64   2.538615    7.258853   2.694226    -69.739410                                           Viet Nam  simple-rnn  12193.0
1395         population ages 15-64   2.931100    9.770027   3.125704    -94.211462                                           Viet Nam   base-lstm  12646.0
2819  population ages 65 and above   0.215436    0.072070   0.268458      0.827234                                           Viet Nam     xgboost   2333.0
3530  population ages 65 and above   0.284012    0.135645   0.368300      0.674831                                           Viet Nam          rf   3242.0
1397  population ages 65 and above   1.320991    1.747783   1.322037     -3.189802                                           Viet Nam   base-lstm   8920.0
686   population ages 65 and above   1.530794    2.354869   1.534558     -4.645115                                           Viet Nam  simple-rnn   9595.0
2108  population ages 65 and above   1.551902    2.414587   1.553894     -4.788273                                           Viet Nam    base-gru   9654.0
2821          population ages 0-14   0.053555    0.004132   0.064280     -1.815522                              Virgin Islands (U.S.)     xgboost   2031.0
3532          population ages 0-14   0.090748    0.016768   0.129492    -10.425988                              Virgin Islands (U.S.)          rf   3220.0
2110          population ages 0-14   1.305807    1.865827   1.365953  -1270.393274                              Virgin Islands (U.S.)    base-gru  10455.0
1399          population ages 0-14   1.623989    2.864865   1.692591  -1951.147119                              Virgin Islands (U.S.)   base-lstm  11196.0
688           population ages 0-14   1.628575    2.876404   1.695997  -1959.010248                              Virgin Islands (U.S.)  simple-rnn  11209.0
2820         population ages 15-64   0.176066    0.071582   0.267548      0.591024                              Virgin Islands (U.S.)     xgboost   2500.0
3531         population ages 15-64   0.768777    0.772269   0.878789     -3.412299                              Virgin Islands (U.S.)          rf   7301.0
2109         population ages 15-64   2.076155    4.422273   2.102920    -24.266296                              Virgin Islands (U.S.)    base-gru  11305.0
687          population ages 15-64   2.666497    7.269039   2.696116    -40.531064                              Virgin Islands (U.S.)  simple-rnn  12116.0
1398         population ages 15-64   2.931812    8.756203   2.959088    -49.027852                              Virgin Islands (U.S.)   base-lstm  12397.0
3533  population ages 65 and above   0.196945    0.061996   0.248990      0.700565                              Virgin Islands (U.S.)          rf   2394.0
2822  population ages 65 and above   0.286466    0.184942   0.430049      0.106742                              Virgin Islands (U.S.)     xgboost   3925.0
2111  population ages 65 and above   1.842817    3.396599   1.842986    -15.405315                              Virgin Islands (U.S.)    base-gru  10692.0
689   population ages 65 and above   1.963317    3.856543   1.963808    -17.626808                              Virgin Islands (U.S.)  simple-rnn  10954.0
1400  population ages 65 and above   1.974782    3.901605   1.975248    -17.844457                              Virgin Islands (U.S.)   base-lstm  10982.0
3535          population ages 0-14   0.406374    0.216619   0.465424     -0.304386                                 West Bank and Gaza          rf   4545.0
2824          population ages 0-14   0.594710    0.524244   0.724047     -2.156765                                 West Bank and Gaza     xgboost   6341.0
2113          population ages 0-14   1.782775    3.537574   1.880844    -20.301686                                 West Bank and Gaza    base-gru  10799.0
1402          population ages 0-14   1.812100    3.654555   1.911689    -21.006091                                 West Bank and Gaza   base-lstm  10872.0
691           population ages 0-14   2.002986    4.440781   2.107316    -25.740393                                 West Bank and Gaza  simple-rnn  11304.0
2823         population ages 15-64   0.063243    0.005220   0.072247      0.941369                                 West Bank and Gaza     xgboost    525.0
3534         population ages 15-64   0.420909    0.244444   0.494413     -1.745751                                 West Bank and Gaza          rf   5137.0
2112         population ages 15-64   1.563915    2.710823   1.646458    -29.449660                                 West Bank and Gaza    base-gru  10479.0
1401         population ages 15-64   1.670362    3.095045   1.759274    -33.765483                                 West Bank and Gaza   base-lstm  10743.0
690          population ages 15-64   1.847718    3.756582   1.938190    -41.196288                                 West Bank and Gaza  simple-rnn  11125.0
3536  population ages 65 and above   0.020998    0.000819   0.028613      0.933676                                 West Bank and Gaza          rf    297.0
2825  population ages 65 and above   0.037460    0.001529   0.039103      0.876130                                 West Bank and Gaza     xgboost    511.0
692   population ages 65 and above   0.178998    0.034611   0.186041     -1.803877                                 West Bank and Gaza  simple-rnn   3114.0
1403  population ages 65 and above   0.218639    0.049563   0.222626     -3.015103                                 West Bank and Gaza   base-lstm   3621.0
2114  population ages 65 and above   0.291526    0.089274   0.298787     -6.232157                                 West Bank and Gaza    base-gru   4554.0
2827          population ages 0-14   0.458164    0.328841   0.573447     -0.220203                                              World     xgboost   4997.0
3538          population ages 0-14   0.841468    0.922902   0.960678     -2.424531                                              World          rf   7437.0
2116          population ages 0-14   2.875225    9.315802   3.052180    -33.567341                                              World    base-gru  12348.0
1405          population ages 0-14   3.082486   10.818674   3.289175    -39.143918                                              World   base-lstm  12607.0
694           population ages 0-14   3.097395   10.819880   3.289359    -39.148394                                              World  simple-rnn  12612.0
2115         population ages 15-64   0.868508    1.019477   1.009692    -36.441553                                              World    base-gru   8645.0
2826         population ages 15-64   0.960332    1.169220   1.081305    -41.941049                                              World     xgboost   8956.0
3537         population ages 15-64   0.996928    1.475819   1.214833    -53.201275                                              World          rf   9390.0
693          population ages 15-64   1.244552    1.987100   1.409645    -71.978686                                              World  simple-rnn  10073.0
1404         population ages 15-64   1.561002    3.107542   1.762822   -113.128277                                              World   base-lstm  10940.0
3539  population ages 65 and above   0.293987    0.156249   0.395283      0.648749                                              World          rf   3424.0
2828  population ages 65 and above   1.071932    1.687253   1.298943     -2.792983                                              World     xgboost   8537.0
1406  population ages 65 and above   1.296822    1.792119   1.338701     -3.028725                                              World   base-lstm   8897.0
695   population ages 65 and above   1.445247    2.238180   1.496055     -4.031480                                              World  simple-rnn   9409.0
2117  population ages 65 and above   1.617599    2.803798   1.674455     -5.303002                                              World    base-gru   9897.0
2830          population ages 0-14   0.065973    0.009949   0.099744      0.964211                                        Yemen, Rep.     xgboost    585.0
1408          population ages 0-14   0.122926    0.020953   0.144751      0.924625                                        Yemen, Rep.   base-lstm   1153.0
2119          population ages 0-14   0.186819    0.048565   0.220374      0.825296                                        Yemen, Rep.    base-gru   1997.0
697           population ages 0-14   0.204637    0.055936   0.236508      0.798777                                        Yemen, Rep.  simple-rnn   2229.0
3541          population ages 0-14   0.301808    0.132646   0.364205      0.522826                                        Yemen, Rep.          rf   3400.0
3540         population ages 15-64   0.038095    0.002312   0.048079      0.992240                                        Yemen, Rep.          rf    180.0
2118         population ages 15-64   0.317064    0.102997   0.320931      0.654234                                        Yemen, Rep.    base-gru   3125.0
696          population ages 15-64   0.366510    0.139273   0.373193      0.532453                                        Yemen, Rep.  simple-rnn   3613.0
1407         population ages 15-64   0.383060    0.148968   0.385964      0.499904                                        Yemen, Rep.   base-lstm   3738.0
2829         population ages 15-64   0.373269    0.176820   0.420499      0.406405                                        Yemen, Rep.     xgboost   3930.0
2831  population ages 65 and above   0.021476    0.000486   0.022047     -0.323369                                        Yemen, Rep.     xgboost   1361.0
3542  population ages 65 and above   0.058991    0.006075   0.077941    -15.539051                                        Yemen, Rep.          rf   2993.0
1409  population ages 65 and above   0.433254    0.202889   0.450431   -551.382605                                        Yemen, Rep.   base-lstm   6673.0
698   population ages 65 and above   0.432334    0.204409   0.452116   -555.522930                                        Yemen, Rep.  simple-rnn   6674.0
2120  population ages 65 and above   0.488079    0.260005   0.509907   -706.886546                                        Yemen, Rep.    base-gru   7043.0
3544          population ages 0-14   0.161164    0.047853   0.218753      0.960752                                             Zambia          rf   1575.0
2833          population ages 0-14   0.310304    0.137387   0.370657      0.887316                                             Zambia     xgboost   3012.0
2122          population ages 0-14   1.828541    3.734120   1.932387     -2.062694                                             Zambia    base-gru   9942.0
1411          population ages 0-14   1.863710    3.852017   1.962656     -2.159392                                             Zambia   base-lstm  10031.0
700           population ages 0-14   2.130423    5.004998   2.237185     -3.105057                                             Zambia  simple-rnn  10614.0
2832         population ages 15-64   0.141662    0.032607   0.180573      0.973489                                             Zambia     xgboost   1301.0
3543         population ages 15-64   0.288503    0.118255   0.343882      0.903854                                             Zambia          rf   2745.0
2121         population ages 15-64   1.053893    1.227529   1.107939      0.001970                                             Zambia    base-gru   7304.0
1410         population ages 15-64   1.054558    1.231795   1.109863     -0.001499                                             Zambia   base-lstm   7312.0
699          population ages 15-64   1.312693    1.892441   1.375660     -0.538629                                             Zambia  simple-rnn   8374.0
2834  population ages 65 and above   0.018812    0.000507   0.022524      0.472930                                             Zambia     xgboost    937.0
3545  population ages 65 and above   0.071754    0.006398   0.079985     -5.646769                                             Zambia          rf   2645.0
701   population ages 65 and above   0.836092    0.767789   0.876236   -796.695401                                             Zambia  simple-rnn   8795.0
1412  population ages 65 and above   0.912622    0.901519   0.949484   -935.634682                                             Zambia   base-lstm   9082.0
2123  population ages 65 and above   0.928830    0.938027   0.968518   -973.564767                                             Zambia    base-gru   9161.0
2836          population ages 0-14   0.837793    0.820111   0.905600     -0.096843                                           Zimbabwe     xgboost   6608.0
3547          population ages 0-14   0.806041    1.167864   1.080678     -0.561940                                           Zimbabwe          rf   7179.0
2125          population ages 0-14   4.867365   24.613745   4.961224    -31.919241                                           Zimbabwe    base-gru  13377.0
1414          population ages 0-14   4.877394   24.742313   4.974165    -32.091192                                           Zimbabwe   base-lstm  13383.0
703           population ages 0-14   5.244046   28.690687   5.356369    -37.371879                                           Zimbabwe  simple-rnn  13482.0
2835         population ages 15-64   0.537533    0.337154   0.580650      0.329926                                           Zimbabwe     xgboost   4890.0
3546         population ages 15-64   0.747367    0.638162   0.798850     -0.268308                                           Zimbabwe          rf   6267.0
2124         population ages 15-64   3.987206   16.369771   4.045957    -31.533937                                           Zimbabwe    base-gru  13141.0
1413         population ages 15-64   4.088819   17.281889   4.157149    -33.346717                                           Zimbabwe   base-lstm  13186.0
702          population ages 15-64   4.472303   20.699068   4.549623    -40.138154                                           Zimbabwe  simple-rnn  13349.0
2837  population ages 65 and above   0.109986    0.015393   0.124067      0.439426                                           Zimbabwe     xgboost   1647.0
3548  population ages 65 and above   0.196606    0.063219   0.251433     -1.302329                                           Zimbabwe          rf   3363.0
704   population ages 65 and above   0.733130    0.593365   0.770302    -20.609487                                           Zimbabwe  simple-rnn   7655.0
1415  population ages 65 and above   0.762278    0.630388   0.793970    -21.957808                                           Zimbabwe   base-lstm   7769.0
2126  population ages 65 and above   0.920478    0.912323   0.955156    -32.225447                                           Zimbabwe    base-gru   8547.0
```


