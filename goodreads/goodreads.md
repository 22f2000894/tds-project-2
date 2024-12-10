## Summary Statistics
|        |   book_id |   goodreads_book_id |     best_book_id |          work_id |   books_count |           isbn |         isbn13 | authors      |   original_publication_year | original_title   | title          | language_code   |   average_rating |    ratings_count |   work_ratings_count |   work_text_reviews_count |   ratings_1 |   ratings_2 |   ratings_3 |      ratings_4 |       ratings_5 | image_url                                                                                | small_image_url                                                                        |
|:-------|----------:|--------------------:|-----------------:|-----------------:|--------------:|---------------:|---------------:|:-------------|----------------------------:|:-----------------|:---------------|:----------------|-----------------:|-----------------:|---------------------:|--------------------------:|------------:|------------:|------------:|---------------:|----------------:|:-----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| count  |   7860    |      7860           |   7860           |   7860           |     7860      | 7860           | 7860           | 7860         |                    7860     | 7860             | 7860           | 7860            |      7860        |   7860           |       7860           |                   7860    |     7860    |     7860    |      7860   |  7860          |  7860           | 7860                                                                                     | 7860                                                                                   |
| unique |    nan    |       nan           |    nan           |    nan           |      nan      | 7860           |  nan           | 3757         |                     nan     | 7755             | 7841           | 23              |       nan        |    nan           |        nan           |                    nan    |      nan    |      nan    |       nan   |   nan          |   nan           | 5269                                                                                     | 5269                                                                                   |
| top    |    nan    |       nan           |    nan           |    nan           |      nan      |    6.17115e+07 |  nan           | Nora Roberts |                     nan     | The Gift         | Selected Poems | eng             |       nan        |    nan           |        nan           |                    nan    |      nan    |      nan    |       nan   |   nan          |   nan           | https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042a9c91a29c1d680899eff700a03.png | https://s.gr-assets.com/assets/nophoto/book/50x75-a91bf249278a81aabab721ef782c4a74.png |
| freq   |    nan    |       nan           |    nan           |    nan           |      nan      |    1           |  nan           | 58           |                     nan     | 4                | 3              | 5557            |       nan        |    nan           |        nan           |                    nan    |      nan    |      nan    |       nan   |   nan          |   nan           | 2592                                                                                     | 2592                                                                                   |
| mean   |   4728.39 |         4.53775e+06 |      4.72388e+06 |      7.55004e+06 |       83.1024 |  nan           |    9.77469e+12 | nan          |                    1980.28  | nan              | nan            | nan             |         3.9954   |  61174.8         |      67493.5         |                   3227.1  |     1533.5  |     3533.42 |     12972.8 | 22601.3        | 26852.5         | nan                                                                                      | nan                                                                                    |
| std    |   2889.67 |         7.03925e+06 |      7.27029e+06 |      1.08211e+07 |      180.049  |  nan           |    2.39655e+11 | nan          |                     161.467 | nan              | nan            | nan             |         0.250907 | 175131           |     186596           |                   6682.34 |     7427.93 |    10800.5  |     31576.8 | 57039.8        | 89048.5         | nan                                                                                      | nan                                                                                    |
| min    |      1    |         1           |      1           |     87           |        1      |  nan           |    1.9517e+08  | nan          |                   -1750     | nan              | nan            | nan             |         2.47     |   2773           |       6323           |                     11    |       11    |       30    |       323   |   872          |   754           | nan                                                                                      | nan                                                                                    |
| 25%    |   2183.75 |     40212.8         |  41700.2         | 987344           |       27      |  nan           |    9.78032e+12 | nan          |                    1989     | nan              | nan            | nan             |         3.84     |  14270.8         |      16222.2         |                    751    |      201    |      690    |      3300   |  5730          |  5542           | nan                                                                                      | nan                                                                                    |
| 50%    |   4604.5  |    284569           | 298972           |      2.48895e+06 |       44      |  nan           |    9.78045e+12 | nan          |                    2004     | nan              | nan            | nan             |         4.01     |  22838           |      25518.5         |                   1498    |      421    |     1257    |      5353.5 |  9063          |  9313           | nan                                                                                      | nan                                                                                    |
| 75%    |   7188.5  |         7.35282e+06 |      7.74729e+06 |      1.08458e+07 |       72      |  nan           |    9.78081e+12 | nan          |                    2010     | nan              | nan            | nan             |         4.17     |  46252.5         |      51445           |                   3084.25 |      992.25 |     2697    |     10532.8 | 18121.8        | 19234.2         | nan                                                                                      | nan                                                                                    |
| max    |   9999    |         3.20757e+07 |      3.55342e+07 |      5.63996e+07 |     3455      |  nan           |    9.79001e+12 | nan          |                    2017     | nan              | nan            | nan             |         4.82     |      4.78065e+06 |          4.94236e+06 |                 155254    |   456191    |   436802    |    793319   |     1.4813e+06 |     3.01154e+06 | nan                                                                                      | nan                                                                                    |

## Correlation Matrix
|                           |     book_id |   goodreads_book_id |   best_book_id |     work_id |   books_count |      isbn13 |   original_publication_year |   average_rating |   ratings_count |   work_ratings_count |   work_text_reviews_count |   ratings_1 |   ratings_2 |   ratings_3 |   ratings_4 |   ratings_5 |
|:--------------------------|------------:|--------------------:|---------------:|------------:|--------------:|------------:|----------------------------:|-----------------:|----------------:|---------------------:|--------------------------:|------------:|------------:|------------:|------------:|------------:|
| book_id                   |  1          |          0.082336   |     0.0693961  |  0.0816829  |   -0.26714    | -0.00051535 |                  0.0430661  |       -0.045711  |     -0.374031   |          -0.383455   |               -0.422065   | -0.239569   |  -0.347115  | -0.415145   | -0.408967   | -0.332248   |
| goodreads_book_id         |  0.082336   |          1          |     0.962863   |  0.928328   |   -0.154726   |  0.00635996 |                  0.124302   |       -0.0474934 |     -0.0657916  |          -0.0560746  |                0.134068   | -0.0331596  |  -0.0468961 | -0.0639239  | -0.0543049  | -0.0515943  |
| best_book_id              |  0.0693961  |          0.962863   |     1          |  0.894735   |   -0.148131   |  0.00679723 |                  0.12176    |       -0.0458066 |     -0.0611632  |          -0.0466456  |                0.142884   | -0.0278028  |  -0.0380566 | -0.053405   | -0.0436094  | -0.0439364  |
| work_id                   |  0.0816829  |          0.928328   |     0.894735   |  1          |   -0.102062   |  0.00758807 |                  0.0983509  |       -0.0436458 |     -0.0538983  |          -0.045459   |                0.111752   | -0.0285435  |  -0.0410592 | -0.0541192  | -0.0442112  | -0.0403852  |
| books_count               | -0.26714    |         -0.154726   |    -0.148131   | -0.102062   |    1          |  0.00916626 |                 -0.319806   |       -0.0701593 |      0.332199   |           0.340957   |                0.200861   |  0.230763   |   0.343544  |  0.393278   |  0.357453   |  0.285114   |
| isbn13                    | -0.00051535 |          0.00635996 |     0.00679723 |  0.00758807 |    0.00916626 |  1          |                 -0.00333179 |       -0.0157096 |      0.00496842 |           0.00534145 |                0.00941404 |  0.00328531 |   0.0053486 |  0.00640804 |  0.00605516 |  0.00411898 |
| original_publication_year |  0.0430661  |          0.124302   |     0.12176    |  0.0983509  |   -0.319806   | -0.00333179 |                  1          |        0.019763  |     -0.0232961  |          -0.0242714  |                0.0284117  | -0.0189407  |  -0.0385001 | -0.0422101  | -0.0249395  | -0.0136669  |
| average_rating            | -0.045711   |         -0.0474934  |    -0.0458066  | -0.0436458  |   -0.0701593  | -0.0157096  |                  0.019763   |        1         |      0.0535073  |           0.053786   |                0.0132802  | -0.0785778  |  -0.11388   | -0.058667   |  0.0464181  |  0.124142   |
| ratings_count             | -0.374031   |         -0.0657916  |    -0.0611632  | -0.0538983  |    0.332199   |  0.00496842 |                 -0.0232961  |        0.0535073 |      1          |           0.995099   |                0.784033   |  0.720123   |   0.843333  |  0.934744   |  0.978912   |  0.964309   |
| work_ratings_count        | -0.383455   |         -0.0560746  |    -0.0466456  | -0.045459   |    0.340957   |  0.00534145 |                 -0.0242714  |        0.053786  |      0.995099   |           1          |                0.81109    |  0.715683   |   0.845973  |  0.94076    |  0.98779    |  0.966809   |
| work_text_reviews_count   | -0.422065   |          0.134068   |     0.142884   |  0.111752   |    0.200861   |  0.00941404 |                  0.0284117  |        0.0132802 |      0.784033   |           0.81109    |                1          |  0.57432    |   0.69972   |  0.766864   |  0.821609   |  0.768602   |
| ratings_1                 | -0.239569   |         -0.0331596  |    -0.0278028  | -0.0285435  |    0.230763   |  0.00328531 |                 -0.0189407  |       -0.0785778 |      0.720123   |           0.715683   |                0.57432    |  1          |   0.926456  |  0.79474    |  0.669565   |  0.593181   |
| ratings_2                 | -0.347115   |         -0.0468961  |    -0.0380566  | -0.0410592  |    0.343544   |  0.0053486  |                 -0.0385001  |       -0.11388   |      0.843333   |           0.845973   |                0.69972    |  0.926456   |   1         |  0.948941   |  0.83528    |  0.702581   |
| ratings_3                 | -0.415145   |         -0.0639239  |    -0.053405   | -0.0541192  |    0.393278   |  0.00640804 |                 -0.0422101  |       -0.058667  |      0.934744   |           0.94076    |                0.766864   |  0.79474    |   0.948941  |  1          |  0.952365   |  0.825276   |
| ratings_4                 | -0.408967   |         -0.0543049  |    -0.0436094  | -0.0442112  |    0.357453   |  0.00605516 |                 -0.0249395  |        0.0464181 |      0.978912   |           0.98779    |                0.821609   |  0.669565   |   0.83528   |  0.952365   |  1          |  0.934432   |
| ratings_5                 | -0.332248   |         -0.0515943  |    -0.0439364  | -0.0403852  |    0.285114   |  0.00411898 |                 -0.0136669  |        0.124142  |      0.964309   |           0.966809   |                0.768602   |  0.593181   |   0.702581  |  0.825276   |  0.934432   |  1          |