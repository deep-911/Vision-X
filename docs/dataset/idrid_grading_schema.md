# IDRiD Grading Dataset Schema

## Columns
Training Set Columns: ['Image name', 'Retinopathy grade', 'Risk of macular edema ', 'Unnamed: 11']
Testing Set Columns: ['Image name', 'Retinopathy grade', 'Risk of macular edema ']

## Data Distribution
### Training Set (Total: 413)
```
Retinopathy grade
0    134
1     20
2    136
3     74
4     49
```

### Testing Set (Total: 103)
```
Retinopathy grade
0    34
1     5
2    32
3    19
4    13
```

## Duplicates / Patient Info
- Unique Image Names in Train: 413 / 413
- Unique Image Names in Test: 103 / 103
- Overlap between Train and Test: 103
- Patient Metadata: None provided in these CSVs.

## Image Directories
- `Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/` (.jpg)
- `Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set/` (.jpg)
