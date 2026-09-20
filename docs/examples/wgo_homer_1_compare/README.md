# WGO HomER Sample Compare: homer_1

This folder contains the gold and predicted annotations for one WGO HomER
sample. The original video is not redistributed; obtain it from WGO-Bench with
`scripts/prepare_wgo_bench.py`.

## Files

- `gold_homer_1.json`: WGO gold annotation
- `pred_homer_1_candidate_selector.json`: our Ver 5.1 HomER candidate-selector prediction

## Summary

- Video id: `homer_1`
- Episode instruction: use pitcher to pour water into tall glass and wine glass
- Gold segments: 20
- Predicted segments: 17
- Greedy temporal overlaps: 15 matched pairs, 8 with IoU >= 0.75

## Gold Segments

| # | start | end | subtask |
|---:|---:|---:|---|
| 0 | 0.00 | 7.71 | twist open the pitcher lid |
| 1 | 7.71 | 10.84 | pick up the pitcher |
| 2 | 10.84 | 20.75 | pour water from the pitcher into the wine glass |
| 3 | 20.75 | 21.71 | place the pitcher on the black board right next to wine glass |
| 4 | 21.71 | 26.51 | twist the pitcher lid to close it |
| 5 | 26.51 | 27.86 | pick up the pitcher |
| 6 | 27.86 | 62.59 | pour water from the pitcher into the wine glass |
| 7 | 62.59 | 65.41 | move the tall glass closer to the wine glass |
| 8 | 65.41 | 67.99 | place the pitcher on the black board right and below the wine glass |
| 9 | 67.99 | 72.94 | twist the pitcher lid to open it slightly |
| 10 | 72.94 | 74.36 | pick up the pitcher |
| 11 | 74.36 | 79.97 | pour a bit of water from the pitcher into the tall glass |
| 12 | 79.97 | 83.20 | move the glass wine 7cm to the left |
| 13 | 83.20 | 134.02 | pour water from the pitcher into the tall glass until it's almost full |
| 14 | 134.02 | 137.13 | pour water from the pitcher into the wine glass until it's full |
| 15 | 137.13 | 140.22 | pour water from the pitcher into the tall glass until it's full |
| 16 | 140.22 | 142.20 | place the pitcher on the table |
| 17 | 142.20 | 147.53 | reposition the wine glass to the left center part of table |
| 18 | 147.53 | 152.14 | move the tall glass to the right-center part of the table |
| 19 | 152.14 | 161.18 | place the pitcher on the center part of the table between the two glasses |

## Our Predicted Segments

| # | start | end | source | subtask |
|---:|---:|---:|---|---|
| 0 | 0.00 | 10.00 | A | Close the pitcher lid |
| 1 | 10.00 | 13.00 | A | pour water into wine glass |
| 2 | 13.00 | 21.00 | A | pour water into wine glass |
| 3 | 21.00 | 28.00 | B | close the pitcher lid |
| 4 | 28.00 | 62.00 | A | Pour water into wine glass |
| 5 | 62.00 | 67.00 | A | Place the pitcher on the table |
| 6 | 67.00 | 73.00 | C | close pitcher lid |
| 7 | 73.00 | 74.00 | A | move glass on table |
| 8 | 74.00 | 74.50 | A | Pour water into wine glass |
| 9 | 74.50 | 75.00 | A | Pour water into wine glass |
| 10 | 75.00 | 80.00 | A | pour water into wine glass |
| 11 | 80.00 | 84.00 | A | pour water into wine glass |
| 12 | 84.00 | 139.50 | A | Pour water from pitcher into tall glass |
| 13 | 139.50 | 141.00 | A | pour water into tall glass |
| 14 | 141.00 | 142.50 | A | place the pitcher on the table |
| 15 | 142.50 | 152.50 | A | place the wine glass on the table |
| 16 | 152.50 | 161.18 | A | move pitcher to center of table |

## Temporal Match View

Greedy matching is only for quick inspection; the official score also uses semantic judging.

| IoU | pred # | gold # | pred subtask | gold subtask |
|---:|---:|---:|---|---|
| 0.979 | 4 | 6 | Pour water into wine glass | pour water from the pitcher into the wine glass |
| 0.960 | 16 | 19 | move pitcher to center of table | place the pitcher on the center part of the table between the two glasses |
| 0.889 | 12 | 13 | Pour water from pitcher into tall glass | pour water from the pitcher into the tall glass until it's almost full |
| 0.880 | 10 | 11 | pour water into wine glass | pour a bit of water from the pitcher into the tall glass |
| 0.826 | 6 | 9 | close pitcher lid | twist the pitcher lid to open it slightly |
| 0.795 | 11 | 12 | pour water into wine glass | move the glass wine 7cm to the left |
| 0.771 | 0 | 0 | Close the pitcher lid | twist open the pitcher lid |
| 0.762 | 2 | 2 | pour water into wine glass | pour water from the pitcher into the wine glass |
| 0.708 | 7 | 10 | move glass on table | pick up the pitcher |
| 0.686 | 3 | 4 | close the pitcher lid | twist the pitcher lid to close it |
| 0.565 | 5 | 7 | Place the pitcher on the table | move the tall glass closer to the wine glass |
| 0.526 | 14 | 16 | place the pitcher on the table | place the pitcher on the table |
| 0.489 | 15 | 17 | place the wine glass on the table | reposition the wine glass to the left center part of table |
| 0.186 | 13 | 15 | pour water into tall glass | pour water from the pitcher into the tall glass until it's full |
| 0.158 | 1 | 1 | pour water into wine glass | pick up the pitcher |
