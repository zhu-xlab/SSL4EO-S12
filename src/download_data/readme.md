## Data collection
We use Google Earth Engine to automatically process and download data. The data collection code is modified from [SeCo](https://github.com/ServiceNow/seasonal-contrast).

### Update
v2 (`ssl4eo_downloader.py`):
- [x] extend sentinel-1/2 to other GEE available products

v1 (`ssl4eo_s12_downloader.py`):
- [x] speed up metadata collection
- [x] reorganize for better resuming
- [x] add support for overlap checking with rtree
- [x] add support to match existing locations (e.g. reproduce the locations of ssl4eo-s12)

### Usage
`ssl4eo_s12_downloader.py`: follow the comments at the beginning of the script.

`ssl4eo_downloader.py`: follow the comments at the beginning of the script.

- Example 1: Sample and download sentinel-2 L1C images with rtree/grid overlap search.
```
python ssl4eo_downloader.py \
    --save_path ./data \
    --collection COPERNICUS/S2 \
    --meta_cloud_name CLOUDY_PIXEL_PERCENTAGE \
    --cloud_pct 20 \
    --dates 2021-12-21 2021-09-22 2021-06-21 2021-03-20 \
    --radius 1320 \
    --bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12 \
    --crops 44 264 264 264 132 132 132 264 132 44 44 132 132 \
    --dtype uint16 \
    --num_workers 8 \
    --log_freq 100 \
    --overlap_check rtree \
    --indices_range 0 250000
```
- Example2: Download Landsat-8 images, match SSL4EO-S12 locations but keep same patch size.
```
python ssl4eo_downloader.py \
    --save_path ./data \
    --collection LANDSAT/LC08/C02/T1_TOA \
    --meta_cloud_name CLOUD_COVER \
    --cloud_pct 20 \
    --dates 2021-12-21 2021-09-22 2021-06-21 2021-03-20 \
    --radius 1980 \
    --bands B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 \
    --crops 132 132 132 132 132 132 132 264 264 132 132 \
    --dtype float32 \
    --num_workers 8 \
    --log_freq 100 \
    --match_file ./data/ssl4eo-s12_center_coords.csv \
    --indices_range 0 250000
```

