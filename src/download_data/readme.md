### Data collection
We use Google Earth Engine to automatically process and download data. The data collection code is modified from [SeCo](https://github.com/ServiceNow/seasonal-contrast).

Update:
- speed up metadata collection
- reorganize for better resuming
- add support for overlap checking with rtree
- add support to match existing locations (e.g. reproduce the locations of ssl4eo-s12)

#### Usage
option 1: match ssl4eo-s12, and fill unmatched locations with newly sampled locations
```
# match ssl4eo-s12 ids, unavailable ids skip
!python ssl4eo_s12_downloader.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --match_file ssl4eo-s12_coords_v1.csv --indices_range 0 250000

# fill unmatched ids with rtree overlap search
!python ssl4eo_s12_downloader.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --resume ./data/checked_locations.csv --overlap_check rtree --indices_range 0 250000
```
option 2: resample new locations
```
# (op1) resample new ids with rtree overlap search
!python ssl4eo_s12_downloader.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --overlap_check rtree --indices_range 0 250000

# (op2) resample new ids with grid overlap search
!python ssl4eo_s12_downloader.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --overlap_check grid --indices_range 0 250000
```

(optional) resume from interruption
```
!python ssl4eo_s12_downloader.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --resume ./data/checked_locations.csv --overlap_check rtree --indices_range 0 250000
```

#### Main parameters
- `num_cities`: number of cities to sample from.
- `num_locations`: number of locations in total.
- `std`: standard deviation of the sampling Gaussian distribution (km).
- `cloud_pct`: cloud percentage threshold, patches above which will be filtered out (%).
- `dates`: define reference dates to sample from.
- `radius/crops/bands`: define the size of one patch and the bands to be downloaded.

#### Download other satellite data
Some items to modify:
- bands/crops/radius
- GEE collection name
- img dtype
- date and period
- cloud mask


