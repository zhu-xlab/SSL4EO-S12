### Data collection
We use Google Earth Engine to automatically process and download data. The data collection code is modified from [SeCo](https://github.com/ServiceNow/seasonal-contrast).

#### Usage
Run `python ssl4eo_s12_downloader.py` or `bash download_data.sh`.

#### Main parameters
- `num_cities`: number of cities to sample from.
- `num_locations`: number of locations (geographical patches) in total.
- `std`: standard deviation of the sampling Gaussian distribution (km).
- `cloud_pct`: cloud percentage threshold, patches above which will be filtered out (%).
- `dates`: define reference dates to sample from.
- `crops/bands`: define the size of one patch and the bands to be downloaded.
