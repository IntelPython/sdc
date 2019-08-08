# HPAT Benchmarks
Benchmarking with Airspeed Velocity.

### Pipeline:
The pipeline is provided by `asv.conf.json` file.
1. Create Conda environment with defined dependencies
2. Pull HPAT source code
3. Uninstall previous HPAT build
4. Build HPAT from source
5. Install HPAT
6. Clone Pandas repo to inherit Pandas benchmarks in HPAT benchmarks
7. Run benchmarks


### Running
##### Install Airspeed Velocity in Anaconda Prompt:
`conda install -c conda-forge asv`

##### Run benchmarking:
`cd asv_bench`<br />
`asv run`

##### View results locally:
1. Run below commands:<br />
`asv publish`<br />
`asv preview`<br />
2. Open `http://127.0.0.1:8080` in a browser.
