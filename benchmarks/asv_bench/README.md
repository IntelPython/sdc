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
`cd <hpat_repo>/benchmarks/asv_bench`<br />
`asv run`<br />
Extra useful flags:<br />
`--quick` is used to run each benchmark only once and prevent saving of results<br />
`--show-stderr` is used to display the stderr output from the benchmarks

##### View results:
1. Console mode:
    1. Show benchmarked commits: `asv show`<br />
    2. Show benchmarks results for a commit: `asv show <commit>`<br />
2. Browser mode:
    1. Collect all results into a website: `asv publish`<br />
    2. Preview results using a local web server: `asv preview`<br />
    2. Open `http://<hostname>:<port>` in a browser, e.g. `http://127.0.0.1:8080`
