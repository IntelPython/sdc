# HPAT Performance Tests
There are 2 ways to run performance testing.
The first one is via the custom runner `runner.py`. The second one is through ASV.

## 1. Custom run
#### Preparing:
1. Manually create Conda environment with installed required packages for HPAT,
activate the environment
2. Install `jinja2` for report generating: `conda install jinja2`
3. Build HPAT via `build_hpat.py` or manually:<br />
`python build_hpat.py --env-dir <activated_env_path> --build-dir <hpat_repo_path>`

#### Running
`python runner.py`

#### Report generating
`python asvgen.py --asv-results ../build/tests_perf --template templates/asvgen.html`

## 2. ASV run
#### Pipeline:
Provided by `asv.conf.json` file.
1. Create Conda environment with defined dependencies
2. Pull HPAT source code
3. Uninstall previous HPAT build
4. Build HPAT from source
5. Install HPAT
7. Run performance tests


#### Running
##### Install Airspeed Velocity in Anaconda Prompt:
`conda install -c conda-forge asv`

##### Run tests:
`cd <hpat_repo>/tests_perf`<br />
`asv run`<br />
Extra useful flags:<br />
`--quick` is used to run each test case only once and prevent saving of results<br />
`--show-stderr` is used to display the stderr output from the tests

##### View results:
1. Console mode:
    1. Show tested commits: `asv show`<br />
    2. Show tests results for a commit: `asv show <commit>`<br />
2. Browser mode:
    1. Collect all results into a website: `asv publish`<br />
    2. Preview results using a local web server: `asv preview`<br />
    2. Open `http://<hostname>:<port>` in a browser, e.g. `http://127.0.0.1:8080`
