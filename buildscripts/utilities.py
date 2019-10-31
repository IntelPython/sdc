import platform
import re
import subprocess
import traceback

"""
Used for create list of packages required for build and test
from conda recipe
"""
def render_sdc_env(sdc_recipe, python='3.7', numpy='1.16'):
    build_env = []
    test_env  = []
    build_env_set = set()
    test_env_set  = set()
    if platform.system() == 'Windows':
        os_list = ['win', 'not unix']
    elif platform.system() == 'Linux':
        os_list = ['lin', 'unix', 'not win', 'not mac']
    elif platform.system() == 'Darwin':
        os_list = ['mac', 'osx', 'unix', 'not win', 'not lin']

    try:
        with open(sdc_recipe, 'r') as recipe:
            jinja_map = {}
            section = 'other'
            requirements_started = False

            for line in recipe:
                # Get jinja values
                # {% set ARROW_CPP_VERSION = "==0.15.0" %} or {% set ARROW_CPP_VERSION = '==0.15.0' %}
                if re.search(r"""{%\s+set\s+[\w-]+\s*=\s*"?'?==\d+\.\d*\.?\d*"?'?\s+%}""", line):
                    jinja_key = re.search(r"""{%\s+set\s+([\w-]+)\s*=\s*"?'?==\d+\.\d*\.?\d*"?'?\s+%}""", line).group(1)
                    jinja_val = re.search(r"""{%\s+set\s+[\w-]+\s*=\s*"?'?==(\d+\.\d*\.?\d*)"?'?\s+%}""", line).group(1)
                    jinja_map[jinja_key] = jinja_val

                # Check current recipe section
                if re.search(r"build:|run:|host:|test:|requires:", line):
                    section = re.search(r"build:|run:|host:|test:|requires:", line).group()
                    requirements_started = True
                    continue
                elif ':' in line:
                    requirements_started = False
                    continue

                # Get package with version (for <= or >= version is not set yet)
                # Search for "- python" or "- {{ pin_compatible('numpy') }}"
                if requirements_started and re.search(r"^\s+- ([\w-]+|{{ pin_compatible\('[\w-]+'\) }})", line):
                    # Get package name
                    if 'pin_compatible' in line:
                        package = re.search(r"^\s+- {{ pin_compatible\('([\w-]+)'\) }}", line).group(1)
                    else:
                        package = re.search(r"^\s+- ([\w-]+)", line).group(1)

                    # Get package version
                    package_version = None
                    if package == 'python':
                        package_version = python
                    if package == 'numpy':
                        package_version = numpy
                    # Search for "==0.46"
                    if re.search(r"==\d+\.\d*\.?\d*", line):
                        package_version = re.search(r"==(\d+\.\d*\.?\d*)", line).group(1)
                    # Search for jinja name like "{{ PYARROW_VERSION }}"
                    elif re.search(r"{{ \w+ }}", line):
                        jinja_version = re.search(r"{{ (\w+) }}", line).group(1)
                        if jinja_version in jinja_map:
                            package_version = jinja_map[jinja_version]

                    # Get os comments like "# [win]" or "# [not win]"
                    if re.search(r"#\s*\[\s*(\w+\s?\w*)\s*\]", line):
                        package_os = re.search(r"#\s*\[\s*(\w+\s*\w*)\s*\]", line).group(1)
                        package_os = ' '.join(package_os.split()) # remove multiple ' '
                        if package_os not in os_list:
                            continue

                    # Finally add package to build or test environment
                    if section in ['build:', 'host:', 'run:'] and package not in build_env_set:
                        build_env.append('{}{}'.format(package, '=' + package_version if package_version else ''))
                        build_env_set.add(package)
                    if section in ['run:', 'requires:'] and package not in test_env_set:
                        test_env.append('{}{}'.format(package, '=' + package_version if package_version else ''))
                        test_env_set.add(package)
    except:
        print('='*80)
        print('WARNING: Render environment for sdc from {} recipe failed'.format(sdc_recipe))
        print(traceback.format_exc())

    return {'build': build_env, 'test': test_env}


def run_command(command):
    print('='*80,  flush=True)
    print(command, flush=True)
    print('='*80,  flush=True)
    if platform.system() == 'Windows':
        subprocess.check_call(command, stdout=None, stderr=None, shell=True)
    else:
        subprocess.check_call(command, executable='/bin/bash', stdout=None, stderr=None, shell=True)


def create_conda_env(env_name, python='3.7', packages=None, channels=''):
    print('='*80)
    print('Setup conda {} environment'.format(env_name), flush=True)
    run_command('conda remove -y --name {} --all'.format(env_name))
    run_command('conda create -y -n {} python={}'.format(env_name, python))
    if packages:
        if platform.system() == 'Windows':
            run_command('activate {} && conda install -y {} {}'.format(env_name, ' '.join(packages), channels))
        else:
            run_command('source activate {} && conda install -y {} {}'.format(env_name, ' '.join(packages), channels))


def get_sdc_built_packages(build_output):
    if platform.system() == 'Windows':
        os_dir = 'win-64'
    elif platform.system() == 'Linux':
        os_dir = 'linux-64'
    elif platform.system() == 'Darwin':
        os_dir = 'osx-64'

    sdc_packages = []
    sdc_build_dir = os.path.join(build_output, os_dir)
    for item in os.listdir(sdc_build_dir):
        item_path = os.path.join(sdc_build_dir, item)
        if os.isfile(item_path) and re.search(r'^hpat.*\.tar\.bz2$|^hpat.*\.whl$'):
            sdc_packages.append(item_path)

    return sdc_packages
