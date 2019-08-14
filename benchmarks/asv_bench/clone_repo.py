"""
Clone a git repo

Usage:
python clone_repo.py --url <repo_url> --local-path <local_repo_path> --branch <branch_name> --revision <revision>
"""
import argparse
import os
import shutil

from pathlib import Path

from git import Repo


def onerror(func, path_, _):
    """Error handler for shutil.rmtree"""
    if not os.access(path_, os.W_OK):
        import stat
        os.chmod(path_, stat.S_IWUSR)
        func(path_)
    else:
        raise PermissionError


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='Repo URL')
    parser.add_argument('--local-path', required=True, type=Path, help='Path to the local repo')
    parser.add_argument('--branch', default='master', help='Branch name')
    parser.add_argument('--revision', help='Revision')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_path.exists():
        shutil.rmtree(args.local_path, onerror=onerror)

    repo = Repo.clone_from(args.url, args.local_path)
    repo.git.checkout(args.branch)
    if args.revision:
        repo.git.reset('--hard', args.revision)


if __name__ == '__main__':
    main()
