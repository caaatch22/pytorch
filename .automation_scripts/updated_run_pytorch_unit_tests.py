import argparse
import os
import shutil
import subprocess
from subprocess import STDOUT, CalledProcessError

from collections import namedtuple
from datetime import datetime
from pathlib import Path
from parse_xml_results import parse_xml_report
from pprint import pprint
from typing import Any, Dict, List

# unit test status list
UT_STATUS_LIST = [
    "PASSED",
    "MISSED",
    "SKIPPED",
    "FAILED",
    "XFAILED",
    "ERROR"
]

DEFAULT_CORE_TESTS = [
    "test_nn",
    "test_torch",
    "test_cuda",
    "test_ops",
    "test_unary_ufuncs",
    "test_binary_ufuncs",
    "test_autograd"
]

DISTRIBUTED_CORE_TESTS = [
    "distributed/test_c10d_common",
    "distributed/test_c10d_nccl",
    "distributed/test_distributed_spawn"
]

CONSOLIDATED_LOG_FILE_NAME = "pytorch_unit_tests.log"

# Modify this function to apply exclusions
def apply_exclusions(test_list, exclusions):
    return [test for test in test_list if test not in exclusions]

def parse_xml_reports_as_dict(workflow_run_id, workflow_run_attempt, tag, workflow_name, path="."):
    # ... unchanged code ...

def get_test_status(test_case):
    # ... unchanged code ...

def get_test_message(test_case, status=None):
    # ... unchanged code ...

def get_test_running_time(test_case):
    # ... unchanged code ...

def summarize_xml_files(path, workflow_name):
    # ... unchanged code ...

def run_command_and_capture_output(cmd):
    # ... unchanged code ...

# Update test execution functions to respect exclusion flags
def run_priority_tests(workflow_name, test_run_test_path, overall_logs_path_current_run, test_reports_src, exclude_tests, exclude_core):
    # Apply exclusions to the appropriate test suites based on flags
    if workflow_name == "default":
        test_suites = apply_exclusions(DEFAULT_CORE_TESTS, exclude_tests)
        if exclude_core:
            test_suites = apply_exclusions(test_suites, DEFAULT_CORE_TESTS)
        # ... rest of function logic ...

    elif workflow_name == "distributed":
        test_suites = apply_exclusions(DISTRIBUTED_CORE_TESTS, exclude_tests)
        if exclude_core:
            test_suites = apply_exclusions(test_suites, DISTRIBUTED_CORE_TESTS)
        # ... rest of function logic ...

def run_selected_tests(workflow_name, test_run_test_path, overall_logs_path_current_run, test_reports_src, selected_list, exclude_tests, exclude_core):
    selected_list = apply_exclusions(selected_list, exclude_tests)
    if workflow_name == "default" and exclude_core:
        selected_list = apply_exclusions(selected_list, DEFAULT_CORE_TESTS)
    elif workflow_name == "distributed" and exclude_core:
        selected_list = apply_exclusions(selected_list, DISTRIBUTED_CORE_TESTS)
    # ... rest of function logic ...

# Main function to handle exclusions
def run_test_and_summarize_results(
    pytorch_root_dir: str,
    priority_tests: bool,
    test_config: List[str],
    default_list: List[str],
    distributed_list: List[str],
    inductor_list: List[str],
    exclude_tests: List[str],
    exclude_default_core: bool,
    exclude_distributed_core: bool
) -> Dict[str, Any]:
    # ... unchanged setup code ...

    # Run tests based on specified configurations
    if not priority_tests and not default_list and not distributed_list and not inductor_list:
        # ... unchanged code for running entire tests ...

    elif priority_tests and not default_list and not distributed_list and not inductor_list:
        if "default" in workflow_list:
            res_default_priority = run_priority_tests("default", test_run_test_path, overall_logs_path_current_run, test_reports_src, exclude_tests, exclude_default_core)
            res_all_tests_dict["default"] = res_default_priority
        if "distributed" in workflow_list:
            res_distributed_priority = run_priority_tests("distributed", test_run_test_path, overall_logs_path_current_run, test_reports_src, exclude_tests, exclude_distributed_core)
            res_all_tests_dict["distributed"] = res_distributed_priority
        # ... unchanged code ...

    # Run specified tests
    elif (default_list or distributed_list or inductor_list) and not test_config and not priority_tests:
        # Update selected test logic
        if default_list:
            res_default_selected = run_selected_tests("default", test_run_test_path, overall_logs_path_current_run, test_reports_src, default_list, exclude_tests, exclude_default_core)
            res_all_tests_dict["default"] = res_default_selected
        if distributed_list:
            res_distributed_selected = run_selected_tests("distributed", test_run_test_path, overall_logs_path_current_run, test_reports_src, distributed_list, exclude_tests, exclude_distributed_core)
            res_all_tests_dict["distributed"] = res_distributed_selected
        # ... unchanged code ...

    return res_all_tests_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Run PyTorch unit tests and generate xml results summary')
    parser.add_argument('--test_config', nargs='+', default=[], type=str, help="space-separated list of test workflows to be executed eg. 'default distributed'")
    parser.add_argument('--priority_tests', action='store_true', help="run priority tests only")
    parser.add_argument('--default_list', nargs='+', default=[], help="space-separated list of 'default' config test suites/files to be executed eg. 'test_weak test_dlpack'")
    parser.add_argument('--distributed_list', nargs='+', default=[], help="space-separated list of 'distributed' config test suites/files to be executed eg. 'distributed/test_c10d_common distributed/test_c10d_nccl'")
    parser.add_argument('--inductor_list', nargs='+', default=[], help="space-separated list of 'inductor' config test suites/files to be executed eg. 'inductor/test_torchinductor test_ops'")
    parser.add_argument('--exclude_tests', nargs='+', default=[], help="space-separated list of individual test suites to exclude")
    parser.add_argument('--exclude_default_core', action='store_true', help="exclude all default core test suites")
    parser.add_argument('--exclude_distributed_core', action='store_true', help="exclude all distributed core test suites")
    parser.add_argument('--pytorch_root', default='.', type=str, help="PyTorch root directory")
    return parser.parse_args()

def main():
    global args
    args = parse_args()
    all_tests_results = run_test_and_summarize_results(
        args.pytorch_root, args.priority_tests, args.test_config,
        args.default_list, args.distributed_list, args.inductor_list,
        args.exclude_tests, args.exclude_default_core, args.exclude_distributed_core
    )
    pprint(dict(all_tests_results))

if __name__ == "__main__":
    main()

