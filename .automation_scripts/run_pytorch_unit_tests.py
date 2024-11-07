#!/usr/bin/env python3

""" The Python PyTorch testing script.
##
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import argparse
import os
import shutil
import subprocess
from subprocess import STDOUT, CalledProcessError

from collections import namedtuple
from datetime import datetime
from pathlib import Path
# Assume parse_xml_results.py exists and contains parse_xml_report function
from parse_xml_results import parse_xml_report
from pprint import pprint
from typing import Any, Dict, List

# Unit test status list
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

# Utility to apply exclusions to a list of tests
def apply_exclusions(test_list, exclusions):
    return [test for test in test_list if test not in exclusions]

def parse_xml_reports_as_dict(workflow_run_id, workflow_run_attempt, tag, workflow_name, path="."):
    test_cases = {}
    items_list = os.listdir(path)
    for dir in items_list:
        new_dir = os.path.join(path, dir)
        if os.path.isdir(new_dir):
            for xml_report in Path(new_dir).glob("**/*.xml"):
                test_cases.update(
                    parse_xml_report(
                        tag,
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                        workflow_name
                    )
                )
    return test_cases

def get_test_status(test_case):
    if "skipped" in test_case and test_case["skipped"]:
        type_message = test_case["skipped"]
        if "type" in type_message and type_message['type'] == "pytest.xfail":
            return "XFAILED"
        else:
            return "SKIPPED"
    elif "failure" in test_case and test_case["failure"]:
        return "FAILED"
    elif "error" in test_case and test_case["error"]:
        return "ERROR"
    else:
        return "PASSED"

def get_test_message(test_case, status=None):
    if status == "SKIPPED":
        return test_case.get("skipped", "")
    elif status == "FAILED":
        return test_case.get("failure", "")
    elif status == "ERROR":
        return test_case.get("error", "")
    else:
        return test_case.get("skipped", "") or test_case.get("failure", "") or test_case.get("error", "") or ""

def get_test_running_time(test_case):
    return test_case.get("time", "")

def summarize_xml_files(path, workflow_name):
    TOTAL_TEST_NUM = 0
    TOTAL_PASSED_NUM = 0
    TOTAL_SKIPPED_NUM = 0
    TOTAL_XFAIL_NUM = 0
    TOTAL_FAILED_NUM = 0
    TOTAL_ERROR_NUM = 0

    test_cases = parse_xml_reports_as_dict(-1, -1, 'testcase', workflow_name, path)
    test_file_and_status = namedtuple("test_file_and_status", ["file_name", "status"])
    res = {}
    res_item_list = ["PASSED", "SKIPPED", "XFAILED", "FAILED", "ERROR"]
    test_file_items = set()

    for (k, v) in list(test_cases.items()):
        file_name = k[0]
        if file_name not in test_file_items:
            test_file_items.add(file_name)
            for item in res_item_list:
                temp_item = test_file_and_status(file_name, item)
                res[temp_item] = {}
            temp_item_statistics = test_file_and_status(file_name, "STATISTICS")
            res[temp_item_statistics] = {'TOTAL': 0, 'PASSED': 0, 'SKIPPED': 0, 'XFAILED': 0, 'FAILED': 0, 'ERROR': 0}

    for (k, v) in list(test_cases.items()):
        file_name = k[0]
        class_name = k[1]
        test_name = k[2]
        combined_name = file_name + "::" + class_name + "::" + test_name
        test_status = get_test_status(v)
        test_running_time = get_test_running_time(v)
        test_message = get_test_message(v, test_status)
        test_info_value = ""
        test_tuple_key_status = test_file_and_status(file_name, test_status)
        test_tuple_key_statistics = test_file_and_status(file_name, "STATISTICS")
        TOTAL_TEST_NUM += 1
        res[test_tuple_key_statistics]["TOTAL"] += 1
        if test_status == "PASSED":
            test_info_value = str(test_running_time)
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["PASSED"] += 1
            TOTAL_PASSED_NUM += 1
        elif test_status == "SKIPPED":
            test_info_value = str(test_running_time)
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["SKIPPED"] += 1
            TOTAL_SKIPPED_NUM += 1
        elif test_status == "XFAILED":
            test_info_value = str(test_running_time)
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["XFAILED"] += 1
            TOTAL_XFAIL_NUM += 1
        elif test_status == "FAILED":
            test_info_value = test_message
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["FAILED"] += 1
            TOTAL_FAILED_NUM += 1
        elif test_status == "ERROR":
            test_info_value = test_message
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["ERROR"] += 1
            TOTAL_ERROR_NUM += 1

    statistics_dict = {
        "TOTAL": TOTAL_TEST_NUM,
        "PASSED": TOTAL_PASSED_NUM,
        "SKIPPED": TOTAL_SKIPPED_NUM,
        "XFAILED": TOTAL_XFAIL_NUM,
        "FAILED": TOTAL_FAILED_NUM,
        "ERROR": TOTAL_ERROR_NUM,
    }
    aggregate_item = workflow_name + "_aggregate"
    total_item = test_file_and_status(aggregate_item, "STATISTICS")
    res[total_item] = statistics_dict

    return res

def run_command_and_capture_output(cmd):
    try:
        print(f"Running command '{cmd}'")
        with open(CONSOLIDATED_LOG_FILE_PATH, "a+") as output_file:
            subprocess.run(cmd, shell=True, stdout=output_file, stderr=STDOUT, text=True, check=True)
    except CalledProcessError as e:
        print(f"ERROR: Cmd {cmd} failed with return code: {e.returncode}!")

def run_entire_tests(
    workflow_name,
    test_run_test_path,
    overall_logs_path_current_run,
    test_reports_src,
    exclude_tests,
    exclude_core
):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    exclude_list = exclude_tests.copy()

    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        if exclude_core:
            exclude_list.extend(DEFAULT_CORE_TESTS)
        exclude_args = ""
        if exclude_list:
            exclude_args = "--exclude " + " ".join(exclude_list)
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "default_xml_results_entire_tests/"
        )
        command = (
            f"python {test_run_test_path} {exclude_args} "
            "--exclude-jit-executor --exclude-distributed-tests --verbose"
        )
        run_command_and_capture_output(command)
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        if exclude_core:
            exclude_list.extend(DISTRIBUTED_CORE_TESTS)
        exclude_args = ""
        if exclude_list:
            exclude_args = "--exclude " + " ".join(exclude_list)
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "distributed_xml_results_entire_tests/"
        )
        command = (
            f"python {test_run_test_path} {exclude_args} "
            "--distributed-tests --verbose"
        )
        run_command_and_capture_output(command)
    elif workflow_name == "inductor":
        os.environ['TEST_CONFIG'] = 'inductor'
        exclude_args = ""
        if exclude_list:
            exclude_args = "--exclude " + " ".join(exclude_list)
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "inductor_xml_results_entire_tests/"
        )
        command = (
            f"python {test_run_test_path} {exclude_args} "
            "--inductor --verbose"
        )
        run_command_and_capture_output(command)

    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    entire_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)
    return entire_results_dict

def run_priority_tests(
    workflow_name,
    test_run_test_path,
    overall_logs_path_current_run,
    test_reports_src,
    exclude_tests,
    exclude_core
):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    test_suites = []

    if workflow_name == "default":
        test_suites = DEFAULT_CORE_TESTS.copy()
        if exclude_core:
            test_suites = []
        else:
            test_suites = apply_exclusions(test_suites, exclude_tests)
        if not test_suites:
            print("No default priority tests to run after applying exclusions.")
            return {}
        os.environ['TEST_CONFIG'] = 'default'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "default_xml_results_priority_tests/"
        )
        # Use run_test.py for tests execution
        command = (
            "python "
            + test_run_test_path
            + " --include "
            + " ".join(test_suites)
            + " --exclude-jit-executor --exclude-distributed-tests --verbose"
        )
        run_command_and_capture_output(command)
        del os.environ['HIP_VISIBLE_DEVICES']

    elif workflow_name == "distributed":
        test_suites = DISTRIBUTED_CORE_TESTS.copy()
        if exclude_core:
            test_suites = []
        else:
            test_suites = apply_exclusions(test_suites, exclude_tests)
        if not test_suites:
            print("No distributed priority tests to run after applying exclusions.")
            return {}
        os.environ['TEST_CONFIG'] = 'distributed'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "distributed_xml_results_priority_tests/"
        )
        # Use run_test.py for tests execution
        command = (
            "python "
            + test_run_test_path
            + " --include "
            + " ".join(test_suites)
            + " --distributed-tests --verbose"
        )
        run_command_and_capture_output(command)
        del os.environ['HIP_VISIBLE_DEVICES']

    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    priority_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)

    return priority_results_dict

def run_selected_tests(
    workflow_name,
    test_run_test_path,
    overall_logs_path_current_run,
    test_reports_src,
    selected_list,
    exclude_tests,
    exclude_core
):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""

    selected_list = selected_list.copy()
    if exclude_core:
        if workflow_name == "default":
            selected_list = apply_exclusions(selected_list, DEFAULT_CORE_TESTS)
        elif workflow_name == "distributed":
            selected_list = apply_exclusions(selected_list, DISTRIBUTED_CORE_TESTS)
    selected_list = apply_exclusions(selected_list, exclude_tests)

    if not selected_list:
        print(f"No tests to run for workflow '{workflow_name}' after applying exclusions.")
        return {}

    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "default_xml_results_selected_tests/"
        )
        # Use run_test.py for tests execution
        default_selected_test_suites = " ".join(selected_list)
        command = (
            "python "
            + test_run_test_path
            + " --include "
            + default_selected_test_suites
            + " --exclude-jit-executor --exclude-distributed-tests --verbose"
        )
        run_command_and_capture_output(command)
        del os.environ['HIP_VISIBLE_DEVICES']

    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "distributed_xml_results_selected_tests/"
        )
        # Use run_test.py for tests execution
        distributed_selected_test_suites = " ".join(selected_list)
        command = (
            "python "
            + test_run_test_path
            + " --include "
            + distributed_selected_test_suites
            + " --distributed-tests --verbose"
        )
        run_command_and_capture_output(command)
        del os.environ['HIP_VISIBLE_DEVICES']

    elif workflow_name == "inductor":
        os.environ['TEST_CONFIG'] = 'inductor'
        copied_logs_path = os.path.join(
            overall_logs_path_current_run, "inductor_xml_results_selected_tests/"
        )
        inductor_selected_test_suites = []
        non_inductor_selected_test_suites = []
        for item in selected_list:
            if "inductor/" in item:
                inductor_selected_test_suites.append(item)
            else:
                non_inductor_selected_test_suites.append(item)
        if not inductor_selected_test_suites and not non_inductor_selected_test_suites:
            print("No inductor tests to run after applying exclusions.")
            return {}
        if inductor_selected_test_suites:
            command = (
                "python "
                + test_run_test_path
                + " --include "
                + " ".join(inductor_selected_test_suites)
                + " --verbose"
            )
            run_command_and_capture_output(command)
        if non_inductor_selected_test_suites:
            command = (
                "python "
                + test_run_test_path
                + " --inductor --include "
                + " ".join(non_inductor_selected_test_suites)
                + " --verbose"
            )
            run_command_and_capture_output(command)

    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    selected_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)

    return selected_results_dict

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
    _environ = dict(os.environ)
    test_shell_path = os.path.join(pytorch_root_dir, ".ci/pytorch/test.sh")
    test_run_test_path = os.path.join(pytorch_root_dir, "test/run_test.py")
    repo_test_log_folder_path = os.path.join(pytorch_root_dir, ".automation_logs/")
    test_reports_src = os.path.join(pytorch_root_dir, "test/test-reports/")

    os.chdir(pytorch_root_dir)
    res_all_tests_dict = {}

    if not os.path.exists(repo_test_log_folder_path):
        os.mkdir(repo_test_log_folder_path)

    os.environ['CI'] = '1'
    os.environ['PYTORCH_TEST_WITH_ROCM'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['PYTORCH_TESTING_DEVICE_ONLY_FOR'] = 'cuda'
    os.environ['CONTINUE_THROUGH_ERROR'] = 'True'

    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    overall_logs_path_current_run = os.path.join(repo_test_log_folder_path, current_datetime)
    os.mkdir(overall_logs_path_current_run)

    global CONSOLIDATED_LOG_FILE_PATH
    CONSOLIDATED_LOG_FILE_PATH = os.path.join(overall_logs_path_current_run, CONSOLIDATED_LOG_FILE_NAME)

    if not priority_tests and not default_list and not distributed_list and not inductor_list:
        # Run entire tests for each workflow
        if not test_config:
            # default test process
            res_default_all = run_entire_tests(
                "default",
                test_run_test_path,
                overall_logs_path_current_run,
                test_reports_src,
                exclude_tests,
                exclude_default_core
            )
            res_all_tests_dict["default"] = res_default_all
            # distributed test process
            res_distributed_all = run_entire_tests(
                "distributed",
                test_run_test_path,
                overall_logs_path_current_run,
                test_reports_src,
                exclude_tests,
                exclude_distributed_core
            )
            res_all_tests_dict["distributed"] = res_distributed_all
            # inductor test process
            res_inductor_all = run_entire_tests(
                "inductor",
                test_run_test_path,
                overall_logs_path_current_run,
                test_reports_src,
                exclude_tests,
                False  # Assuming no core exclusion for inductor
            )
            res_all_tests_dict["inductor"] = res_inductor_all
        else:
            workflow_list = test_config
            if "default" in workflow_list:
                res_default_all = run_entire_tests(
                    "default",
                    test_run_test_path,
                    overall_logs_path_current_run,
                    test_reports_src,
                    exclude_tests,
                    exclude_default_core
                )
                res_all_tests_dict["default"] = res_default_all
            if "distributed" in workflow_list:
                res_distributed_all = run_entire_tests(
                    "distributed",
                    test_run_test_path,
                    overall_logs_path_current_run,
                    test_reports_src,
                    exclude_tests,
                    exclude_distributed_core
                )
                res_all_tests_dict["distributed"] = res_distributed_all
            if "inductor" in workflow_list:
                res_inductor_all = run_entire_tests(
                    "inductor",
                    test_run_test_path,
                    overall_logs_path_current_run,
                    test_reports_src,
                    exclude_tests,
                    False  # Assuming no core exclusion for inductor
                )
                res_all_tests_dict["inductor"] = res_inductor_all
    elif priority_tests:
        # Run priority tests for each workflow
        if test_config:
            workflow_list = test_config
        else:
            workflow_list = ["default", "distributed"]
        for workflow in workflow_list:
            if workflow == "default":
                res_default_priority = run_priority_tests(
                    "default",
                    test_run_test_path,
                    overall_logs_path_current_run,
                    test_reports_src,
                    exclude_tests,
                    exclude_default_core
                )
                res_all_tests_dict["default"] = res_default_priority
            elif workflow == "distributed":
                res_distributed_priority = run_priority_tests(
                    "distributed",
                    test_run_test_path,
                    overall_logs_path_current_run,
                    test_reports_src,
                    exclude_tests,
                    exclude_distributed_core
                )
                res_all_tests_dict["distributed"] = res_distributed_priority
            elif workflow == "inductor":
                print("Inductor priority tests cannot run since no core tests defined with inductor workflow.")
    else:
        # Run specified tests
        if default_list:
            res_default_selected = run_selected_tests(
                "default",
                test_run_test_path,
                overall_logs_path_current_run,
                test_reports_src,
                default_list,
                exclude_tests,
                exclude_default_core
            )
            res_all_tests_dict["default"] = res_default_selected
        if distributed_list:
            res_distributed_selected = run_selected_tests(
                "distributed",
                test_run_test_path,
                overall_logs_path_current_run,
                test_reports_src,
                distributed_list,
                exclude_tests,
                exclude_distributed_core
            )
            res_all_tests_dict["distributed"] = res_distributed_selected
        if inductor_list:
            res_inductor_selected = run_selected_tests(
                "inductor",
                test_run_test_path,
                overall_logs_path_current_run,
                test_reports_src,
                inductor_list,
                exclude_tests,
                False  # Assuming no core exclusion for inductor
            )
            res_all_tests_dict["inductor"] = res_inductor_selected

    # Restore environment variables
    os.environ.clear()
    os.environ.update(_environ)

    return res_all_tests_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Run PyTorch unit tests and generate xml results summary', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--test_config', nargs='+', default=[], type=str, help="Space-separated list of test workflows to be executed, e.g., 'default distributed'")
    parser.add_argument('--priority_tests', action='store_true', help="Run priority tests only")
    parser.add_argument('--default_list', nargs='+', default=[], help="Space-separated list of 'default' config test suites/files to be executed, e.g., 'test_weak test_dlpack'")
    parser.add_argument('--distributed_list', nargs='+', default=[], help="Space-separated list of 'distributed' config test suites/files to be executed, e.g., 'distributed/test_c10d_common distributed/test_c10d_nccl'")
    parser.add_argument('--inductor_list', nargs='+', default=[], help="Space-separated list of 'inductor' config test suites/files to be executed, e.g., 'inductor/test_torchinductor test_ops'")
    parser.add_argument('--exclude_tests', nargs='+', default=[], help="Space-separated list of individual test suites to exclude")
    parser.add_argument('--exclude_default_core', action='store_true', help="Exclude all default core test suites")
    parser.add_argument('--exclude_distributed_core', action='store_true', help="Exclude all distributed core test suites")
    parser.add_argument('--pytorch_root', default='.', type=str, help="PyTorch root directory")
    parser.add_argument('--example_output', type=str, help="{'workflow_name': {\n"
                                                           "  test_file_and_status(file_name='workflow_aggregate', status='STATISTICS'): {}, \n"
                                                           "  test_file_and_status(file_name='test_file_name_1', status='ERROR'): {}, \n"
                                                           "  test_file_and_status(file_name='test_file_name_1', status='FAILED'): {}, \n"
                                                           "  test_file_and_status(file_name='test_file_name_1', status='PASSED'): {}, \n"
                                                           "  test_file_and_status(file_name='test_file_name_1', status='SKIPPED'): {}, \n"
                                                           "  test_file_and_status(file_name='test_file_name_1', status='STATISTICS'): {} \n"
                                                           "}}\n")
    parser.add_argument('--example_usages', type=str, help="RUN ALL TESTS: python run_pytorch_unit_tests.py \n"
                                                           "RUN PRIORITY TESTS: python run_pytorch_unit_tests.py --test_config distributed --priority_tests \n"
                                                           "RUN SELECTED TESTS: python run_pytorch_unit_tests.py --default_list test_weak test_dlpack --inductor_list inductor/test_torchinductor")
    return parser.parse_args()

def main():
    global args
    args = parse_args()
    all_tests_results = run_test_and_summarize_results(
        args.pytorch_root,
        args.priority_tests,
        args.test_config,
        args.default_list,
        args.distributed_list,
        args.inductor_list,
        args.exclude_tests,
        args.exclude_default_core,
        args.exclude_distributed_core
    )
    pprint(dict(all_tests_results))

if __name__ == "__main__":
    main()

