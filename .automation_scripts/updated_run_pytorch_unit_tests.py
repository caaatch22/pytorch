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

# Utility to apply exclusions to a list of tests
def apply_exclusions(test_list, exclusions):
    return [test for test in test_list if test not in exclusions]

def parse_xml_reports_as_dict(workflow_run_id, workflow_run_attempt, tag, workflow_name, path="."):
    test_cases = {}
    items_list = os.listdir(path)
    for dir in items_list:
        new_dir = path + '/' + dir + '/'
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
        if type_message.__contains__('type') and type_message['type'] == "pytest.xfail":
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
        return test_case["skipped"] if "skipped" in test_case else ""
    elif status == "FAILED":
        return test_case["failure"] if "failure" in test_case else ""
    elif status == "ERROR":
        return test_case["error"] if "error" in test_case else ""
    else:
        if "skipped" in test_case:
            return test_case["skipped"]
        elif "failure" in test_case:
            return test_case["failure"]
        elif "error" in test_case:
            return test_case["error"]
        else:
            return ""

def get_test_running_time(test_case):
    if test_case.__contains__('time'):
        return test_case["time"]
    return ""

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
            subprocess.run(cmd, shell=True, stdout=output_file, stderr=STDOUT, text=True)
    except CalledProcessError as e:
        print(f"ERROR: Cmd {cmd} failed with return code: {e.returncode}!")

def run_priority_tests(workflow_name, test_run_test_path, overall_logs_path_current_run, test_reports_src, exclude_tests, exclude_core):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    test_suites = []

    if workflow_name == "default":
        test_suites = apply_exclusions(DEFAULT_CORE_TESTS, exclude_tests)
        if exclude_core:
            test_suites = apply_exclusions(test_suites, DEFAULT_CORE_TESTS)
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_priority_tests/"
        command = "python " + test_run_test_path + " --include " + " ".join(test_suites) + " --exclude-jit-executor --exclude-distributed-tests --verbose"
        run_command_and_capture_output(command)

    elif workflow_name == "distributed":
        test_suites = apply_exclusions(DISTRIBUTED_CORE_TESTS, exclude_tests)
        if exclude_core:
            test_suites = apply_exclusions(test_suites, DISTRIBUTED_CORE_TESTS)
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_priority_tests/"
        command = "python " + test_run_test_path + " --include " + " ".join(test_suites) + " --distributed-tests --verbose"
        run_command_and_capture_output(command)

    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    return summarize_xml_files(copied_logs_path_destination, workflow_name)

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
    test_shell_path = pytorch_root_dir + "/.ci/pytorch/test.sh"
    test_run_test_path = pytorch_root_dir + "/test/run_test.py"
    repo_test_log_folder_path = pytorch_root_dir + "/.automation_logs/"
    test_reports_src = pytorch_root_dir + "/test/test-reports/"

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
    overall_logs_path_current_run = repo_test_log_folder_path + current_datetime + "/"
    os.mkdir(overall_logs_path_current_run)

    global CONSOLIDATED_LOG_FILE_PATH
    CONSOLIDATED_LOG_FILE_PATH = overall_logs_path_current_run + CONSOLIDATED_LOG_FILE_NAME

    if priority_tests:
        if "default" in test_config:
            res_default_priority = run_priority_tests("default", test_run_test_path, overall_logs_path_current_run, test_reports_src, exclude_tests, exclude_default_core)
            res_all_tests_dict["default"] = res_default_priority
        if "distributed" in test_config:
            res_distributed_priority = run_priority_tests("distributed", test_run_test_path, overall_logs_path_current_run, test_reports_src, exclude_tests, exclude_distributed_core)
            res_all_tests_dict["distributed"] = res_distributed_priority
    else:
        raise Exception("Invalid test configurations!")

    os.environ.clear()
    os.environ.update(_environ)

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

