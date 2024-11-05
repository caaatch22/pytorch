import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple, List

# Backends list
BACKENDS_LIST = [
    "dist-gloo",
    "dist-nccl"
]

TARGET_WORKFLOW = "--rerun-disabled-tests"

# Default and distributed test suite exclusions
DEFAULT_EXCLUDED_SUITES = [
    "test_nn", "test_torch", "test_cuda", "test_ops",
    "test_unary_ufuncs", "test_binary_ufuncs", "test_autograd"
]

DISTRIBUTED_EXCLUDED_SUITES = [
    "distributed/test_c10d_common", "distributed/test_c10d_nccl",
    "distributed/test_distributed_spawn"
]

# Exclusion flags
EXCLUDE_SUITES: List[str] = []  # User-defined exclusion list
EXCLUDE_DEFAULT = False         # Flag to exclude default suites
EXCLUDE_DISTRIBUTED = False     # Flag to exclude distributed suites

def get_job_id(report: Path) -> int:
    try:
        return int(report.parts[0].rpartition("_")[2])
    except ValueError:
        return -1

def is_rerun_disabled_tests(root: ET.ElementTree) -> bool:
    skipped = root.find(".//*skipped")
    if skipped is None:
        return False

    message = skipped.attrib.get("message", "")
    return TARGET_WORKFLOW in message or "num_red" in message

def should_exclude(test_suite_name: str) -> bool:
    """Determines if a test suite should be excluded based on exclusion flags."""
    if test_suite_name in EXCLUDE_SUITES:
        return True
    if EXCLUDE_DEFAULT and test_suite_name in DEFAULT_EXCLUDED_SUITES:
        return True
    if EXCLUDE_DISTRIBUTED and test_suite_name in DISTRIBUTED_EXCLUDED_SUITES:
        return True
    return False

def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
    work_flow_name: str
) -> Dict[Tuple[str], Dict[str, Any]]:
    print(f"Parsing {tag}s for test report: {report}")

    job_id = get_job_id(report)
    print(f"Found job id: {job_id}")

    test_cases: Dict[Tuple[str], Dict[str, Any]] = {}

    root = ET.parse(report)
    if is_rerun_disabled_tests(root):
        return test_cases

    for test_case in root.iter(tag):
        test_suite_name = report.stem  # Assume file name as suite name
        if should_exclude(test_suite_name):
            print(f"Excluding test suite: {test_suite_name}")
            continue

        case = process_xml_element(test_case)
        if tag == 'testcase':
            case["workflow_id"] = workflow_id
            case["workflow_run_attempt"] = workflow_run_attempt
            case["job_id"] = job_id
            case["work_flow_name"] = work_flow_name

            case_name = report.parent.name
            for ind in range(len(BACKENDS_LIST)):
                if BACKENDS_LIST[ind] in report.parts:
                    case_name = case_name + "_" + BACKENDS_LIST[ind]
                    break
            case["invoking_file"] = case_name
            test_cases[(case["invoking_file"], case["classname"], case["name"], case["work_flow_name"])] = case
        elif tag == 'testsuite':
            case["work_flow_name"] = work_flow_name
            case["invoking_file"] = report.parent.name
            case["invoking_xml"] = report.name
            case["running_time_xml"] = case["time"]
            test_cases[(case["invoking_file"], case["invoking_xml"], case["work_flow_name"])] = case

    return test_cases

def process_xml_element(element: ET.Element) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    ret.update(element.attrib)

    for k, v in ret.items():
        try:
            ret[k] = int(v)
        except ValueError:
            pass
        try:
            ret[k] = float(v)
        except ValueError:
            pass

    if element.text and element.text.strip():
        ret["text"] = element.text
    if element.tail and element.tail.strip():
        ret["tail"] = element.tail

    for child in element:
        if child.tag not in ret:
            ret[child.tag] = process_xml_element(child)
        else:
            if not isinstance(ret[child.tag], list):
                ret[child.tag] = [ret[child.tag]]
            ret[child.tag].append(process_xml_element(child))
    return ret

