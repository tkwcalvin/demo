# This file contains code copied and modified from the following repository:
# Repository: https://github.com/huangd1999/AgentCoder/tree/main
# Original Author: Dong Huang, Jie M.Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, Heming Cui
# License: MIT

"""
AgentFramework Executor Module
==============================

这个模块实现了AgentCoder框架的执行器代理组件。
执行器代理负责通过以下方式测试和改进生成的代码：
1. 对生成的测试用例运行代码
2. 从多个候选中识别最佳代码完成方案
3. 通过错误修复循环迭代改进代码
4. 管理整个AgentCoder工作流程

This module implements the executor agent component of the AgentCoder framework.
The executor agent is responsible for testing and improving the generated code by:
1. Running code against generated test cases
2. Identifying the best code completion from multiple candidates
3. Iteratively improving code through bug fixing cycles
4. Managing the overall AgentCoder workflow

核心功能 (Key Features):
- 在超时保护下执行Python代码
- 测试多个代码完成方案与多个测试用例的组合
- 选择性能最佳的代码完成方案
- 实现迭代改进循环
- 处理函数名映射（candidate vs 原始entry_point）
- 提供全面的测试报告

- Executes Python code with timeout protection
- Tests multiple code completions against multiple test cases
- Selects the best performing code completion
- Implements iterative improvement cycles
- Handles function name mapping (candidate vs original entry_point)
- Provides comprehensive test reporting

主要函数 (Main Functions):
- process_humaneval_test(): 准备测试执行环境
- preprocess_data(): 预处理任务数据，清理代码块标记
- test_report(): 生成测试报告，统计通过率
- test_agent_concurrency(): 并发测试所有代码-测试组合
- executor_main(): 协调整个执行和改进过程
- 各种用于代码预处理和执行的实用函数

- process_humaneval_test(): Prepares test execution environment
- preprocess_data(): Preprocesses task data, cleans code block markers
- test_report(): Generates test reports, calculates pass rates
- test_agent_concurrency(): Tests all code-test combinations concurrently
- executor_main(): Orchestrates the entire execution and improvement process
- Various utility functions for code preprocessing and execution

工作流程 (Workflow):
1. 加载数据集 -> 2. 并发测试选择最佳方案 -> 3. 生成测试报告
4. 调用程序员代理改进代码 -> 5. 调用设计师代理改进测试用例
6. 保存中间结果 -> 7. 再次测试和报告

1. Load dataset -> 2. Concurrent testing to select best solution -> 3. Generate test report
4. Call programmer agent to improve code -> 5. Call designer agent to improve test cases
6. Save intermediate results -> 7. Test and report again
"""

# 标准库导入
import json  # JSON数据处理
import random  # 随机数生成
from typing import Optional, Callable, Dict  # 类型提示
from concurrent.futures import ThreadPoolExecutor, as_completed  # 并发执行
import numpy as np  # 数值计算
import sys  # 系统相关功能
import contextlib  # 上下文管理器
import io  # 输入输出流
import signal  # 信号处理
import concurrent.futures  # 并发执行框架
import tempfile  # 临时文件处理

# 第三方库导入
from tqdm import tqdm  # 进度条显示

# 添加CodeGeeX路径到系统路径
sys.path.append('./CodeGeeX/')

# 项目内部模块导入
from AgentFramework.programmer import call_fetch_completion_helper  # 程序员代理
from AgentFramework.designer import call_fetch_test_completion_helper  # 设计师代理
from CodeGeeX.codegeex.benchmark.utils import read_dataset, IMPORT_HELPER  # 数据集工具
from CodeGeeX.codegeex.benchmark.execution import check_correctness  # 代码正确性检查

# 全局计数器，用于跟踪不同阶段的测试结果
# Global counters for tracking test results across different phases
correct_doctest = 0  # doctest 测试通过的次数
correct_before_doctest = 0  # doctest 之前测试通过的次数
correct_after_doctest = 0  # doctest 之后测试通过的次数
result_original = 0  # 原始代码的测试结果
result_canonical_solution = 0  # 标准解决方案的测试结果
result_fuzzer = 0  # 模糊测试的结果
result_fuzzer_canonical_solution = 0  # 模糊测试标准解决方案的结果
idx_run_tests_orginal = []  # 运行原始测试的索引列表
idx_run_tests_canonical_solution = []  # 运行标准解决方案测试的索引列表
idx_run_tests_fuzzer = []  # 运行模糊测试的索引列表
idx_run_tests_fuzzer_canonical_solution = []  # 运行模糊测试标准解决方案的索引列表

# 支持的语言列表
language = ["python","cpp","js","go","js"]


class TimeoutException(Exception):
    """
    自定义异常类，用于处理代码执行超时错误。
    
    当代码执行时间超过预设限制时抛出此异常，
    用于防止无限循环或长时间运行的代码阻塞整个测试流程。
    """
    pass

class WriteOnlyStringIO(io.StringIO):
    """
    只写StringIO类，读取时抛出异常。
    
    这个类继承自io.StringIO，但重写了所有读取方法使其抛出IOError异常。
    用于在代码执行期间抑制输出，同时仍然允许写入操作。
    这是实现输出抑制机制的核心组件。
    """

    def read(self, *args, **kwargs):
        """
        阻止从StringIO读取数据以抑制输出。
        
        Raises:
            IOError: 总是抛出IOError异常
        """
        raise IOError

    def readline(self, *args, **kwargs):
        """
        阻止从StringIO读取单行数据以抑制输出。
        
        Raises:
            IOError: 总是抛出IOError异常
        """
        raise IOError

    def readlines(self, *args, **kwargs):
        """
        阻止从StringIO读取所有行数据以抑制输出。
        
        Raises:
            IOError: 总是抛出IOError异常
        """
        raise IOError

    def readable(self, *args, **kwargs):
        """
        返回False表示此流不可读取。
        
        Returns:
            bool: 总是返回False
        """
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    """
    标准输入重定向上下文管理器。
    
    继承自contextlib._RedirectStream，用于在代码执行期间重定向stdin。
    这是swallow_io()函数的重要组成部分，用于完全抑制代码执行时的输入输出。
    """
    _stream = 'stdin'  # 指定要重定向的流类型

@contextlib.contextmanager
def swallow_io():
    """
    I/O抑制上下文管理器，在代码执行期间抑制所有输入输出。
    
    这个上下文管理器通过重定向stdout、stderr和stdin到WriteOnlyStringIO对象，
    完全抑制代码执行时的所有输入输出。这对于测试环境非常重要，
    可以防止生成的代码干扰测试流程或产生不必要的输出。
    
    Usage:
        with swallow_io():
            exec(code)  # 执行代码时不会产生任何输出
    
    Note:
        - 使用WriteOnlyStringIO确保输出被完全丢弃
        - 同时抑制标准输出、错误输出和标准输入
        - 适用于需要静默执行的代码测试场景
    """
    stream = WriteOnlyStringIO()  # 创建只写流对象
    with contextlib.redirect_stdout(stream):  # 重定向标准输出
        with contextlib.redirect_stderr(stream):  # 重定向错误输出
            with redirect_stdin(stream):  # 重定向标准输入
                yield  # 执行代码块

@contextlib.contextmanager
def time_limit(seconds: float):
    """
    执行时间限制上下文管理器。
    
    这个上下文管理器使用Unix信号机制来强制执行时间限制，
    防止代码执行时间过长而阻塞整个测试流程。当执行时间超过限制时，
    会抛出TimeoutException异常。
    
    Args:
        seconds (float): 最大执行时间（秒）
        
    Raises:
        TimeoutException: 当执行时间超过限制时抛出
        
    Usage:
        with time_limit(2.0):
            exec(code)  # 代码必须在2秒内完成执行
    
    Note:
        - 使用Unix信号处理机制实现超时功能
        - 基于signal.ITIMER_REAL实时计时器
        - 自动清理信号处理器，确保资源正确释放
    """
    def signal_handler(signum, frame):
        """
        信号处理器函数，在超时时抛出TimeoutException。
        
        Args:
            signum: 信号编号
            frame: 当前栈帧
        """
        raise TimeoutException("Timed out!")
    
    # 设置实时计时器
    signal.setitimer(signal.ITIMER_REAL, seconds)
    # 注册SIGALRM信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    
    try:
        yield  # 执行代码块
    finally:
        # 清理计时器，确保资源正确释放
        signal.setitimer(signal.ITIMER_REAL, 0)

def process_humaneval_test(sample, problems, example_test=False, language=language, test_case=True):
    """
    为HumanEval问题准备测试执行环境。
    
    这个函数构建完整的测试字符串，包括导入语句、代码完成和测试用例。
    它处理不同的测试用例来源和语言特定的要求，是代码测试流程的关键组件。
    
    Args:
        sample (dict): 包含prompt、completion和entry_point的问题样本
        problems (list): 所有问题的列表，用于参考
        example_test (bool): 是否使用示例测试而不是完整测试
        language (list): 支持的语言列表
        test_case (bool): 是否使用生成的测试用例
        
    Returns:
        str: 准备执行的完整测试字符串
        
    Note:
        - 处理原始HumanEval测试和生成的测试用例
        - 为执行设置适当的Python导入
        - 构建check()调用进行验证
        - 支持类定义和函数定义的不同处理方式
    """
    # 获取任务ID和提示信息
    task_id = sample["task_id"]
    task_id = problems.index(sample)  # 在问题列表中找到当前样本的索引
    prompt = sample["prompt"]  # 获取问题提示
    
    # 根据参数选择测试来源
    if example_test and "example_test" in problems[task_id] and problems[task_id]["example_test"] != "":
        # 使用示例测试（如果可用且不为空）
        test = problems[task_id]["example_test"]
    else:
        # 使用标准测试
        test = problems[task_id]["test"]
    
    # 如果指定使用生成的测试用例，则覆盖之前的测试选择
    if test_case:
        test = problems[task_id]["test_case"]
    
    # 获取代码完成部分
    code = sample["completion"]
    
    # 为不同语言进行预处理
    if language == "python":
        # 设置Python导入环境
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        
        # 检查代码是否包含类定义
        if f"class sample['entry_point']" in code:
            # 如果是类定义，直接使用代码和测试
            test_string = test_setup + code + "\n" + test + "\n" + f"check({sample['entry_point']})"
        else:
            # 如果是函数定义，需要包含提示和代码
            test_string = test_setup + prompt + code + "\n" + test + "\n" + f"check({sample['entry_point']})"
    
    return test_string



def preprocess_data(task, lg):
    """
    预处理任务数据，清理代码块标记和断言语句。
    
    这个函数用于清理从模型生成或数据集中获取的代码, 移除markdown代码块标记
    和不需要的断言语句，确保代码可以正确执行。
    
    Args:
        task (dict): 包含prompt和completion的任务字典
        lg (str): 编程语言标识符
        
    Returns:
        dict: 清理后的任务字典
        
    Note:
        - 移除代码块标记 (```language 和 ```)
        - 移除prompt中的assert语句
        - 保持原始数据结构不变
    """
    # 清理completion中的代码块标记
    if f"```{lg}" in task["completion"]:
        # 移除语言特定的代码块开始标记
        task["completion"] = task["completion"][task["completion"].find(f"```{lg}") +len(f"```{lg}"):]
        # 移除代码块结束标记
        task["completion"] = task["completion"][:task["completion"].find("```")]
    elif "```" in task["completion"]:
        # 移除通用代码块标记
        task["completion"] = task["completion"][task["completion"].find("```") +3:]
        task["completion"] = task["completion"][:task["completion"].find("```")]

    # 清理prompt中的代码块标记
    if f"```{lg}" in task["prompt"]:
        # 移除语言特定的代码块开始标记
        task["prompt"] = task["prompt"][task["prompt"].find(f"```{lg}") +len(f"```{lg}"):]
        # 移除代码块结束标记
        task["prompt"] = task["prompt"][:task["prompt"].find("```")]
    elif "```" in task["prompt"]:
        # 移除通用代码块标记
        task["prompt"] = task["prompt"][task["prompt"].find("```") +3:]
        task["prompt"] = task["prompt"][:task["prompt"].find("```")]

    # 移除prompt中的assert语句，避免干扰代码执行
    if "assert" in task["prompt"]:
        task["prompt"] = task["prompt"][:task["prompt"].find("assert")]
    
    return task
                

def test_report(dataset, lg):
    """
    生成测试报告，统计数据集中的代码通过率。
    
    这个函数遍历整个数据集，对每个样本执行测试，统计通过测试的代码数量，
    并计算通过率百分比。用于评估代码质量和模型性能。
    
    Args:
        dataset (list): 包含代码样本的数据集
        lg (str): 编程语言标识符
        
    Note:
        - 使用2秒超时限制防止无限循环
        - 抑制所有I/O输出避免干扰
        - 处理函数名映射问题(entry_point vs candidate)
        - 输出通过率统计信息
    """
    correct = 0  # 通过测试的代码数量
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"  # Python导入设置
    
    # 遍历数据集中的每个样本
    for i in tqdm(range(len(dataset))):
        try:
            # 抑制I/O输出并设置超时
            with swallow_io():
                with time_limit(2.0):
                    try:
                        # 尝试使用原始entry_point执行测试
                        exec(test_setup + "\n" + dataset[i]["completion"] + "\n" + dataset[i]["test"] + "\n" + f"check({dataset[i]['entry_point']})")
                    except NameError:
                        # 如果entry_point不存在，使用candidate作为函数名
                        exec(test_setup + "\n" + dataset[i]["completion"] + "\n" + dataset[i]["test"] + "\n" + f"check(candidate)")
                correct += 1  # 测试通过，计数器加1
        except Exception as exc:
            # 测试失败，忽略异常继续下一个样本
            pass
    
    # 输出测试报告
    print("==============Start Report Testing==============")
    print(f"test_report: {(correct/len(dataset)*100):.1f}")  # 计算并显示通过率百分比

def test_agent_concurrency(dataset, lg):
    """
    并发测试代理代码，选择最佳代码完成方案。
    
    这个函数使用多线程并发执行，测试多个代码完成方案与多个测试用例的组合，
    选择通过测试最多的代码完成方案作为最终结果。这是AgentCoder框架的核心功能。
    
    Args:
        dataset (list): 包含代码完成列表和测试用例列表的数据集
        lg (str): 编程语言标识符
        
    Returns:
        list: 更新后的数据集，包含选择的最佳代码完成方案
        
    Note:
        - 使用ThreadPoolExecutor实现并发执行
        - 测试所有代码-测试用例组合
        - 选择通过测试最多的代码完成方案
        - 设置通过率阈值(3个测试用例)
        - 处理函数名映射问题
    """
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"  # Python导入设置
    _for_completion = 0  # 成功完成的任务计数器
    
    def process_item(i):
        """
        处理单个数据项的内部函数。
        
        测试该数据项的所有代码完成方案，返回最佳方案的信息。
        
        Args:
            i (int): 数据项索引
            
        Returns:
            tuple: (max_correct, idx) 最大通过数和最佳方案索引
        """
        # 如果已经处理过且不需要重新生成，直接返回结果
        if "need_reproduce" in dataset[i].keys() and dataset[i]["need_reproduce"] == False:
            return dataset[i]["max_correct"], dataset[i]["idx"]
            
        completion_list = dataset[i]["completion_list"]  # 代码完成方案列表
        test_case_list = dataset[i]["test_case_list"]    # 测试用例列表
        correct_list = []  # 每个代码完成方案的通过测试数列表
        
        # 随机选择一个代码完成方案初始化completion条目
        # 这个条目是check_correctness()正常运行所需要的
        dataset[i]["completion"] = random.choice(completion_list)

        # 遍历所有代码完成方案
        for j in range(len(completion_list)):
            correct = 0  # 当前代码完成方案通过的测试数
            
            # 检查代码完成方案是否包含正确的函数定义
            if f"def {dataset[i]['entry_point']}" and f"def candidate" not in completion_list[j]:
                correct_list.append(correct)  # 没有正确函数定义，通过数为0
                continue
                
            # 遍历所有测试用例
            for k in range(len(test_case_list)):
                # 检查测试用例是否包含正确的函数调用
                if f"assert {dataset[i]['entry_point']}(" and f"assert candidate(" not in test_case_list[k]:
                    continue  # 跳过不匹配的测试用例
                    
                # 构建完整的测试代码
                dataset[i]["full_code"] = test_setup + "\n" + completion_list[j] + "\n" + test_case_list[k]
                
                try:
                    # 执行测试，超时时间3秒，临时目录"./tmp"
                    result = check_correctness(dataset[i]["task_id"], dataset[i], lg, 3, "./tmp")
                except NameError:
                    # 如果函数名不匹配，尝试使用candidate作为函数名
                    dataset[i]["full_code"] = test_setup + "\n" + completion_list[j].replace(dataset[i]["entry_point"], "candidate") + "\n" + test_case_list[k].replace(dataset[i]["entry_point"], "candidate")
                    result = check_correctness(dataset[i]["task_id"], dataset[i], lg, 3, "./tmp")
                    
                # 如果测试通过，增加通过计数
                if result["passed"]:
                    correct += 1
                    
            correct_list.append(correct)  # 记录当前代码完成方案的通过数

        # 找到通过测试最多的代码完成方案
        max_correct = max(correct_list)
        idx = correct_list.index(max_correct)  # 最佳方案的索引
        return max_correct, idx

    # 使用线程池并发处理所有数据项
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有任务到线程池
        futures = [executor.submit(process_item, i) for i in range(len(dataset))]

        # 处理完成的任务结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset)):
            max_correct, idx = future.result()
            
            # 如果通过测试数达到阈值（3个），认为任务成功完成
            # GPT-3.5-turbo-1106的测试用例准确率约为67%，所以选择60%作为阈值
            if max_correct >= 3:
                i = futures.index(future)  # 获取对应的数据项索引
                dataset[i]["completion"] = dataset[i]["completion_list"][idx]  # 设置最佳代码完成方案
                print("created completion list")
                dataset[i]["need_reproduce"] = False  # 标记不需要重新生成
                dataset[i]["idx"] = idx  # 记录最佳方案索引
                dataset[i]["max_correct"] = max_correct  # 记录最大通过数
                _for_completion += 1  # 成功完成任务计数
            else:
                # 即使没有达到阈值，也选择最佳方案
                i = futures.index(future)
                dataset[i]["completion"] = dataset[i]["completion_list"][idx]

    return dataset

def executor_main(task_id):
    """
    执行器主函数, 协调整个AgentCoder工作流程。
    
    这是AgentCoder框架的主要入口点, 负责加载数据集、执行代码测试、
    调用其他代理（程序员和设计师）进行代码改进，并保存最终结果。
    
    Args:
        task_id (str): 任务标识符，用于定位对应的数据集文件
        
    Returns:
        list: 处理完成的数据集
        
    Workflow:
        1. 加载初始数据集
        2. 执行并发测试选择最佳代码完成方案
        3. 生成测试报告
        4. 调用程序员代理改进代码
        5. 调用设计师代理改进测试用例
        6. 保存中间结果
        7. 再次执行测试和报告
        
    Note:
        - 支持多个模型和语言(当前配置为AgentCoder + Python)
        - 实现迭代改进循环
        - 自动保存处理结果
    """
    model_list = ["AgentCoder"]  # 支持的模型列表
    language = ["python"]        # 支持的编程语言列表
    
    # 遍历所有模型和语言组合
    for model in model_list:
        for lg in language:
            # 构建数据集文件路径
            path = f"./dataset/{model}_{lg}_{task_id}.json"
            
            # 加载初始数据集
            with open(path, "r") as f:
                dataset = json.load(f)
            
            # 注释掉的迭代循环代码
            # epoch = 1
            # for current_epoch in range(epoch):
            # # We choose to run only one epoch for our baseline
            
            # 第一阶段：测试和选择最佳代码完成方案
            dataset = test_agent_concurrency(dataset, lg)
            test_report(dataset, lg)  # 生成初始测试报告
            
            # 第二阶段：调用其他代理进行改进
            # 调用程序员代理获取改进的代码完成方案
            dataset = call_fetch_completion_helper(dataset, model, lg)
            # 调用设计师代理获取改进的测试用例
            dataset = call_fetch_test_completion_helper(dataset, model, lg)
            
            # 保存中间结果到文件
            with open(f"./dataset/{model}_{task_id}.json", "w") as f:
                json.dump(dataset, f, indent=4)
            
            # 第三阶段：再次测试改进后的结果
            dataset = test_agent_concurrency(dataset, lg)
            test_report(dataset, lg)  # 生成最终测试报告
            
    return dataset
