"""
LLM 失败检测工具

用法：
    python llm_failure_detector.py "@batch_evaluation_log_20260118_032727.md (42-55)"

或在 Python 中使用：
    from llm_failure_detector import analyze_failure_from_log
    result = analyze_failure_from_log("@batch_evaluation_log_20260118_032727.md (42-55)")
"""

import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 尝试导入 LLMPrompter
try:
    from craft.reasoning.llm_prompter import LLMPrompter
except ImportError:
    try:
        from reasoning.llm_prompter import LLMPrompter
    except ImportError:
        LLMPrompter = None


def parse_input(input_str: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    解析输入格式：@filename (start-end) 或 @filename (start)
    
    示例：
        "@batch_evaluation_log_20260118_032727.md (42-55)"
        "@batch_evaluation_log_20260118_032727.md (42)"
    """
    # 移除 @ 符号
    input_str = input_str.strip().lstrip('@')
    
    # 匹配格式：filename (start-end) 或 filename (start)
    pattern = r'^(.+?)\s*\((\d+)(?:-(\d+))?\)\s*$'
    match = re.match(pattern, input_str)
    
    if not match:
        raise ValueError(f"输入格式错误。请使用格式: @filename (start-end) 或 @filename (start)")
    
    filename = match.group(1).strip()
    start_line = int(match.group(2))
    end_line = int(match.group(3)) if match.group(3) else None
    
    return filename, start_line, end_line


def read_log_lines(log_file: Path, start_line: int, end_line: Optional[int] = None) -> str:
    """读取日志文件的指定行号范围"""
    if not log_file.exists():
        raise FileNotFoundError(f"日志文件不存在: {log_file}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 转换为 0-based 索引
    start_idx = start_line - 1
    if end_line is None:
        end_idx = len(lines)
    else:
        end_idx = end_line
    
    selected_lines = lines[start_idx:end_idx]
    return ''.join(selected_lines)


def build_prompt(log_content: str) -> str:
    """构建 LLM prompt"""
    prompt = """You are a robot failure diagnosis module.

Given a sequence of robot actions and symbolic constraint checking results,
your task is to determine:

1. Whether the task execution has failed.
2. The most likely failure step (earliest root cause).
3. The reason for failure in concise natural language.
4. The failure type, chosen from the following taxonomy:

- Precondition Violation
- Postcondition Violation
- Physical State Conflict
- Causal Chain Break
- Perception Uncertainty
- No Failure

Rules:
- Do NOT simply pick the last violated constraint.
- Prefer the earliest violation that can causally explain subsequent violations.
- If multiple violations exist, identify the root cause and treat later ones as consequences.
- If all violations are explainable as downstream effects, classify accordingly.
- Be conservative: only mark failure if constraints indicate irrecoverable execution error.

Here is the constraint checking log:

```
{log_content}
```

Please provide your analysis in the following JSON format:
{{
    "has_failure": true/false,
    "failure_step": "step number (e.g., 'Step 7' or 'Action 7') or null",
    "failure_reason": "concise reason in one sentence",
    "failure_type": "one of: Precondition Violation, Postcondition Violation, Physical State Conflict, Causal Chain Break, Perception Uncertainty, No Failure",
    "root_cause_analysis": "brief explanation of why this is the root cause and how it explains other violations"
}}
"""
    return prompt.format(log_content=log_content)


def parse_llm_response(response: str) -> Dict:
    """解析 LLM 响应（可能是 JSON 格式或文本格式）"""
    # 尝试提取 JSON
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            return result
        except json.JSONDecodeError:
            pass
    
    # 如果无法解析 JSON，尝试从文本中提取信息
    result = {
        "has_failure": None,
        "failure_step": None,
        "failure_reason": None,
        "failure_type": None,
        "root_cause_analysis": None
    }
    
    # 提取 failure type
    failure_types = [
        "Precondition Violation", "Postcondition Violation",
        "Physical State Conflict", "Causal Chain Break",
        "Perception Uncertainty", "No Failure"
    ]
    for ft in failure_types:
        if ft.lower() in response.lower():
            result["failure_type"] = ft
            break
    
    # 提取 has_failure
    if "has_failure" in response.lower() or "failure" in response.lower():
        if "no failure" in response.lower() or "success" in response.lower():
            result["has_failure"] = False
        elif "failure" in response.lower():
            result["has_failure"] = True
    
    # 提取 failure_step
    step_match = re.search(r'step\s+(\d+)', response, re.IGNORECASE)
    if step_match:
        result["failure_step"] = f"Step {step_match.group(1)}"
    
    # 使用完整响应作为 reason（如果无法解析）
    if not result.get("failure_reason"):
        # 尝试提取第一句话作为 reason
        sentences = re.split(r'[.!?]\s+', response)
        if sentences:
            result["failure_reason"] = sentences[0].strip()
        else:
            result["failure_reason"] = response[:200]  # 截取前200字符
    
    return result


def analyze_failure_from_log(
    input_str: str,
    log_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    gpt_model: str = 'gpt-3.5-turbo',
    verbose: bool = True
) -> Dict:
    """
    从日志文件中分析失败，或直接分析提供的文本
    
    Args:
        input_str: 
            - 文件格式: "@filename (start-end)" 或 "@filename (start)"（读取文件指定行）
            - 文本格式: 直接提供日志文本内容（整段话）
        log_dir: 日志文件目录（默认: evaluation_results）
        api_key: API Key（如果为 None，会尝试从环境变量或全局变量获取）
        gpt_model: GPT 模型（默认: gpt-3.5-turbo）
        verbose: 是否打印详细输出
    
    Returns:
        分析结果字典
    """
    log_content = None
    log_source = None
    
    # 尝试解析为文件格式（@filename (start-end)）
    if input_str.strip().startswith('@'):
        try:
            filename, start_line, end_line = parse_input(input_str)
            
            # 确定日志文件路径
            if log_dir is None:
                log_dir = Path('evaluation_results')
            
            log_file = log_dir / filename
            if not log_file.exists():
                # 尝试在当前目录查找
                log_file = Path(filename)
                if not log_file.exists():
                    raise FileNotFoundError(f"找不到日志文件: {log_dir / filename} 或 {filename}")
            
            if verbose:
                print(f"📂 读取日志文件: {log_file}")
                print(f"📄 行号范围: {start_line}-{end_line if end_line else 'EOF'}")
            
            # 读取日志内容
            log_content = read_log_lines(log_file, start_line, end_line)
            log_source = f"文件: {log_file}, 行号: {start_line}-{end_line if end_line else 'EOF'}"
        except ValueError:
            # 如果解析失败，将整个输入（包括@）视为文本内容
            if verbose:
                print("ℹ️  输入不符合文件格式，将作为文本内容处理")
            log_content = input_str
            log_source = "直接输入的文本"
    else:
        # 直接作为文本内容处理
        log_content = input_str
        log_source = "直接输入的文本"
    
    if verbose:
        print(f"\n📋 日志内容预览:")
        print("-" * 80)
        log_lines_list = log_content.split('\n')
        preview_lines = log_lines_list[:10]
        for line in preview_lines:
            print(line)
        if len(log_lines_list) > 10:
            total_lines = len(log_lines_list)
            print(f"... (共 {total_lines} 行)")
        print("-" * 80)
    
    # 构建 prompt
    prompt = build_prompt(log_content)
    
    # 初始化 LLM Prompter
    if LLMPrompter is None:
        raise ImportError("无法导入 LLMPrompter。请确保 craft 模块已正确安装。")
    
    if api_key is None:
        # 尝试从环境变量获取
        api_key = os.environ.get('POLOAPI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            # 尝试从全局变量获取（如果在 notebook 中运行）
            try:
                import __main__
                api_key = getattr(__main__, 'API_KEY', None)
            except:
                pass
    
    if not api_key:
        raise ValueError("未找到 API Key。请设置环境变量 POLOAPI_API_KEY 或 OPENAI_API_KEY，或在调用时提供 api_key 参数。")
    
    llm_prompter = LLMPrompter(
        gpt_version=gpt_model,
        api_key=api_key,
        base_url="https://poloai.top/v1"
    )
    
    if verbose:
        print(f"\n🤖 正在使用 LLM ({gpt_model}) 分析失败...")
    
    # 调用 LLM（LLMPrompter 使用 query 方法，需要 system_prompt 和 user_prompt）
    try:
        # 将 prompt 分为 system 和 user 部分
        # 由于我们的 prompt 是一个完整的 prompt，我们将其作为 user_prompt
        system_prompt = "You are a robot failure diagnosis module."
        user_prompt = prompt  # 完整的 prompt 作为 user_prompt
        
        response, metadata = llm_prompter.query(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000
        )
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}")
    
    if verbose:
        print(f"\n📝 LLM 原始响应:")
        print("-" * 80)
        print(response)
        print("-" * 80)
    
    # 解析响应
    result = parse_llm_response(response)
    result['raw_response'] = response
    result['log_source'] = log_source
    
    return result


def print_result(result: Dict):
    """格式化打印分析结果"""
    print("\n" + "=" * 80)
    print("🔍 LLM 失败检测结果")
    print("=" * 80)
    print()
    
    log_source = result.get('log_source', 'N/A')
    print(f"📋 日志来源: {log_source}")
    print()
    
    print("📊 分析结果:")
    print("-" * 80)
    
    has_failure = result.get('has_failure')
    if has_failure is not None:
        status = "❌ 失败" if has_failure else "✅ 成功"
        print(f"执行状态: {status}")
    
    failure_step = result.get('failure_step')
    if failure_step:
        print(f"失败步骤: {failure_step}")
    
    failure_type = result.get('failure_type')
    if failure_type:
        print(f"失败类型: {failure_type}")
    
    failure_reason = result.get('failure_reason')
    if failure_reason:
        print(f"失败原因: {failure_reason}")
    
    root_cause = result.get('root_cause_analysis')
    if root_cause:
        print(f"\n根因分析:")
        print(f"  {root_cause}")
    
    print("-" * 80)


def main():
    """命令行入口"""
    if len(sys.argv) < 2:
        print("用法: python llm_failure_detector.py \"@filename (start-end)\"")
        print("示例: python llm_failure_detector.py \"@batch_evaluation_log_20260118_032727.md (42-55)\"")
        sys.exit(1)
    
    input_str = sys.argv[1]
    
    try:
        result = analyze_failure_from_log(input_str, verbose=True)
        print_result(result)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()