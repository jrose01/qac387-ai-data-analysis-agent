"""
build4 with streamlit UI: HITL + Router (Tool Routing + Optional CodeGen/Execute) + RAG(FAISS retrieval from knowledge base)
+ Langfuse tracing (LangChain callbacks + observe decorator) + prompt management for router prompt

THis build adds a top-level RAG ROUTER that decides whether to:
    (A) run one of the Build0 tools, OR
    (B) fall back to CodeGen + optional Execute (subprocess).
    (C) Create a streamlit UI to interact with the agent in a more user-friendly way.


It includes a single main command:
    ask <request>   (router decides tool vs codegen)

Keeps power-user commands:
    tool <request>  (force tool mode)
    code <request>  (force codegen mode)
    run             (execute last approved code)

You will need the expected Build0 tool registry (tools.py in the updated src folder)

Each tool function should accept (df, report_dir, **kwargs) and ideally return ToolResult.

The runtime is backward compatible and will also normalize:
- a string
- a dict with "text" and optional "artifact_paths"
- a tuple of (text, artifact_paths)

To run this script, you will need to make sure you have the most updated src and requirements.txt file
from the course repository.

Then, in the terminal or command line, run:
  python builds/build4_rag_router_agent_prompt_mgmt.py --data data/penguins.csv --knowledge_dir knowledge --report_dir reports --tags build4 --memory

  To stream LLM output, add the --stream flag to the command above

To interact with the agent, use the following commands:
    help                         Show this help text
    schema                       Print dataset schema
    suggest <question>           Questions about the dataset or analysis (LLM)
    ask <request>                ROUTER decides: tool-run OR codegen (HITL)
    tool <request>               Force tool-run: choose one Build0 tool + args (HITL)
    code <request>               Force code generation (HITL) + approve to save
    run                          Execute last approved script via subprocess (HITL)
    exit                         Quit
"""

from __future__ import annotations

import argparse

# from ast import If
# import code
import importlib
import inspect
import json

# from os import read
import re
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from textwrap import dedent
# from wsgiref import validate

# from matplotlib.pylab import save
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

# Add project root to Python path for findingbuilds folder and importing src modules;
sys.path.append(str(Path(__file__).resolve().parents[1]))

# import the ToolResult dataclass and normalize_tool_return function
# for consistent tool output formatting
from src.utils.tool_result_utils import ToolResult, normalize_tool_return

from src import ensure_dirs, read_data, basic_profile
from src.rag_faiss_utils_pdf import (
    load_faiss_index,
    retrieve_chunks,
    format_rag_context,
)

load_dotenv(".env")


# --------------------------------------------------------------------------------------
# Langfuse instrumentation
# --------------------------------------------------------------------------------------
LANGFUSE_AVAILABLE = False
langfuse = None

try:
    from langfuse import get_client as lf_get_client, observe, propagate_attributes  # type: ignore
    from langfuse.langchain import CallbackHandler  # type: ignore

    langfuse = lf_get_client()
    LANGFUSE_AVAILABLE = True
except Exception:
    LANGFUSE_AVAILABLE = False

    def observe(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap

    class propagate_attributes:  # type: ignore
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False


# --------------------------------------------------------------------------------------
# New helper code for Langfuse prompt management
# --------------------------------------------------------------------------------------
def load_langfuse_prompt(
    prompt_name: str,
    label: str = "dev",
) -> tuple[Any, Dict[str, Any]]:
    if not LANGFUSE_AVAILABLE or langfuse is None:
        raise RuntimeError("Langfuse is not available.")

    prompt = langfuse.get_prompt(prompt_name, label=label, cache_ttl_seconds=0)

    cfg = getattr(prompt, "config", None) or {}
    return prompt, cfg


def get_prompt_config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe defaults for prompt-managed settings.
    """
    return {
        "model": cfg.get("model", "gpt-4o-mini"),
        "temperature": float(cfg.get("temperature", 0.0)),
        "stream": bool(cfg.get("stream", False)),
    }


def compile_router_prompt_from_langfuse(
    *,
    prompt_name: str,
    label: str,
    allowed_tools: list[str],
    tool_descriptions: Dict[str, str],
    tool_arg_hints: str,
) -> tuple[str, Any, Dict[str, Any]]:
    prompt, cfg = load_langfuse_prompt(prompt_name=prompt_name, label=label)

    allowed_tools_text = format_capability_hints(allowed_tools, tool_descriptions)

    compiled_prompt = prompt.compile(
        allowed_tools_text=allowed_tools_text,
        tool_arg_hints=tool_arg_hints,
    )

    return str(compiled_prompt), prompt, cfg


# --------------------------------------------------------------------------------------
# Minimal RAG helpers (Build4: retrieval added to codegen path)
# --------------------------------------------------------------------------------------


@dataclass
class RagIndex:
    """Container for a prebuilt FAISS-backed RAG index."""

    index: Any
    chunks: list[Any]
    knowledge_dir: Path
    embedding_model: str


def load_saved_rag_index(knowledge_dir: str | Path) -> RagIndex:
    """Load a previously built FAISS index and its chunk metadata."""
    knowledge_dir = Path(knowledge_dir)
    index, chunks, embedding_model = load_faiss_index(knowledge_dir)
    return RagIndex(
        index=index,
        chunks=chunks,
        knowledge_dir=knowledge_dir,
        embedding_model=embedding_model,
    )


def prepare_codegen_request_with_rag(
    req: str,
    schema_text: str,
    rag_index: Optional[RagIndex],
    rag_k: int = 4,
) -> tuple[str, Optional[str]]:
    if rag_index is None:
        return req, None

    retrieval_query = f"User request: {req}\n\nDataset schema:\n{schema_text}"
    results = retrieve_chunks(
        query=retrieval_query,
        index=rag_index.index,
        chunks=rag_index.chunks,
        k=rag_k,
        embedding_model=rag_index.embedding_model,
    )
    rag_context = format_rag_context(results)

    augmented_request = dedent(
        f"""
        Retrieved reference material:
        {rag_context}

        Original user request:
        {req}

        Use the retrieved material when it is relevant, but only reference dataset columns
        that actually appear in the schema.
        """
    ).strip()

    return augmented_request, rag_context


# notification helper to print RAG status in the CLI at startup and after loading the index
def print_rag_status(rag_index):
    print("\nRAG STATUS")
    print("----------")

    if rag_index is None:
        print("RAG disabled")
        print("(no knowledge_dir provided)\n")
        return

    print("RAG enabled")
    print(f"knowledge_dir  : {rag_index.knowledge_dir}")
    print(f"chunks loaded  : {len(rag_index.chunks)}")
    print(f"embedding model: {rag_index.embedding_model}\n")


# --------------------------------------------------------------------------------------
# Artifact Helpers
# --------------------------------------------------------------------------------------


def setup_artifact_dirs(report_dir: Path) -> tuple[Path, Path]:
    """Create and return standardized artifact directories."""
    tool_output_dir = report_dir / "tool_outputs"
    tool_figure_dir = report_dir / "tool_figures"
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    tool_figure_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== ARTIFACT DIRECTORIES ===")
    print("tool_outputs :", tool_output_dir)
    print("tool_figures :", tool_figure_dir)
    print()
    return tool_output_dir, tool_figure_dir


def inject_artifact_paths(
    tool_fn,
    tool_name: str,
    args: Dict[str, Any],
    tool_output_dir: Path,
    tool_figure_dir: Path,
) -> Dict[str, Any]:
    """
    Inject standard artifact directories into tool arguments if the tool supports them
    and they weren't explicitly provided by the router/user.
    """
    sig = inspect.signature(tool_fn)
    params = sig.parameters

    # Common directory-style params across your tools
    dir_param_candidates = {
        "fig_dir": tool_figure_dir,
        "plot_dir": tool_figure_dir,
        "plots_dir": tool_figure_dir,
        "figure_dir": tool_figure_dir,
        "figures_dir": tool_figure_dir,
        "out_dir": tool_output_dir,
        "output_dir": tool_output_dir,
        "artifact_dir": tool_output_dir,
        "report_dir": tool_output_dir,
    }

    for p, default_dir in dir_param_candidates.items():
        if p in params and p not in args:
            args[p] = default_dir

    # Common single-file “output path” parameters (optional but helpful)
    # Only set if the tool takes it AND it wasn't provided.
    file_param_candidates = ["out_path", "output_path", "save_path"]
    for p in file_param_candidates:
        if p in params and p not in args:
            # Choose an extension that won’t break most tools; many plotting functions accept .png
            # and table functions often accept .csv/.json. If tool needs a specific extension, it should
            # set its own default or the router can pass it explicitly.
            default_path = tool_output_dir / f"{tool_name}_output"
            args[p] = default_path

    return args


def print_artifact_summary(tool_output_dir: Path, tool_figure_dir: Path) -> None:
    """Nice CLI printout; handy now and later for a UI."""
    print("\n=== ARTIFACT LOCATIONS ===")
    print(f"Tool outputs : {tool_output_dir}")
    print(f"Tool figures : {tool_figure_dir}\n")


def profile_to_schema_text(profile: dict) -> str:
    lines = [
        f"Rows: {profile.get('n_rows')}",
        f"Columns: {profile.get('n_cols')}",
        "",
        "Columns and dtypes:",
    ]
    for col in profile["columns"]:
        lines.append(f"- {col}: {profile['dtypes'].get(col)}")
    return "\n".join(lines)


# Regexes to extract fenced code blocks and JSON blocks from LLM output (best-effort, for flexibility in formatting)
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text or "")
    return m.group(1).strip() if m else None


# This is a best-effort split of the LLM response into PLAN / CODE / VERIFY sections,
# based on simple substring searches.
def split_sections(text: str) -> Tuple[str, str, str]:
    """Split an LLM response into PLAN / CODE / VERIFY sections (best-effort)."""
    if not text:
        return "", "", ""
    up = text.upper()
    i_plan = up.find("PLAN:")
    i_code = up.find("CODE:")
    i_ver = up.find("VERIFY:")
    if i_plan == -1 or i_code == -1 or i_ver == -1:
        return text.strip(), "", ""
    return text[i_plan:i_code].strip(), text[i_code:i_ver].strip(), text[i_ver:].strip()


def invoke_chain_text(
    chain,
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    stream: bool,
    print_output: bool = True,
) -> str:
    if stream:
        chunks = []
        for chunk in chain.stream(inputs, config=config):
            if print_output:
                print(chunk, end="", flush=True)
            chunks.append(chunk)
        if print_output:
            print("\n")
        return "".join(chunks)

    out = chain.invoke(inputs, config=config)
    if print_output:
        print("\n" + out + "\n")
    return out


def parse_json_object(raw: str) -> Dict[str, Any]:
    """
    Parse a JSON object from:
      - raw JSON text
      - a fenced ```json block
      - or a near-JSON object that uses doubled braces like {{ ... }}

    Returns {} on failure.
    """
    raw = (raw or "").strip()

    candidates = [raw]

    # If the model copied doubled braces from prompt examples, normalize them.
    if "{{" in raw or "}}" in raw:
        candidates.append(raw.replace("{{", "{").replace("}}", "}"))

    # If wrapped in a fenced json block, try that too.
    match = JSON_BLOCK_RE.search(raw)
    if match:
        block = match.group(1).strip()
        candidates.append(block)
        if "{{" in block or "}}" in block:
            candidates.append(block.replace("{{", "{").replace("}}", "}"))

    # Fallback: try substring from first { to last }
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        sub = raw[i : j + 1].strip()
        candidates.append(sub)
        if "{{" in sub or "}}" in sub:
            candidates.append(sub.replace("{{", "{").replace("}}", "}"))

    for text in candidates:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            continue

    return {}


def find_unknown_columns(args_obj: Any, known_columns: set[str]) -> set[str]:
    """
    Walk the tool args and identify unknown column references in common keys
    (column/columns/x/y/outcome/predictors/etc.). This prevents hallucinated columns.
    """
    expected_column_keys = {
        "column",
        "columns",
        "col",
        "cols",
        "x",
        "y",
        "outcome",
        "predictor",
        "predictors",
        "feature",
        "features",
        "target",
        "groupby",
    }
    unknown: set[str] = set()

    def walk(obj: Any, key_hint: Optional[str] = None) -> None:
        key_l = (key_hint or "").lower()
        expects_column = (
            key_l in expected_column_keys
            or key_l.endswith("_col")
            or key_l.endswith("_cols")
        )

        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, str(k))
            return

        if isinstance(obj, list):
            for item in obj:
                walk(item, key_hint)
            return

        if isinstance(obj, str) and expects_column and obj not in known_columns:
            unknown.add(obj)

    walk(args_obj)
    return unknown


def coerce_tool_args(raw_args: Any) -> Dict[str, Any]:
    """Ensure tool args are a dict so **kwargs calls are safe."""
    if isinstance(raw_args, dict):
        return raw_args
    return {}


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def parse_tags(tags_csv: str) -> list[str]:
    return [t.strip() for t in (tags_csv or "").split(",") if t.strip()]


def make_langfuse_config(session_id: str, tags: list[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"configurable": {"session_id": session_id}}

    if LANGFUSE_AVAILABLE:
        try:
            cfg["callbacks"] = [CallbackHandler(sessionId=session_id, tags=tags)]  # type: ignore
        except TypeError:
            cfg["callbacks"] = [CallbackHandler()]  # type: ignore
        cfg["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_tags": tags,
        }

    return cfg


# --------------------------------------------------------------------------------------
# Build0 tool registry loader
# --------------------------------------------------------------------------------------
ToolFn = Callable[..., Any]


def load_tools() -> Dict[str, ToolFn]:
    """
    Load TOOLS registry from your Build0 codebase.

    Search order:
      1) src.tools: TOOLS
      2) src.build0_tools: TOOLS
      3) src: TOOLS  (exported in src/__init__.py)
    """
    candidates = [
        ("src.tools", "TOOLS"),
        ("src.build0_tools", "TOOLS"),
        ("src", "TOOLS"),
    ]

    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            tools = getattr(mod, attr)
            if isinstance(tools, dict) and tools:
                return tools
        except Exception:
            continue

    raise RuntimeError(
        dedent("""
        Could not import a TOOLS registry.
        Create src/tools.py with something like:
        from src.some_build0_module import describe_numeric, freq_table, simple_ols
        TOOLS = {
        'describe_numeric': describe_numeric,
        'freq_table': freq_table,
        'simple_ols': simple_ols,
        }
        Then rerun this script.
        
        """)
    )


def load_tool_descriptions() -> Dict[str, str]:
    """Best-effort load of optional TOOL_DESCRIPTIONS from src.tools."""
    try:
        mod = importlib.import_module("src.tools")
        raw = getattr(mod, "TOOL_DESCRIPTIONS", {})
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    except Exception:
        pass
    return {}


def format_capability_hints(
    allowed_tools: list[str], descriptions: Dict[str, str]
) -> str:
    lines = []
    for tool in allowed_tools:
        desc = descriptions.get(tool, "")
        if desc:
            lines.append(f"- {tool}: {desc}")
        else:
            lines.append(f"- {tool}")
    return "\n".join(lines)


def format_tool_arg_hints(tools: Dict[str, ToolFn], allowed_tools: list[str]) -> str:
    """
    Build argument-name guidance from real tool signatures.

    Excludes framework/runtime params like df/report_dir and variadic params.
    """
    lines: list[str] = []
    for tool_name in allowed_tools:
        fn = tools.get(tool_name)
        if fn is None:
            continue

        required: list[str] = []
        optional: list[str] = []
        try:
            sig = inspect.signature(fn)
            for p in sig.parameters.values():
                if p.name in {"df", "report_dir"}:
                    continue
                if p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
                else:
                    optional.append(p.name)
        except (TypeError, ValueError):
            lines.append(f"- {tool_name}: args unknown (could not inspect signature)")
            continue

        if required and optional:
            lines.append(f"- {tool_name}: required={required}; optional={optional}")
        elif required:
            lines.append(f"- {tool_name}: required={required}; optional=[]")
        elif optional:
            lines.append(f"- {tool_name}: required=[]; optional={optional}")
        else:
            lines.append(f"- {tool_name}: required=[]; optional=[]")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------------------
def build_suggest_chain(
    model: str, temperature: float = 0.0, stream: bool = False, memory: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    suggest_system_text = """
        You are a data analysis assistant.
        You ONLY see the dataset schema (columns + dtypes). Do NOT invent columns.
        Return:
        1) 2-3 plausible research questions that can be tested based on the dataset (bulleted)
        2) For each: outcome(s), predictor(s), and suggested analysis type
        3) 5-7 clarifying questions
        """

    if memory:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=suggest_system_text),
                ("human", "Dataset schema:\n{schema_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User question:\n{user_query}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=suggest_system_text),
                (
                    "human",
                    "Dataset schema:\n{schema_text}\n\nUser question:\n{user_query}\n",
                ),
            ]
        )

    base_chain = prompt | llm | StrOutputParser()
    if not memory:
        return base_chain

    history = InMemoryChatMessageHistory()
    return RunnableWithMessageHistory(
        base_chain,
        lambda _session_id: history,
        input_messages_key="user_query",
        history_messages_key="history",
    )


def build_codegen_chain(
    model: str, temperature: float = 0.0, stream: bool = False, memory: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    codegen_system_text = """
    You are a careful Python data analysis code generator.

    IMPORTANT RULES:
    - You ONLY know the dataset schema. Do NOT invent columns.
    - Produce ONE Python script that can run as a standalone file.
    - The script MUST:
      (1) use argparse with --data and --report_dir
      (2) read the CSV at --data with pandas
      (3) handle missing values explicitly
      (4) If missing data are present, use listwise deletion unless specified otherwise.
      (5) save at least ONE artifact into --report_dir
      (6) validate referenced columns exist (exit nonzero if not)

    OUTPUT FORMAT (exactly):

    PLAN:
    - ...brief plan for the analysis and what the code will do...

    CODE:
    ```python
    # full script
    ```

    VERIFY:
    - ...brief verification checklist to ensure code correctness and validity...
    """

    if memory:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=codegen_system_text),
                ("human", "Dataset schema:\n{schema_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User request:\n{user_request}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=codegen_system_text),
                (
                    "human",
                    "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
                ),
            ]
        )

    base_chain = prompt | llm | StrOutputParser()

    if not memory:
        return base_chain

    history = InMemoryChatMessageHistory()
    return RunnableWithMessageHistory(
        base_chain,
        lambda _session_id: history,
        input_messages_key="user_request",
        history_messages_key="history",
    )


def build_toolplan_chain(
    model: str,
    allowed_tools: list[str],
    tool_descriptions: Dict[str, str],
    tool_arg_hints: str,
    temperature: float = 0.0,
    stream: bool = False,
):
    """Pick one tool + args ONLY (JSON)."""
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)

    allow_str = format_capability_hints(allowed_tools, tool_descriptions)

    build_toolplan_system_text = dedent("""
    Return ONLY valid JSON in exactly ONE of these forms.
    If you choose a tool:
    {"mode":"tool","tool":"plot_histograms","args":{"numeric_cols":["numeric_col","numeric_col"]},"note":"<brief>"}

    If no tool can satisfy the request:
    {"mode":"codegen","code_request":"<brief concrete coding request>","note":"<brief>"}
    
    Rules:
    - Use ONLY columns in the schema.
    - args keys MUST use valid parameter names for the selected tool signature above.
    - Do NOT use generic keys like "column" unless that exact parameter exists.
    - If there is no tool to complete the request, fall back to codegen mode.
    - IMPORTANT: If the selected tool requires an input column, args MUST include it.
    - Never output an empty args object for summarize_categorical.

    - For summarize_categorical:
    - If the user requests one column, use args={"column":"<col>"}
    - If the user requests multiple categorical columns, use args={"cat_cols":["<col1>","<col2>"]}
    - Filesystem paths, report directories, and session folders are handled by the runtime.
    
    Examples:
    User: "frequency table for "cat_col""
    {"mode":"tool","tool":"summarize_categorical","args":{"column":"cat_col"},"note":"Frequency table is a categorical summary."}
    
    User: "frequency tables for "cat_col" and "cat_col""
    {"mode":"tool","tool":"summarize_categorical","args":{"cat_cols":["cat_col","cat_col"]},"note":"Summarize multiple categorical columns."}
    
    User: "show missingness"
    {"mode":"tool","tool":"missingness_table","args":{},"note":"Missingness summary is available as a tool."}
    
    User: "histograms for numeric columns"
    {"mode":"tool","tool":"plot_histograms","args":{"numeric_cols":["numeric_col","numeric_col"]},"note":"Histogram tool visualizes numeric distributions."}

    FINAL OUTPUT REQUIREMENTS:
    - Output MUST be valid JSON.
    - Do NOT include markdown, backticks, or explanations.
    - Do NOT include any text before or after the JSON.
    - The response must be parseable by json.loads().
    """)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=build_toolplan_system_text),
            (
                "human",
                "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


# NEW Langfuse prompt management function for building the router system text as a fallback
# if Langfuse prompt management fails or isn't used.
# This is the same text as the prompt managed in Langfuse, but with dynamic variables
# for the allowed tools and their argument hints.
def build_router_system_text_fallback(
    *,
    allowed_tools: list[str],
    tool_descriptions: Dict[str, str],
    tool_arg_hints: str,
) -> str:
    allow_str = format_capability_hints(allowed_tools, tool_descriptions)

    router_system_text = dedent("""
    You are a TOOL ROUTER for a data analysis CLI.

    You see:
    - Dataset schema (columns + dtypes)
    - Allow-list tools + tool signatures
    - User request

    Allow-list tools:
    {{allowed_tools_text}}

    Tool argument names by signature:
    {{tool_arg_hints}}

    Your job:
    1. Decide whether an allow-listed tool can directly satisfy the user request.
    2. If yes, return a tool decision.
    3. If no, return a codegen decision.

    Rules:
    - Use only tool names from the allow-list.
    - Use only parameter names from the tool signature hints.
    - Include all required arguments.
    - Do not invent column names; only use columns that exist in the schema.
    - The router should include only analysis parameters in args.
    - Do not include filesystem paths, report directories, or session folders.
    - Do not include explanations, reasoning, prose, bullet points, or markdown fences.
    - Do not show your checklist or intermediate reasoning.
    - Return exactly one JSON object and nothing else.

    Output schema:

    For tool use:
    {
        "mode": "tool",
        "tool": "<exact_tool_name>",
        "args": {
            "<arg_name>": <value>
        }
    }

    For code generation:
    {
        "mode": "codegen",
        "plan": "<brief plan>",
        "codegen_instructions": "<what code should do>"
    }

    Validation requirements:
    - The top-level key "mode" is required.
    - "mode" must be exactly "tool" or "codegen".
    - If "mode" is "tool", include both "tool" and "args".
    - If "mode" is "codegen", include both "plan" and "codegen_instructions".
    - Do not output any keys outside the selected schema unless necessary.
    - Return only the JSON object.
    """)

    return router_system_text.replace("{{allowed_tools_text}}", allow_str).replace(
        "{{tool_arg_hints}}", tool_arg_hints
    )


def build_router_chain(
    *,
    system_text: str,
    model: str,
    temperature: float = 0.0,
    stream: bool = False,
):
    """
    Build router chain from a precompiled system prompt.

    The system_text can come from:
      - a Python string (CLI fallback)
      - a Langfuse-managed prompt (preferred for prompt management)
    """
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_text),
            (
                "human",
                "Dataset schema:\n{schema_text}\n\nUser request:\n{user_request}\n",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_results_summarizer_chain(
    model: str, temperature: float = 0.2, stream: bool = False
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)
    results_summarizer_system_text = """
        You are an expert at explaining data analysis results.
        Given a user request and tool outputs, do:
        1) What we ran (1-2 sentences)
        2) Key results (bullets)
        3) Interpretation (plain language)
        4) Caveats/assumptions (bullets)
        5) Next steps (2-3 suggestions)
        Do NOT invent results; use only what is provided.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=results_summarizer_system_text),
            ("human", "User request:\n{user_request}\n\nTool output:\n{tool_output}\n"),
        ]
    )
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------------------
# Execution (subprocess, not exec)
# --------------------------------------------------------------------------------------
@observe(name="execute-generated-script", as_type="span", capture_output=True)
def run_generated_script(
    script_path: Path, data_path: Path, report_dir: Path, timeout_s: int = 60
) -> subprocess.CompletedProcess:
    with propagate_attributes(tags=["build", "execute"]):
        cmd = [
            sys.executable,
            str(script_path),
            "--data",
            str(data_path),
            "--report_dir",
            str(report_dir),
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


HELP_TEXT = """Commands:
  help                         Show this help text
  schema                       Print dataset schema
  suggest <question>           Build1-style suggestions (LLM)
  ask <request>                ROUTER decides: tool-run OR codegen (HITL)
  tool <request>               Force tool-run: choose one Build0 tool + args (HITL)
  code <request>               Force code generation (HITL) + approve to save
  run                          Execute last approved script via subprocess (HITL)
  exit                         Quit

Examples:
  ask run a frequency table for sex
  ask fit a regression of bill_length_mm on flipper_length_mm and sex
  tool run a correlation heatmap for numeric columns
  code create a scatterplot of numeric column by numeric column and save it
"""


# --------------------------------------------------------------------------------------
# Traced wrappers
# --------------------------------------------------------------------------------------
@observe(name="build-suggest", as_type="span")
def traced_suggest(
    suggest_chain,
    schema_text: str,
    question: str,
    config: Dict[str, Any],
    stream: bool,
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "suggest"]):
        return invoke_chain_text(
            suggest_chain,
            {"schema_text": schema_text, "user_query": question},
            config=config,
            stream=stream,
        )


@observe(name="build-codegen", as_type="generation")
def traced_codegen(
    codegen_chain,
    schema_text: str,
    request: str,
    config: Dict[str, Any],
    stream: bool,
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "codegen"]):
        return invoke_chain_text(
            codegen_chain,
            {"schema_text": schema_text, "user_request": request},
            config=config,
            stream=stream,
        )


@observe(name="build-toolplan", as_type="generation")
def traced_toolplan(
    toolplan_chain,
    schema_text: str,
    request: str,
    config: Dict[str, Any],
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "toolplan"]):
        return toolplan_chain.invoke(
            {"schema_text": schema_text, "user_request": request}, config=config
        )


@observe(name="build-router", as_type="generation")
def traced_router(
    router_chain,
    router_prompt_obj,
    schema_text: str,
    request: str,
    config: Dict[str, Any],
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "router"]):
        if (
            LANGFUSE_AVAILABLE
            and langfuse is not None
            and router_prompt_obj is not None
        ):
            langfuse.update_current_generation(prompt=router_prompt_obj)

        return router_chain.invoke(
            {"schema_text": schema_text, "user_request": request},
            config=config,
        )


@observe(name="build-summarize", as_type="generation")
def traced_summarize(
    summarize_chain,
    request: str,
    tool_output: str,
    config: Dict[str, Any],
    tags: list[str],
) -> str:
    with propagate_attributes(tags=tags + ["build", "summarize"]):
        return summarize_chain.invoke(
            {"user_request": request, "tool_output": tool_output}, config=config
        )


@observe(name="build-run-tool", as_type="span", capture_output=False)
def traced_run_tool(
    tool_name: str,
    tool_fn: ToolFn,
    df: pd.DataFrame,
    report_dir: Path,
    tool_args: Dict[str, Any],
    tags: list[str],
) -> ToolResult:
    # --- Standard artifact folders (always present) ---
    tool_output_dir = report_dir / "tool_outputs"
    tool_figure_dir = report_dir / "tool_figures"
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    tool_figure_dir.mkdir(parents=True, exist_ok=True)

    # --- Signature inspection (once) ---
    try:
        sig = inspect.signature(tool_fn)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        sig = None
        params = {}
        accepts_kwargs = True  # safest fallback

    # --- Inject dirs into tool_args if the tool supports them and they're missing ---
    # Your plotting tools in src/plotting.py typically take fig_dir; keep this flexible.
    dir_defaults = {
        # figures
        "fig_dir": tool_figure_dir,
        "plot_dir": tool_figure_dir,
        "plots_dir": tool_figure_dir,
        "figure_dir": tool_figure_dir,
        "figures_dir": tool_figure_dir,
        # outputs
        "out_dir": tool_output_dir,
        "output_dir": tool_output_dir,
        "artifact_dir": tool_output_dir,
        # NOTE: we still pass report_dir separately below for tools that support it
    }

    for k, default_path in dir_defaults.items():
        if k not in tool_args and (k in params or accepts_kwargs):
            tool_args[k] = default_path

    # Normalize string paths → Path objects (helps if router emitted strings)
    for k in list(tool_args.keys()):
        if k in dir_defaults and isinstance(tool_args[k], str):
            tool_args[k] = Path(tool_args[k])

    # --- Trace + execute ---
    with propagate_attributes(
        tags=tags + ["build", "toolrun"],
        metadata={
            "tool": tool_name,
            "args": json.dumps(tool_args, ensure_ascii=False, default=str),
        },
    ):
        # Preserve your existing "report_dir if supported" behavior
        supports_report_dir = ("report_dir" in params) or accepts_kwargs

        if supports_report_dir:
            result = tool_fn(df, report_dir=report_dir, **tool_args)
        else:
            result = tool_fn(df, **tool_args)

        print("\n=== TOOL COMPLETE ===")
        print("Tool:", tool_name)
        print("Outputs saved to:", tool_output_dir)
        print("Figures saved to:", tool_figure_dir)
        print()

        return normalize_tool_return(tool_name, result)


# --------------------------------------------------------------------------------------
# Core routines (HITL)
# --------------------------------------------------------------------------------------
def do_tool_run(
    *,
    req: str,
    toolplan_chain,
    summarize_chain,
    tools: Dict[str, ToolFn],
    allowed_tools: list[str],
    df: pd.DataFrame,
    df_columns: set[str],
    report_dir: Path,
    schema_text: str,
    base_config: Dict[str, Any],
    tags: list[str],
) -> None:
    """Tool planner -> HITL approve -> run tool -> save output -> summarize."""
    toolplan_raw = traced_toolplan(toolplan_chain, schema_text, req, base_config, tags)
    plan = parse_json_object(toolplan_raw)
    if not plan:
        print("\nERROR: Tool planner did not return valid JSON. Try again.\n")
        print("Raw output was:\n", toolplan_raw, "\n")
        return

    do_tool_run_from_plan(
        req=req,
        plan=plan,
        summarize_chain=summarize_chain,
        tools=tools,
        allowed_tools=allowed_tools,
        df=df,
        df_columns=df_columns,
        report_dir=report_dir,
        base_config=base_config,
        tags=tags,
        title="TOOL PLAN",
    )


def do_tool_run_from_plan(
    *,
    req: str,
    plan: Dict[str, Any],
    summarize_chain,
    tools: Dict[str, ToolFn],
    allowed_tools: list[str],
    df: pd.DataFrame,
    df_columns: set[str],
    report_dir: Path,
    base_config: Dict[str, Any],
    tags: list[str],
    title: str = "TOOL PLAN",
) -> None:
    """Run a validated tool plan directly (used by router to avoid a second LLM plan call)."""
    tool_name = plan.get("tool")
    tool_args = coerce_tool_args(plan.get("args", {}))
    note = plan.get("note", "")

    print(f"\n=== {title} ===")
    print(json.dumps(plan, indent=2))
    if note:
        print(f"\nNote: {note}")
    print()

    if tool_name not in tools:
        print(f"\nERROR: Proposed tool '{tool_name}' is not in TOOLS registry.\n")
        print(f"Available tools: {', '.join(allowed_tools)}\n")
        return

    unknown_cols = find_unknown_columns(tool_args, df_columns)
    if unknown_cols:
        print("\nERROR: Tool args reference unknown columns.\n")
        print("Unknown columns:", ", ".join(sorted(unknown_cols)), "\n")
        return

    confirm = input(f"Run tool '{tool_name}' now? (y/n) ").strip().lower()
    if confirm != "y":
        print("\nTool execution not approved.\n")
        return

    try:
        res = traced_run_tool(
            tool_name, tools[tool_name], df, report_dir, tool_args, tags
        )
    except Exception as e:
        print(f"\nERROR running tool: {e}\n")
        return

    out_txt = report_dir / "tool_outputs" / f"{tool_name}_output.txt"

    save_text(out_txt, res.text)
    if res.artifact_paths:
        print("Artifacts:")
        for p in res.artifact_paths:
            print(f"- {p}")
            print()

    print(f"\nSaved tool output to: {out_txt}\n")

    summary = traced_summarize(summarize_chain, req, res.text, base_config, tags)
    print("\n=== INTERPRETATION & SUMMARY ===\n")
    print(summary + "\n")


def do_codegen(
    *,
    req: str,
    codegen_chain,
    schema_text: str,
    base_config: Dict[str, Any],
    stream: bool,
    tags: list[str],
    script_path: Path,
    state: Dict[str, Any],
    rag_index: Optional[RagIndex] = None,
    rag_k: int = 4,
) -> None:
    codegen_request, rag_context = prepare_codegen_request_with_rag(
        req=req,
        schema_text=schema_text,
        rag_index=rag_index,
        rag_k=rag_k,
    )

    if rag_context:
        print("\n=== RAG CONTEXT RETRIEVED FOR CODEGEN ===\n")
        print(rag_context + "\n")

    out = traced_codegen(
        codegen_chain, schema_text, codegen_request, base_config, stream, tags
    )

    candidate = extract_python_code(out)

    if not candidate:
        print(
            "WARNING: No fenced ```python code block found. Ask again and require it.\n"
        )
        return

    _, _, verify = split_sections(out)
    print("=== HUMAN VERIFICATION CHECKLIST (from model) ===")
    print((verify + "\n") if verify else "(No VERIFY section found.)\n")

    approve = input("Approve and save this code? (y/n) ").strip().lower()
    if approve != "y":
        print("\nCode not approved.\n")
        return

    state["code_approved"] = candidate
    save_text(script_path, candidate)
    print(f"\nApproved and saved to: {script_path}\n")
    print("Next: type 'run' to execute, or 'ask <request>' to route another request.\n")


def do_execute(
    *,
    script_path: Path,
    data_path: Path,
    report_dir: Path,
    timeout_s: int,
    state: Dict[str, Any],
) -> None:
    if not state.get("code_approved") or not script_path.exists():
        print(
            "\nNo approved script found yet. Use: code <request> (or ask <request> that routes to codegen)\n"
        )
        return

    confirm = input(f"Execute {script_path.name} now? (y/n) ").strip().lower()
    if confirm != "y":
        print("\nExecution not approved.\n")
        return

    print("\nRunning generated script...\n")
    run_log_path = report_dir / "run_log.txt"
    try:
        result = run_generated_script(
            script_path, data_path, report_dir, timeout_s=timeout_s
        )
    except subprocess.TimeoutExpired:
        msg = f"ERROR: Script timed out after {timeout_s} seconds.\n"
        save_text(run_log_path, msg)
        print(msg)
        return

    log = []
    log.append("=== COMMAND ===\n")
    log.append(
        f"{sys.executable} {script_path} --data {data_path} --report_dir {report_dir}\n\n"
    )
    log.append("=== STDOUT ===\n")
    log.append(result.stdout or "(empty)\n")
    log.append("\n=== STDERR ===\n")
    log.append(result.stderr or "(empty)\n")
    log.append(f"\n=== RETURN CODE ===\n{result.returncode}\n")
    save_text(run_log_path, "".join(log))

    print(f"Finished. Return code: {result.returncode}")
    print(f"Saved execution log to: {run_log_path}\n")


def do_router(
    *,
    req: str,
    router_chain,
    router_prompt_obj,
    codegen_chain,
    summarize_chain,
    tools: Dict[str, ToolFn],
    allowed_tools: list[str],
    df: pd.DataFrame,
    df_columns: set[str],
    report_dir: Path,
    schema_text: str,
    base_config: Dict[str, Any],
    stream: bool,
    tags: list[str],
    script_path: Path,
    rag_index: Optional[RagIndex] = None,
    rag_k: int = 4,
    state: Dict[str, Any],
) -> None:
    """
    Router -> (tool-run OR codegen).

    If router selects tool mode but no matching tool exists in TOOLS,
    fall back to code generation.

    This version is intentionally defensive:
    - accepts the ideal schema with "mode"
    - recovers if the LLM forgets "mode" but clearly returned a tool/codegen shape
    - tolerates either "codegen_instructions" or older "code_request" naming
    """
    raw = traced_router(
        router_chain,
        router_prompt_obj,
        schema_text,
        req,
        base_config,
        tags,
    )
    plan = parse_json_object(raw)

    if not plan:
        print("\nERROR: Router did not return valid JSON. Try again.\n")
        print("Raw output was:\n", raw, "\n")
        return

    if not isinstance(plan, dict):
        print("\nERROR: Router returned JSON, but not a JSON object. Try again.\n")
        print("Raw output was:\n", raw, "\n")
        return

    # ------------------------------------------------------------
    # Recover missing mode when the router output shape is obvious
    # ------------------------------------------------------------
    mode = str(plan.get("mode") or "").strip().lower()

    if not mode:
        if "tool" in plan and "args" in plan:
            mode = "tool"
            plan["mode"] = "tool"
        elif "plan" in plan and "codegen_instructions" in plan:
            mode = "codegen"
            plan["mode"] = "codegen"
        elif "code_request" in plan:
            mode = "codegen"
            plan["mode"] = "codegen"

    note = str(plan.get("note") or "").strip()

    print("\n=== ROUTER DECISION ===")
    print(json.dumps(plan, indent=2))
    if note:
        print(f"\nNote: {note}")
    print()

    # ------------------------------------------------------------
    # TOOL MODE
    # ------------------------------------------------------------
    if mode == "tool":
        router_tool = str(plan.get("tool") or "").strip()
        router_args = plan.get("args", {})

        if not router_tool:
            print("\nERROR: Router chose tool mode but did not provide a tool name.\n")
            print("Raw output was:\n", raw, "\n")
            return

        if not isinstance(router_args, dict):
            print("\nERROR: Router chose tool mode but 'args' is not a JSON object.\n")
            print("Raw output was:\n", raw, "\n")
            return

        if router_tool not in tools:
            print(
                "Router fallback: no matching tool is available in TOOLS. "
                "Falling back to code generation.\n"
            )
            do_codegen(
                req=req,
                codegen_chain=codegen_chain,
                schema_text=schema_text,
                base_config=base_config,
                stream=stream,
                tags=tags,
                script_path=script_path,
                state=state,
                rag_index=rag_index,
                rag_k=rag_k,
            )
            return

        do_tool_run_from_plan(
            req=req,
            plan=plan,
            summarize_chain=summarize_chain,
            tools=tools,
            allowed_tools=allowed_tools,
            df=df,
            df_columns=df_columns,
            report_dir=report_dir,
            base_config=base_config,
            tags=tags,
            title="TOOL PLAN (from router)",
        )
        return

    # ------------------------------------------------------------
    # CODEGEN MODE
    # ------------------------------------------------------------
    if mode == "codegen":
        code_req = plan.get("codegen_instructions") or plan.get("code_request") or req
        code_req = str(code_req).strip()

        do_codegen(
            req=code_req,
            codegen_chain=codegen_chain,
            schema_text=schema_text,
            base_config=base_config,
            stream=stream,
            tags=tags,
            script_path=script_path,
            state=state,
            rag_index=rag_index,
            rag_k=rag_k,
        )
        return

    # ------------------------------------------------------------
    # INVALID MODE
    # ------------------------------------------------------------
    print("\nERROR: Router 'mode' must be 'tool' or 'codegen'. Try again.\n")
    print("Raw output was:\n", raw, "\n")


# --------------------------------------------------------------------------------------
# Streamlit backend helpers
# --------------------------------------------------------------------------------------
def initialize_build4_backend(
    *,
    data_path: Path,
    report_dir: Path,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    memory: bool = False,
    stream: bool = False,
    session_id: str = "streamlit-session",
    knowledge_dir: Optional[Path] = None,
    rag_k: int = 4,
    tags: Optional[list[str]] = None,
):
    """
    Initialize everything needed for the UI and return a reusable backend state dict.
    """
    tags = tags or ["build4", "streamlit"]

    ensure_dirs(report_dir)
    ensure_dirs(report_dir / "tool_outputs")

    df = read_data(data_path)
    df_columns = set(df.columns)
    schema_text = profile_to_schema_text(basic_profile(df))

    tools = load_tools()
    allowed_tools = sorted(tools.keys())
    tool_descriptions = load_tool_descriptions()
    tool_arg_hints = format_tool_arg_hints(tools, allowed_tools)

    rag_index: Optional[RagIndex] = None
    if knowledge_dir is not None:
        if not knowledge_dir.exists():
            raise FileNotFoundError(f"knowledge_dir does not exist: {knowledge_dir}")
        rag_index = load_saved_rag_index(knowledge_dir)

    router_prompt_cfg: Dict[str, Any] = {}
    router_prompt_obj = None

    try:
        router_system_text, router_prompt_obj, router_prompt_cfg = (
            compile_router_prompt_from_langfuse(
                prompt_name="build_router_system",
                label="dev",
                allowed_tools=allowed_tools,
                tool_descriptions=tool_descriptions,
                tool_arg_hints=tool_arg_hints,
            )
        )
    except Exception:
        router_system_text = build_router_system_text_fallback(
            allowed_tools=allowed_tools,
            tool_descriptions=tool_descriptions,
            tool_arg_hints=tool_arg_hints,
        )

    router_defaults = get_prompt_config_defaults(router_prompt_cfg)
    effective_router_model = model or router_defaults["model"]
    effective_router_stream = stream or router_defaults["stream"]

    suggest_chain = build_suggest_chain(model, temperature, stream, memory)
    codegen_chain = build_codegen_chain(model, temperature, stream, memory)
    toolplan_chain = build_toolplan_chain(
        model,
        allowed_tools=allowed_tools,
        tool_descriptions=tool_descriptions,
        tool_arg_hints=tool_arg_hints,
        temperature=0.0,
        stream=stream,
    )
    router_chain = build_router_chain(
        system_text=router_system_text,
        model=effective_router_model,
        temperature=0.0,
        stream=effective_router_stream,
    )
    summarize_chain = build_results_summarizer_chain(model, temperature, stream)

    base_config = make_langfuse_config(session_id=session_id, tags=tags)
    script_path = report_dir / "build4_generated_analysis.py"

    return {
        "df": df,
        "df_columns": df_columns,
        "schema_text": schema_text,
        "tools": tools,
        "allowed_tools": allowed_tools,
        "tool_descriptions": tool_descriptions,
        "tool_arg_hints": tool_arg_hints,
        "rag_index": rag_index,
        "rag_k": rag_k,
        "router_prompt_obj": router_prompt_obj,
        "suggest_chain": suggest_chain,
        "codegen_chain": codegen_chain,
        "toolplan_chain": toolplan_chain,
        "router_chain": router_chain,
        "summarize_chain": summarize_chain,
        "base_config": base_config,
        "script_path": script_path,
        "report_dir": report_dir,
        "data_path": data_path,
        "tags": tags,
        "stream": stream,
        "temperature": temperature,
        "model": model,
    }


def ui_run_suggest(backend: Dict[str, Any], question: str) -> str:
    return traced_suggest(
        backend["suggest_chain"],
        backend["schema_text"],
        question,
        backend["base_config"],
        backend["stream"],
        backend["tags"],
    )


def ui_plan_tool(backend: Dict[str, Any], request: str) -> Dict[str, Any]:
    raw = traced_toolplan(
        backend["toolplan_chain"],
        backend["schema_text"],
        request,
        backend["base_config"],
        backend["tags"],
    )
    plan = parse_json_object(raw)
    return {"raw": raw, "plan": plan}


def ui_run_tool_from_plan(
    backend: Dict[str, Any],
    request: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    tool_name_raw = plan.get("tool")
    if not isinstance(tool_name_raw, str) or not tool_name_raw:
        return {"ok": False, "error": "Invalid or missing tool name in plan."}

    tool_name = tool_name_raw
    tool_args = coerce_tool_args(plan.get("args", {}))

    if tool_name not in backend["tools"]:
        return {"ok": False, "error": f"Tool '{tool_name}' is not in the registry."}

    unknown_cols = find_unknown_columns(tool_args, backend["df_columns"])
    if unknown_cols:
        return {
            "ok": False,
            "error": f"Unknown columns referenced: {', '.join(sorted(unknown_cols))}",
        }

    try:
        res = traced_run_tool(
            tool_name,
            backend["tools"][tool_name],
            backend["df"],
            backend["report_dir"],
            tool_args,
            backend["tags"],
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    out_txt = backend["report_dir"] / "tool_outputs" / f"{tool_name}_output.txt"
    save_text(out_txt, res.text)

    summary = traced_summarize(
        backend["summarize_chain"],
        request,
        res.text,
        backend["base_config"],
        backend["tags"],
    )

    return {
        "ok": True,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_text": res.text,
        "summary": summary,
        "artifact_paths": res.artifact_paths,
        "output_txt": str(out_txt),
    }


def ui_run_codegen(backend: Dict[str, Any], request: str) -> Dict[str, Any]:
    codegen_request, rag_context = prepare_codegen_request_with_rag(
        req=request,
        schema_text=backend["schema_text"],
        rag_index=backend["rag_index"],
        rag_k=backend["rag_k"],
    )

    out = traced_codegen(
        backend["codegen_chain"],
        backend["schema_text"],
        codegen_request,
        backend["base_config"],
        backend["stream"],
        backend["tags"],
    )

    candidate = extract_python_code(out)
    plan_text, _, verify_text = split_sections(out)

    return {
        "raw": out,
        "code": candidate,
        "plan_text": plan_text,
        "verify_text": verify_text,
        "rag_context": rag_context,
    }


def ui_save_generated_code(backend: Dict[str, Any], code: str) -> str:
    save_text(backend["script_path"], code)
    return str(backend["script_path"])


def ui_run_saved_code(backend: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    if not backend["script_path"].exists():
        return {
            "ok": False,
            "error": "No saved generated script exists yet. Save code first.",
        }

    report_dir = backend["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    script_path = backend["script_path"].resolve()
    run_log_path = (report_dir / "run_log.txt").resolve()

    def snapshot_files(root: Path) -> Dict[str, float]:
        snap = {}
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    snap[str(p.resolve())] = p.stat().st_mtime_ns
                except FileNotFoundError:
                    # File may disappear between scan and stat; skip it
                    continue
        return snap

    before_files = snapshot_files(report_dir)

    try:
        result = run_generated_script(
            backend["script_path"],
            backend["data_path"],
            backend["report_dir"],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Script timed out after {timeout_s} seconds."}

    log = []
    log.append("=== STDOUT ===\n")
    log.append(result.stdout or "(empty)\n")
    log.append("\n=== STDERR ===\n")
    log.append(result.stderr or "(empty)\n")
    log.append(f"\n=== RETURN CODE ===\n{result.returncode}\n")
    save_text(run_log_path, "".join(log))

    after_files = snapshot_files(report_dir)

    excluded_paths = {str(script_path), str(run_log_path)}

    artifact_paths = []
    for path_str, after_mtime in after_files.items():
        if path_str in excluded_paths:
            continue

        before_mtime = before_files.get(path_str)

        # Include file if it is new or modified during this run
        if before_mtime is None or after_mtime > before_mtime:
            artifact_paths.append(path_str)

    artifact_paths = sorted(artifact_paths)

    return {
        "ok": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "run_log_path": str(run_log_path),
        "artifact_paths": artifact_paths,
        "has_artifacts": len(artifact_paths) > 0,
        "artifact_message": (
            f"{len(artifact_paths)} generated artifact(s) detected."
            if artifact_paths
            else "No new or modified artifacts were detected in the report directory."
        ),
    }


def prepare_router_request_with_rag(
    req: str,
    schema_text: str,
    rag_index: Optional[RagIndex],
    rag_k: int = 4,
) -> tuple[str, Optional[str]]:
    if rag_index is None:
        return req, None

    retrieval_query = f"""
    User request: {req}

    Dataset schema:
    {schema_text}

    Retrieve guidance about:
    - which existing tool should be used
    - when to prefer a tool over code generation
    - valid mappings from request types to tool names
    """.strip()

    results = retrieve_chunks(
        query=retrieval_query,
        index=rag_index.index,
        chunks=rag_index.chunks,
        k=rag_k,
        embedding_model=rag_index.embedding_model,
    )
    rag_context = format_rag_context(results)

    router_request = dedent(f"""
    Retrieved guidance for routing:
    {rag_context}

    Original user request:
    {req}

    Prefer an existing tool whenever one can directly satisfy the request.
    Use codegen only when no existing tool is appropriate.
    """).strip()

    return router_request, rag_context


def ui_run_router(backend: Dict[str, Any], request: str) -> Dict[str, Any]:
    router_request, router_rag_context = prepare_router_request_with_rag(
        req=request,
        schema_text=backend["schema_text"],
        rag_index=backend["rag_index"],
        rag_k=backend["rag_k"],
    )

    raw = traced_router(
        backend["router_chain"],
        backend["router_prompt_obj"],
        backend["schema_text"],
        router_request,
        backend["base_config"],
        backend["tags"],
    )

    plan = parse_json_object(raw)
    if not isinstance(plan, dict):
        return {"ok": False, "raw": raw, "error": "Router did not return valid JSON."}

    mode = str(plan.get("mode") or "").strip().lower()
    if not mode:
        if "tool" in plan and "args" in plan:
            mode = "tool"
            plan["mode"] = "tool"
        elif "plan" in plan and "codegen_instructions" in plan:
            mode = "codegen"
            plan["mode"] = "codegen"
        elif "code_request" in plan:
            mode = "codegen"
            plan["mode"] = "codegen"

    return {
        "ok": True,
        "raw": raw,
        "plan": plan,
        "mode": mode,
        "rag_context": router_rag_context,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="build4: HITL + Router + RAG + Langfuse"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--out_file", type=str, default="build4_generated_analysis.py")
    parser.add_argument("--timeout_s", type=int, default=60)
    parser.add_argument("--session_id", type=str, default="cli-session")
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        default=None,
        help="Optional path to a knowledge/ folder containing a prebuilt FAISS RAG index and chunk metadata",
    )
    parser.add_argument(
        "--rag_k",
        type=int,
        default=4,
        help="Number of retrieved chunks to inject into codegen",
    )

    parser.add_argument(
        "--tags", type=str, default="build4", help="Comma-separated Langfuse tags"
    )
    args = parser.parse_args()

    tag_list = parse_tags(args.tags)

    data_path = Path(args.data)
    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)
    ensure_dirs(report_dir / "tool_outputs")

    # Load data + schema
    df = read_data(data_path)
    df_columns = set(df.columns)
    schema_text = profile_to_schema_text(basic_profile(df))

    # Load Build0 tools registry
    tools = load_tools()
    allowed_tools = sorted(tools.keys())
    tool_descriptions = load_tool_descriptions()
    tool_arg_hints = format_tool_arg_hints(tools, allowed_tools)

    # Optional Build4 RAG index (used on the codegen path only)
    rag_index: Optional[RagIndex] = None
    if args.knowledge_dir:
        knowledge_dir = Path(args.knowledge_dir)
        if not knowledge_dir.exists():
            raise FileNotFoundError(f"knowledge_dir does not exist: {knowledge_dir}")
        print(f"\nLoading saved FAISS RAG index from: {knowledge_dir}")
        try:
            rag_index = load_saved_rag_index(knowledge_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n\nBuild the RAG index first with:\n"
                f"  python build_rag_index.py --knowledge_dir {knowledge_dir}"
            ) from e
        print(
            f"RAG ready: {len(rag_index.chunks)} chunks loaded using {rag_index.embedding_model}.\n"
        )

    # ------------------------------------------------------------
    # NEW: get router prompt text with fallback to built-in string
    # if Langfuse prompt not found
    # ------------------------------------------------------------
    router_prompt_cfg: Dict[str, Any] = {}
    router_prompt_obj = None

    try:
        router_system_text, router_prompt_obj, router_prompt_cfg = (
            compile_router_prompt_from_langfuse(
                prompt_name="build_router_system",
                label="dev",
                allowed_tools=allowed_tools,
                tool_descriptions=tool_descriptions,
                tool_arg_hints=tool_arg_hints,
            )
        )
        print("Router prompt: loaded from Langfuse")

    except Exception as e:
        print(f"Router prompt: Langfuse load failed ({e}); using local fallback.")
        router_system_text = build_router_system_text_fallback(
            allowed_tools=allowed_tools,
            tool_descriptions=tool_descriptions,
            tool_arg_hints=tool_arg_hints,
        )
        router_prompt_obj = None

    # Optional: read defaults from Langfuse config if present
    router_defaults = get_prompt_config_defaults(router_prompt_cfg)

    # Current CLI behavior:
    # args.model and args.stream act as runtime overrides
    effective_router_model = args.model or router_defaults["model"]
    effective_router_stream = args.stream or router_defaults["stream"]

    # Chains
    suggest_chain = build_suggest_chain(
        args.model, args.temperature, args.stream, args.memory
    )
    codegen_chain = build_codegen_chain(
        args.model, args.temperature, args.stream, args.memory
    )
    toolplan_chain = build_toolplan_chain(
        args.model,
        allowed_tools=allowed_tools,
        tool_descriptions=tool_descriptions,
        tool_arg_hints=tool_arg_hints,
        temperature=0.0,
        stream=args.stream,
    )
    router_chain = build_router_chain(
        system_text=router_system_text,
        model=effective_router_model,
        temperature=0.0,
        stream=effective_router_stream,
    )
    summarize_chain = build_results_summarizer_chain(
        args.model, args.temperature, args.stream
    )

    base_config = make_langfuse_config(session_id=args.session_id, tags=tag_list)

    script_path = report_dir / args.out_file

    print("\n=== build4: HITL + Router + RAG + Langfuse ===\n")
    print(f"Tags: {tag_list}")
    print(f"Build0 tools loaded: {', '.join(allowed_tools)}\n")

    if rag_index is not None:
        print(
            f"RAG: ENABLED ({len(rag_index.chunks)} chunks from {rag_index.knowledge_dir})\n"
        )
    else:
        print(
            "RAG: disabled (pass --knowledge_dir to enable Build4A retrieval on codegen)\n"
        )

    if LANGFUSE_AVAILABLE:
        print("Langfuse: ENABLED (CallbackHandler + observe decorator)\n")
    else:
        print("Langfuse: not installed or not available (running without tracing)\n")

    print("Type 'help' for commands. Type 'exit' to quit.\n")

    state: Dict[str, Any] = {"code_approved": None}

    while True:
        user_in = input("> ").strip()
        if not user_in:
            continue
        low = user_in.lower()

        if low in {"exit", "quit"}:
            print("Goodbye!")
            break

        if low == "help":
            print("\n" + HELP_TEXT + "\n")
            continue

        if low == "schema":
            print("\n=== DATASET SCHEMA ===")
            print(schema_text + "\n")
            continue

        if low.startswith("suggest "):
            q = user_in[len("suggest ") :].strip()
            if not q:
                print("\nUsage: suggest <question>\n")
                continue
            _ = traced_suggest(
                suggest_chain, schema_text, q, base_config, args.stream, tag_list
            )
            continue

        if low.startswith("ask "):
            req = user_in[len("ask ") :].strip()
            if not req:
                print("\nUsage: ask <analysis request>\n")
                continue
            do_router(
                req=req,
                router_chain=router_chain,
                router_prompt_obj=router_prompt_obj,
                codegen_chain=codegen_chain,
                summarize_chain=summarize_chain,
                tools=tools,
                allowed_tools=allowed_tools,
                df=df,
                df_columns=df_columns,
                report_dir=report_dir,
                schema_text=schema_text,
                base_config=base_config,
                stream=args.stream,
                tags=tag_list,
                script_path=script_path,
                state=state,
                rag_index=rag_index,
                rag_k=args.rag_k,
            )
            continue

        if low.startswith("tool "):
            req = user_in[len("tool ") :].strip()
            if not req:
                print("\nUsage: tool <analysis request>\n")
                continue
            do_tool_run(
                req=req,
                toolplan_chain=toolplan_chain,
                summarize_chain=summarize_chain,
                tools=tools,
                allowed_tools=allowed_tools,
                df=df,
                df_columns=df_columns,
                report_dir=report_dir,
                schema_text=schema_text,
                base_config=base_config,
                tags=tag_list,
            )
            continue

        if low.startswith("code "):
            req = user_in[len("code ") :].strip()
            if not req:
                print("\nUsage: code <analysis request>\n")
                continue
            do_codegen(
                req=req,
                codegen_chain=codegen_chain,
                schema_text=schema_text,
                base_config=base_config,
                stream=args.stream,
                tags=tag_list,
                script_path=script_path,
                rag_index=rag_index,
                rag_k=args.rag_k,
                state=state,
            )
            continue

        if low == "run":
            do_execute(
                script_path=script_path,
                data_path=data_path,
                report_dir=report_dir,
                timeout_s=args.timeout_s,
                state=state,
            )
            continue

        print("\nUnrecognized command. Type 'help' for options.\n")


if __name__ == "__main__":
    main()
