"""
Paste these helper functions into your backend file
build4_rag_router_agent_prompt_mgmt.py above main().

These wrappers convert the CLI-oriented Build4 agent into a Streamlit-friendly backend.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------------------
# Streamlit-friendly backend helpers
# --------------------------------------------------------------------------------------
def initialize_build4_backend(
    *,
    data_path: Path,
    report_dir: Path,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
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
    tool_name = plan.get("tool")
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

    try:
        result = run_generated_script(
            backend["script_path"],
            backend["data_path"],
            backend["report_dir"],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Script timed out after {timeout_s} seconds."}

    run_log_path = backend["report_dir"] / "run_log.txt"
    log = []
    log.append("=== STDOUT ===\n")
    log.append(result.stdout or "(empty)\n")
    log.append("\n=== STDERR ===\n")
    log.append(result.stderr or "(empty)\n")
    log.append(f"\n=== RETURN CODE ===\n{result.returncode}\n")
    save_text(run_log_path, "".join(log))

    return {
        "ok": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "run_log_path": str(run_log_path),
    }



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
