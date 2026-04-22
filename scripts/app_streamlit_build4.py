"""
Streamlit app for Build 4: RAG + Router + Agent with Prompt Management
This app provides a UI for interacting with the Build 4 backend, which includes:
- RAG retrieval from a knowledge base
- A router that decides whether to use tools or generate code
- An agent that can execute tools or run generated code
- Prompt management to keep track of the conversation and context
To run this app:
1. Make sure you have the Build 4 backend implemented in builds/build4_rag_router_agent_streamlit.py
2. Install Streamlit if you haven't: pip install streamlit
3. Run this script: streamlit run scripts/app_streamlit_build4.py

"""

from __future__ import annotations


import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Add project root to Python path for findingbuilds folder and importing src modules;
sys.path.append(str(Path(__file__).resolve().parents[1]))

# IMPORTANT:
# This import assumes your backend file is named exactly:
# builds/build4_rag_router_agent_streamlit.py
# If your local file has a different name, rename it first.
from builds.build4_rag_router_agent_streamlit import (
    initialize_build4_backend,
    ui_run_codegen,
    ui_run_router,
    ui_run_saved_code,
    ui_run_suggest,
    ui_plan_tool,
    ui_run_tool_from_plan,
    ui_save_generated_code,
)

st.set_page_config(page_title="Data Analysis Router Agent", layout="wide")

st.title("Data Analysis Router Agent")
st.caption(
    "This a Streamlit interface for a data analysis router agent. "
    "Use the sidebar to upload a CSV and start the agent, "
    "then explore the tabs below to interact with it."
)

st.info(
    "To start the agent: upload a CSV in the sidebar, choose your settings, "
    "and then click **Initialize Agent**."
)
# -----------------------------------------------------------------------------
# Session state: Memory that holds the backend object, last router result, last tool plan,
# last tool run result, and last generated code so actions can happen across
# multiple button clicks
# -----------------------------------------------------------------------------
if "backend" not in st.session_state:
    st.session_state.backend = None

if "uploaded_data_path" not in st.session_state:
    st.session_state.uploaded_data_path = None

if "last_code_codegen_result" not in st.session_state:
    st.session_state.last_code_codegen_result = None

if "last_router_result" not in st.session_state:
    st.session_state.last_router_result = None

if "last_tool_plan_result" not in st.session_state:
    st.session_state.last_tool_plan_result = None

if "last_tool_run_result" not in st.session_state:
    st.session_state.last_tool_run_result = None

if "last_execute_result" not in st.session_state:
    st.session_state.last_execute_result = None

if "backend_signature" not in st.session_state:
    st.session_state.backend_signature = None


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def save_uploaded_csv(uploaded_file) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "build4_streamlit_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = tmp_dir / uploaded_file.name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def render_download_button(path: Path, prefix: str, instance_id: str) -> None:
    with open(path, "rb") as f:
        st.download_button(
            label=f"Download {path.name}",
            data=f.read(),
            file_name=path.name,
            key=f"download_{prefix}_{instance_id}_{path.name}_{path.stat().st_mtime_ns}",
            width="content",
        )


def render_single_artifact(
    path: Path,
    prefix: str = "artifact",
    instance_id: str = "0",
) -> None:
    if not path.exists():
        st.warning(f"Missing artifact: {path}")
        return

    st.markdown(f"**{path.name}**")
    st.caption(str(path))

    suffix = path.suffix.lower()

    try:
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            st.image(str(path), caption=path.name, width="stretch")

        elif suffix == ".csv":
            df = pd.read_csv(path)
            st.dataframe(df, width="stretch")

        elif suffix in {".txt", ".log", ".py", ".md", ".json"}:
            text = safe_read_text(path)

            if suffix == ".py":
                st.code(text, language="python")

            elif suffix == ".json":
                st.code(text, language="json")

            else:
                st.text_area(
                    label=f"Preview: {path.name}",
                    value=text,
                    height=220,
                    key=f"text_{prefix}_{instance_id}_{path.name}_{path.stat().st_mtime_ns}",
                )

        else:
            st.info("Preview not available for this file type.")

    except Exception as e:
        st.warning(f"Could not preview {path.name}: {e}")

    render_download_button(path, prefix=prefix, instance_id=instance_id)


def render_artifacts(
    artifact_paths,
    title: str = "Artifacts",
    prefix: str = "artifact",
) -> None:
    if not artifact_paths:
        st.info("No artifacts were produced.")
        return

    st.subheader(title)

    unique_paths = []
    seen = set()

    for p in artifact_paths:
        p = Path(p)
        try:
            p_key = str(p.resolve())
        except Exception:
            p_key = str(p)

        if p_key not in seen:
            seen.add(p_key)
            unique_paths.append(p)

    for i, path in enumerate(unique_paths):
        render_single_artifact(path, prefix=prefix, instance_id=str(i))


def list_report_files(report_dir: Path) -> list[Path]:
    if not report_dir.exists():
        return []

    return sorted(
        [p for p in report_dir.rglob("*") if p.is_file()],
        key=lambda p: str(p).lower(),
    )


def render_report_browser(report_dir: Path) -> None:
    if not report_dir.exists():
        st.info(f"Report directory does not exist yet: {report_dir}")
        return

    report_files = list_report_files(report_dir)

    if not report_files:
        st.info("No saved reports or artifacts were found yet.")
        return

    st.caption(f"Report directory: {report_dir}")

    render_artifacts(
        report_files,
        title="All Saved Files",
        prefix="reports_artifact",
    )


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Get Started")
st.sidebar.markdown(
    """
    1. Upload a CSV dataset  
    2. Choose model settings  
    3. Click **Initialize Agent**  
    4. Then use the tabs to interact with the agent
    """
)
st.sidebar.caption(
    "Important: the app will not run analyses until you click **Initialize Agent**."
)

model = st.sidebar.text_input("Model", value="gpt-4o-mini")
st.sidebar.caption(
    "Set the temperature for the model. Lower values make the model more deterministic."
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
stream = st.sidebar.toggle("Streaming", value=False)
memory = st.sidebar.toggle("Conversation Memory", value=False)
timeout_s = st.sidebar.number_input(
    "Execution timeout (seconds)", min_value=10, max_value=600, value=60, step=10
)

report_dir_str = st.sidebar.text_input("Report directory", value="reports_streamlit")
knowledge_dir_str = st.sidebar.text_input(
    "Knowledge folder for RAG (optional)",
    value="",
    help="Enter the folder path to your RAG knowledge base if you want the agent to use RAG retrieval. Leave blank to skip RAG.",
)
rag_k = st.sidebar.number_input("RAG k", min_value=1, max_value=10, value=4, step=1)

st.sidebar.divider()
st.sidebar.subheader("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    st.session_state.uploaded_data_path = save_uploaded_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

init_clicked = st.sidebar.button("Initialize Agent", width="stretch")

if init_clicked:
    if st.session_state.uploaded_data_path is None:
        st.sidebar.error("Please upload a CSV file first.")
    else:
        knowledge_dir: Optional[Path] = None
        if knowledge_dir_str.strip():
            knowledge_dir = Path(knowledge_dir_str.strip())

        try:
            backend = initialize_build4_backend(
                data_path=Path(st.session_state.uploaded_data_path),
                report_dir=Path(report_dir_str),
                model=model,
                temperature=temperature,
                memory=memory,
                stream=stream,
                session_id="streamlit-session",
                knowledge_dir=knowledge_dir,
                rag_k=int(rag_k),
                tags=["build4", "streamlit"],
            )
            st.session_state.backend = backend
            st.session_state.backend_signature = {
                "data_path": str(st.session_state.uploaded_data_path),
                "model": model,
                "temperature": temperature,
                "memory": memory,
                "stream": stream,
                "knowledge_dir": str(knowledge_dir) if knowledge_dir else "",
                "rag_k": int(rag_k),
            }
            st.success("Agent initialized successfully.")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

backend = st.session_state.backend


# -----------------------------------------------------------------------------
# Dataset + schema
# -----------------------------------------------------------------------------
st.header("1) Dataset and Schema")

if backend is None:
    st.info("Upload a CSV and click 'Initialize Agent' in the sidebar.")
else:
    df = backend["df"]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Schema")
        st.text(backend["schema_text"])

    with right:
        st.subheader("Preview")
        st.dataframe(df.head(20), width="stretch")

    with st.expander("Column details"):
        schema_df = pd.DataFrame(
            {
                "column": list(df.columns),
                "dtype": [str(df[c].dtype) for c in df.columns],
                "missing_n": [int(df[c].isna().sum()) for c in df.columns],
            }
        )
        st.dataframe(
            schema_df,
            width="stretch",
        )

    with st.expander("Loaded tools"):
        st.write(backend["allowed_tools"])


# -----------------------------------------------------------------------------
# Command interface
# -----------------------------------------------------------------------------
st.header("2) Agent Commands")

if backend is not None:
    tab_suggest, tab_ask, tab_tool, tab_code, tab_run, tab_reports = st.tabs(
        ["Suggest", "Ask", "Tool", "Code", "Run", "Reports"]
    )

    # -----------------------------------------------------------------
    # Suggest
    # -----------------------------------------------------------------
    with tab_suggest:
        st.subheader("suggest")
        suggest_q = st.text_area(
            "Enter a dataset question or ask for possible analyses",
            placeholder="Example: What are 3 good research questions I can test with this dataset?",
            key="suggest_q",
        )

        if st.button("Run suggest", key="btn_suggest"):
            if suggest_q.strip():
                with st.spinner("Generating suggestions..."):
                    out = ui_run_suggest(backend, suggest_q.strip())
                st.markdown(out)
            else:
                st.warning("Enter a question first.")

    # -----------------------------------------------------------------
    # Ask
    # -----------------------------------------------------------------
    with tab_ask:
        st.subheader("ask")

        ask_req = st.text_area(
            "Enter your data analysis request.",
            placeholder="Example: Predict body mass from sex and bill length",
            key="ask_req",
        )

        if st.button("Route request", key="btn_route_request"):
            if ask_req.strip():
                st.session_state.last_router_result = None
                st.session_state.last_tool_run_result = None
                st.session_state.last_ask_codegen_result = None
                st.session_state.pop("ask_router_plan_decision", None)
                st.session_state.pop("ask_codegen_decision", None)
                st.session_state.last_execute_result = None

                if backend["script_path"].exists():
                    backend["script_path"].unlink()

                with st.spinner("Routing request..."):
                    result = ui_run_router(backend, ask_req.strip())

                st.session_state.last_router_result = result
            else:
                st.warning("Enter a request first.")

        router_result = st.session_state.get("last_router_result")

        if router_result:
            if not router_result.get("ok"):
                st.error(router_result.get("error", "Unknown routing error"))
                st.code(router_result.get("raw", ""), language="json")

            else:
                st.write(f"**Router mode:** {router_result['mode']}")

                if router_result.get("rag_context"):
                    with st.expander("Router RAG context"):
                        st.text(router_result["rag_context"])

                st.write("**Parsed router plan**")
                st.code(str(router_result["plan"]), language="python")

                if router_result["mode"] == "tool":
                    if st.button("Approve and run tool", key="approve_router_tool"):
                        with st.spinner("Running tool..."):
                            run_res = ui_run_tool_from_plan(
                                backend,
                                ask_req.strip(),
                                router_result["plan"],
                            )
                        st.session_state.last_tool_run_result = run_res

                    tool_run_res = st.session_state.get("last_tool_run_result")
                    if tool_run_res:
                        if tool_run_res.get("ok"):
                            st.success(f"Tool ran: {tool_run_res['tool_name']}")
                            st.write("**Tool output**")
                            st.text(tool_run_res["tool_text"])
                            st.write("**Summary**")
                            st.markdown(tool_run_res["summary"])
                            render_artifacts(
                                tool_run_res.get("artifact_paths", []),
                                title="Tool Artifacts",
                                prefix="ask_artifact",
                            )
                        else:
                            st.error(tool_run_res["error"])

                elif router_result["mode"] == "codegen":
                    code_req = (
                        router_result["plan"].get("codegen_instructions")
                        or router_result["plan"].get("code_request")
                        or ask_req.strip()
                    )

                    if "ask_router_plan_decision" not in st.session_state:
                        st.session_state.ask_router_plan_decision = "Review only"

                    if "ask_codegen_decision" not in st.session_state:
                        st.session_state.ask_codegen_decision = "Review only"

                    st.write("**Proposed code generation request**")
                    st.text(code_req)

                    plan_decision = st.radio(
                        "Do you want to approve this analysis plan?",
                        options=["Review only", "Approve plan", "Discard plan"],
                        horizontal=True,
                        key="ask_router_plan_decision",
                    )

                    if plan_decision == "Discard plan":
                        st.warning(
                            "Plan discarded. Revise the request and click 'Route request' again."
                        )

                    elif plan_decision == "Approve plan":
                        if st.button(
                            "Generate code from approved plan",
                            key="approve_router_codegen",
                        ):
                            with st.spinner("Generating code..."):
                                cg = ui_run_codegen(backend, code_req)
                            st.session_state.last_ask_codegen_result = cg

                    cg = st.session_state.get("last_ask_codegen_result")

                    if cg:
                        if cg.get("rag_context"):
                            with st.expander("Codegen RAG context"):
                                st.text(cg["rag_context"])

                        st.write("**Plan**")
                        st.text(cg.get("plan_text", ""))

                        st.write("**Verification checklist**")
                        st.text(cg.get("verify_text", ""))

                        if cg.get("code"):
                            st.code(cg["code"], language="python")

                        decision = st.radio(
                            "What would you like to do with this generated code?",
                            options=["Review only", "Save", "Discard"],
                            horizontal=True,
                            key="ask_codegen_decision",
                        )

                        if decision == "Save":
                            if st.button(
                                "Save generated code",
                                key="btn_save_ask_generated_code",
                            ):
                                saved_path = ui_save_generated_code(backend, cg["code"])
                                st.success(f"Saved to: {saved_path}")
                                render_artifacts(
                                    [saved_path],
                                    title="Saved Script",
                                    prefix="ask_saved_script",
                                )

                        elif decision == "Discard":
                            if st.button(
                                "Discard generated code",
                                key="btn_discard_ask_generated_code",
                            ):
                                st.session_state.last_ask_codegen_result = None
                                st.session_state.pop("ask_codegen_decision", None)
                                st.success("Generated code discarded.")
                                st.rerun()

    # -----------------------------------------------------------------
    # Tool
    # -----------------------------------------------------------------
    with tab_tool:
        st.subheader("tool")
        tool_req = st.text_area(
            "Force tool mode",
            placeholder="Example: Create a correlation heatmap for numeric variables",
            key="tool_req",
        )

        if st.button("Plan tool", key="btn_plan_tool"):
            if tool_req.strip():
                with st.spinner("Planning tool..."):
                    result = ui_plan_tool(backend, tool_req.strip())
                st.session_state.last_tool_plan_result = result
            else:
                st.warning("Enter a request first.")

        tool_plan_result = st.session_state.get("last_tool_plan_result")
        if tool_plan_result:
            st.write("**Raw planner output**")
            st.code(tool_plan_result["raw"], language="json")

            st.write("**Parsed plan**")
            st.code(str(tool_plan_result["plan"]), language="python")

            if st.button("Approve and run planned tool", key="btn_run_planned_tool"):
                plan = tool_plan_result["plan"]
                if not plan:
                    st.error("Planner did not return valid JSON.")
                else:
                    with st.spinner("Running tool..."):
                        run_res = ui_run_tool_from_plan(backend, tool_req.strip(), plan)
                    st.session_state.last_tool_run_result = run_res

            tool_run_res = st.session_state.get("last_tool_run_result")
            if tool_run_res:
                if tool_run_res.get("ok"):
                    st.success(f"Tool ran: {tool_run_res['tool_name']}")
                    st.write("**Tool output**")
                    st.text(tool_run_res["tool_text"])
                    st.write("**Summary**")
                    st.markdown(tool_run_res["summary"])
                    render_artifacts(
                        tool_run_res.get("artifact_paths", []),
                        title="Tool Artifacts",
                        prefix="tool_artifact",
                    )
                else:
                    st.error(tool_run_res["error"])

    # -----------------------------------------------------------------
    # Code
    # -----------------------------------------------------------------
    with tab_code:
        st.subheader("code")
        code_req = st.text_area(
            "Force code generation",
            placeholder="Example: Create a scatterplot matrix for all numeric columns and save it",
            key="code_req",
        )

        if st.button("Generate code", key="btn_generate_code"):
            if code_req.strip():
                st.session_state.last_code_codegen_result = None
                st.session_state.last_execute_result = None

                with st.spinner("Generating code..."):
                    cg = ui_run_codegen(backend, code_req.strip())
                st.session_state.last_code_codegen_result = cg
            else:
                st.warning("Enter a request first.")

        cg = st.session_state.get("last_code_codegen_result")
        if cg is not None:
            if cg.get("rag_context"):
                with st.expander("Codegen RAG context"):
                    st.text(cg["rag_context"])

            st.write("**Plan**")
            st.text(cg.get("plan_text", ""))
            st.write("**Verification checklist**")
            st.text(cg.get("verify_text", ""))

            if cg.get("code"):
                st.code(cg["code"], language="python")
                if st.button(
                    "Approve and save generated code", key="btn_save_generated_code"
                ):
                    saved_path = ui_save_generated_code(backend, cg["code"])
                    st.success(f"Saved to: {saved_path}")
                    render_artifacts(
                        [saved_path],
                        title="Saved Script",
                        prefix="saved_script",
                    )
                else:
                    st.error("No Python code block returned.")

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    with tab_run:
        st.subheader("run")
        st.write("Execute the last approved and saved generated script.")

        saved_script_exists = backend["script_path"].exists()
        st.write(
            f"**Saved script available:** {'Yes' if saved_script_exists else 'No'}"
        )

        if saved_script_exists:
            if st.button("Clear saved script", key="btn_clear_saved_script"):
                backend["script_path"].unlink()
                st.session_state.last_execute_result = None
                st.success("Saved script cleared.")
                st.rerun()

            if st.button("Run saved code", key="btn_run_saved_code"):
                with st.spinner("Executing script..."):
                    run_res = ui_run_saved_code(backend, timeout_s=int(timeout_s))
                st.session_state.last_execute_result = run_res
        else:
            st.info("No saved script is available yet.")

        run_res = st.session_state.get("last_execute_result")
        if run_res:
            if run_res["ok"]:
                st.success(f"Finished. Return code: {run_res['returncode']}")
                st.write("**STDOUT**")
                st.text(run_res["stdout"] or "(empty)")
                st.write("**STDERR**")
                st.text(run_res["stderr"] or "(empty)")

                render_artifacts(
                    [run_res["run_log_path"]],
                    title="Execution Log",
                    prefix="exec_log",
                )

                if run_res.get("has_artifacts"):
                    render_artifacts(
                        run_res.get("artifact_paths", []),
                        title="Generated Artifacts",
                        prefix="run_artifact",
                    )
                else:
                    st.info(
                        run_res.get(
                            "artifact_message",
                            "No generated artifacts were detected.",
                        )
                    )
            else:
                st.error(run_res["error"])

    # -----------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------
    with tab_reports:
        st.subheader("Saved reports and artifacts")
        if backend:
            render_report_browser(Path(backend["report_dir"]))
        else:
            st.info("Click **Initialize Agent** to unlock the command tabs.")
