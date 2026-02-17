"""
Build1: This build creates a simple LLM assistant using LangChain's LCEL framework
and imports build0 functions to load data and perform basic data set profiling.

Your job is to fill in the blanks to create a working interactive command-line assistant
that can answer questions about the dataset schema. Then, you will run the script 3 times in different modes
(no memory, memory, streaming). At each step, you should test the assistant by asking questions about the
dataset and observing how it responds. Make sure to try follow-up questions in memory mode to make sure
it retains context across interactions.

How to run it:
1) Make sure you have your environment activated and set up with the necessary libraries. There are new
libraries so run the requirements.txt file to install them.

2) Run the script with the --data argument pointing to your CSV file.

Run it 3 times to see the differences between no memory, memory, and streaming modes:

Run 1 (no memory):
python builds/build1_llm_assistant_assignment_2.py --data data/penguins.csv

This will start an interactive command-line interface where you can ask questions about the dataset.

Run 2 (with memory): run it again with the --memory flag to enable conversation memory:

python builds/build1_llm_assistant_assignment_2.py --data data/penguins.csv --memory

This allows the assistant to remember previous interactions in the same session,
which can lead to more coherent and context-aware responses. Try asking follow-up questions that
reference previous answers to ensure that memory is working as expected.

Run 3 (with streaming): you can also enable streaming output to see the model's response as it is generated:

python builds/build1_llm_assistant_assignment_2.py --data data/penguins.csv --memory --stream
"""

# This import allows us to use modern Python type hints which helps developers understand
# what types of data are expected)
# (e.g., list[str]), which can be used in function annotations even if the function is defined in a string.
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# import reusable functions from build0 (defined in src/__init__.py)
# These are just examples of functions you might have defined in build0 Adjust as needed.
# Add project root to Python path so it can find the src folder
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import ensure_dirs, read_data, basic_profile


# -------------------------------------------------------------------------------------------------
# TODO: Write your own SYSTEM PROMPT
# -------------------------------------------------------------------------------------------------
# Instructions:
# 1) Replace the text in SYSTEM_PROMPT with your own system prompt.
# 2) Your prompt MUST:
#    - Define the assistant's role (e.g., "You are a data analysis assistant for students.")
#    - State that the assistant ONLY sees the dataset schema (columns + dtypes)
#    - Instruct the model NOT to invent columns that are not in the schema
#    - Specify the output format (research questions + variables + analysis + clarifying questions)
#
# Tip: Keep it short and explicit. You can iterate after testing.
SYSTEM_PROMPT = """
TODO: Replace this with your own system prompt.

Required elements:
- Role
- Only sees schema
- No hallucinated columns
- Output format instructions
"""


# -------------------------------------------------------------------------------------------------
# Helper (supporting functions that are not part of the LCEL chain
# that help with formatting and other tasks)
# -------------------------------------------------------------------------------------------------

##################################################################################
# # TODO (Student): Complete the profile_to_schema_text section below.
##################################################################################


def profile_to_schema_text(profile: dict) -> str:
    """
    Convert basic_profile() output into a compact prompt-ready string.
    """

    lines = [
        f"Rows: {profile.get('n_rows')}",
        f"Columns: {profile.get('n_cols')}",
        "",
        "Columns and dtypes:",
    ]
    for col in profile["_____"]:
        lines.append(f"- {col}: {profile['dtypes'].get(col)}")

    return "\n".join(lines)


# funtion to build the LCEL chain, with optional streaming and memory support.
def build_chain(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    stream: bool = False,
    memory: bool = False,
):
    """
    Returns either:
      - a normal LCEL chain (no memory), OR
      - a RunnableWithMessageHistory (memory-enabled chain)
    """
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=stream)

    if memory:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _________),
                ("human", "Dataset schema:\n{schema_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User question:\n{user_query}"),
            ]
        )

        base_chain = prompt | ______ | StrOutputParser()

        history = InMemoryChatMessageHistory()
        chain_with_history = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: history,
            input_messages_key="user_query",
            history_messages_key="history",
        )
        return chain_with_history

    # No memory: simpler prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Dataset schema:\n{schema_text}\n\nUser question:\n{user_query}\n",
            ),
        ]
    )

    base_chain = prompt | llm | StrOutputParser()
    return base_chain


# help text for user commands and example prompts.
# This is just a simple string, but it could be expanded into a more complex
# help system if desired.
HELP_TEXT = """Commands:
  help     - show example prompts
  schema   - show the dataset schema
  exit     - quit the program

Example prompts:
  - What research questions could I ask with this dataset?
  - What are strong candidate outcomes vs predictors?
  - Suggest group comparison questions.
  - Suggest regression-style questions.
  - What variables might act as confounders?
"""


def main():
    """
    This function defines the entry point of the script.

    In larger AI or data analysis projects, files often serve two roles:
    1) As reusable modules (imported by other files)
    2) As runnable scripts (executed directly)

    Wrapping execution logic inside `main()` allows this file to act
    as a clean, reusable component in an agentic system while still
    supporting direct execution for testing and demos.

    The `if __name__ == "__main__":` guard ensures that this code runs
    only when explicitly intended, which becomes critical as systems
    grow more modular and interconnected.
    """

    load_dotenv()

    # ---------------------------------------------------------------------------------------------
    # TODO (Student): Complete the argparse section below.
    # Fill in the BLANKS (_____) so the script runs correctly.
    #
    # Hints:
    # - "--data" should be required and should accept a string path
    # - "--report_dir" should have default "reports"
    # - "--model" should default to "gpt-4o-mini"
    # - "--temperature" should default to 0.2
    # - "--quiet_schema", "--memory", and "--stream" should use action="store_true"
    # ---------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Build1 LLM Assistant (Interactive CLI)"
    )

    parser.add_argument(
        "_____",
        type=_____,
        required=_____,
        help="Path to CSV file",
    )
    parser.add_argument("--report_dir", type=str, default="_____")
    parser.add_argument("--model", type=str, default="_____")
    parser.add_argument("--temperature", type=float, default=_____)

    parser.add_argument(
        "--quiet_schema",
        action="_____",
        help="Do not print schema automatically at startup",
    )
    parser.add_argument(
        "--memory",
        action="_____",
        help="Enable conversation memory for this session",
    )
    parser.add_argument(
        "--stream",
        action="_____",
        help="Stream model output to terminal as it is generated",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    reports = Path(args.report_dir)

    ensure_dirs(reports)
    df = read_data(data_path)
    profile = basic_profile(df)
    schema_text = profile_to_schema_text(profile)

    print("\n=== BUILD1 LLM ASSISTANT ===\n")

    if not args.quiet_schema:
        print("=== DATASET SCHEMA ===")
        print(schema_text)

    print("\nType 'help' for commands. Type 'exit' to quit.\n")
    # FILL IN THE BLANK
    chain = build_chain(
        model=args.model,
        temperature=args.temperature,
        stream=args.stream,
        memory=args.________,
    )

    while True:
        user_query = input("> ").strip()

        if not user_query:
            continue

        cmd = user_query.lower()

        if cmd in {"exit", "quit"}:
            print("Goodbye!")
            break

        if cmd == "help":
            print("\n" + HELP_TEXT + "\n")
            continue

        if cmd == "schema":
            print("\n=== DATASET SCHEMA ===")
            print(schema_text + "\n")
            continue
        # Streaming vs non-streaming response handling. If streaming is enabled,
        # we print chunks as they come in.
        inputs = {"schema_text": schema_text, "user_query": user_query}
        config = (
            {"configurable": {"session_id": "cli-session"}} if args.memory else None
        )

        if args.stream:
            print()
            if config:
                for chunk in chain.stream(inputs, config=config):
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                for chunk in chain.stream(inputs):
                    print(chunk, end="", flush=True)
                print("\n")
        else:
            if config:
                response = chain.invoke(inputs, config=config)
            else:
                response = chain.invoke(inputs)
            print("\n" + response + "\n")


if __name__ == "__main__":
    main()
