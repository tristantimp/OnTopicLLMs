# app.py
# Run with:  streamlit run app/app.py
# Dependencies: pip install streamlit pandas

import os
import json
import time
import random
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st

from utils import (
    normalize_headers,
    parse_conversation_any,
    ensure_data_dir,
    safe_append_row,
)


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Distractor Builder", layout="wide")

INPUT_DATA_DIR = "data"  # where domain CSVs live (loaded by you)
OUTPUT_DATA_DIR = "data/distractors"  # where distractors will be saved

REQUIRED_CORE = ["domain", "scenario", "system_instruction", "conversation"]


# -------------------------
# Helpers
# -------------------------
def on_add_pair():
    bot = st.session_state.cur_bot.strip()
    dist = st.session_state.cur_dist.strip()
    tgt = st.session_state.cur_tgt.strip()
    if bot and dist:
        st.session_state.pairs.append(
            {"bot turn": bot, "distractor": dist, "target_instruction": tgt}
        )
        # mark to clear on next rerun *before* widgets instantiate
        st.session_state._clear_inputs = True


def validate_columns(df: pd.DataFrame) -> Optional[str]:
    missing = [c for c in REQUIRED_CORE if c not in df.columns]
    if missing:
        return f"Missing required columns: {', '.join(missing)}"
    return None


def select_random_index(df: pd.DataFrame) -> int:
    return int(df.sample(1, random_state=random.randint(0, 10_000)).index[0])


def render_conversation(conv_value):
    """
    Renders a conversation in Streamlit.
    Accepts: list[dict] with {'role', 'content'}.
    If parsing fails, prints raw text.
    """
    conv = parse_conversation_any(conv_value)
    if isinstance(conv, list) and all(
        isinstance(x, dict) and "role" in x and "content" in x for x in conv
    ):
        for turn in conv:
            role = str(turn.get("role", "user")).lower()
            role = "assistant" if role == "assistant" else "user"
            with st.chat_message(role):
                st.write(turn.get("content", ""))
    else:
        st.info("Conversation (raw):")
        st.code(str(conv_value), language="json")


def extract_last_assistant_turn(conv_value) -> str:
    """
    Convenience: pull the last assistant message from the parsed conversation.
    """
    conv = parse_conversation_any(conv_value)
    if isinstance(conv, list):
        for turn in reversed(conv):
            if str(turn.get("role", "")).lower() == "assistant":
                return str(turn.get("content", "")).strip()
    return ""


def save_distractor_row_multi(domain, src_row, pairs, target_sys_instr_payload):
    """Save a row with multiple distractor pairs to the domain-specific CSV."""
    ensure_data_dir(OUTPUT_DATA_DIR)
    out_path = os.path.join(OUTPUT_DATA_DIR, f"{domain}.csv")

    # Strip target_instruction before writing
    pairs_clean = [
        {"bot turn": p.get("bot turn", ""), "distractor": p.get("distractor", "")}
        for p in pairs
    ]

    record = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "domain": src_row.get("domain", ""),
        "scenario": src_row.get("scenario", ""),
        "system_instruction": src_row.get("system_instruction", ""),
        "target_system_instruction": target_sys_instr_payload.strip(),
        "conversation_json": json.dumps(
            parse_conversation_any(src_row.get("conversation", "")), ensure_ascii=False
        ),
        "distractors": json.dumps(pairs_clean, ensure_ascii=False),
    }
    safe_append_row(
        out_path,
        record,
        header_columns=[
            "timestamp",
            "domain",
            "scenario",
            "system_instruction",
            "target_system_instruction",
            "conversation_json",
            "distractors",
        ],
    )
    return out_path


# -------------------------
# Session state
# -------------------------
if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = None
if "last_saved" not in st.session_state:
    st.session_state.last_saved = None

# builder state
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "cur_bot" not in st.session_state:
    st.session_state.cur_bot = ""
if "cur_dist" not in st.session_state:
    st.session_state.cur_dist = ""
if "cur_tgt" not in st.session_state:
    st.session_state.cur_tgt = ""  # NEW: per-pair target instruction(s)

# NEW: control flags
if "_clear_inputs" not in st.session_state:
    st.session_state._clear_inputs = False


# -------------------------
# Sidebar: CSV loader
# -------------------------
st.sidebar.title("Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload CSV with columns: domain, scenario, system_instruction, conversation",
    type=["csv"],
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        df = normalize_headers(df)
        err = validate_columns(df)
        if err:
            st.sidebar.error(err)
        else:
            st.session_state.dataset_df = df
            st.sidebar.success(f"Loaded {len(df):,} rows.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# Preview
if st.session_state.dataset_df is not None:
    with st.sidebar.expander("Preview (first 6 rows)"):
        st.dataframe(st.session_state.dataset_df.head(6), use_container_width=True)


# -------------------------
# Main UI
# -------------------------
st.title("Distractor Builder (Streamlit)")

if st.session_state.dataset_df is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# --- Scenario picker row ---
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    if st.button("🎲 Random Scenario", use_container_width=True):
        st.session_state.current_idx = select_random_index(st.session_state.dataset_df)
        st.session_state.last_saved = None
        # Reset builder state for a fresh sample
        st.session_state.pairs = []
        st.session_state.cur_bot = ""
        st.session_state.cur_dist = ""
        st.session_state.cur_tgt = ""

with col2:
    seed = st.number_input(
        "Optional seed",
        min_value=0,
        value=0,
        step=1,
        help="If non-zero, influences the next random pick.",
    )
    if seed:
        random.seed(int(seed))

with col3:
    # NEW: choose exact row index to load
    max_idx = len(st.session_state.dataset_df) - 1
    chosen_idx = st.number_input(
        "Row index",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1,
        help=f"Choose an exact row (0..{max_idx}) to build distractors for.",
    )
with col4:
    st.write("")
    if st.button("📌 Load by Index", use_container_width=True):
        st.session_state.current_idx = int(chosen_idx)
        st.session_state.last_saved = None
        # Reset builder state for a fresh sample
        st.session_state.pairs = []
        st.session_state.cur_bot = ""
        st.session_state.cur_dist = ""
        st.session_state.cur_tgt = ""

if st.session_state.current_idx is None:
    st.warning("Pick a random scenario or load by index to begin.")
    st.stop()

row = st.session_state.dataset_df.loc[st.session_state.current_idx]
domain = str(row.get("domain", "")).strip() or "unknown"

# Scenario + instructions
st.subheader(f"Domain: `{domain}`  •  Row: {st.session_state.current_idx}")
with st.expander("Scenario", expanded=True):
    st.write(row.get("scenario", ""))

with st.expander("System Instruction (from dataset)", expanded=False):
    st.code(str(row.get("system_instruction", "")), language="markdown")

st.markdown("### Conversation")
render_conversation(row.get("conversation", ""))

# --------------- Pair Builder ---------------
st.markdown("---")
st.markdown("### Build distractor pairs")

prefill_col1, prefill_col2 = st.columns([1, 2])
with prefill_col1:
    if st.button("↪️ Prefill Bot Turn from last assistant"):
        st.session_state.cur_bot = extract_last_assistant_turn(
            row.get("conversation", "")
        )
with prefill_col2:
    st.caption(
        "Quickly grabs the last assistant message from this conversation as your starting 'Bot Turn'."
    )

# Ensure we clear AFTER a previous click but BEFORE widgets are drawn this run
if st.session_state._clear_inputs:
    st.session_state.cur_bot = ""
    st.session_state.cur_dist = ""
    st.session_state.cur_tgt = ""
    st.session_state._clear_inputs = False

st.text_area(
    "Bot Turn",
    key="cur_bot",
    height=120,
    placeholder="Paste the assistant's sentence here.",
)

st.text_area(
    "Distractor",
    key="cur_dist",
    height=160,
    placeholder="Write a distractor that plausibly follows but deviates from the target instruction.",
)

# NEW: per-pair target instruction(s)
st.text_area(
    "Target instruction(s) this distractor attempts to break (free text or JSON list)",
    key="cur_tgt",
    height=120,
    placeholder='Example: "Do not reveal private user data" OR ["do not use tools","never browse"].',
)

add_col1, add_col2, add_col3 = st.columns([1, 1, 3])
with add_col1:
    add_disabled = not (
        st.session_state.cur_bot.strip() and st.session_state.cur_dist.strip()
    )
    st.button(
        "➕ Add pair",
        disabled=add_disabled,
        use_container_width=True,
        on_click=on_add_pair,
    )
with add_col2:
    if st.button(
        "🧹 Clear all pairs",
        use_container_width=True,
        disabled=len(st.session_state.pairs) == 0,
    ):
        st.session_state.pairs = []

# Show current pairs with remove buttons
if len(st.session_state.pairs) > 0:
    st.markdown("#### Current pairs")
    for i, pair in enumerate(st.session_state.pairs):
        with st.expander(f"Pair {i+1}", expanded=False):
            st.markdown(f"**Bot Turn**\n\n{pair.get('bot turn','')}")
            st.markdown(f"**Distractor**\n\n{pair.get('distractor','')}")
            ti = pair.get("target_instruction", "")
            if ti:
                st.markdown(f"**Target instruction(s) for this distractor**\n\n{ti}")
            else:
                st.caption("_No per-pair target instruction provided._")
            if st.button(f"❌ Remove pair {i+1}", key=f"remove_{i}"):
                st.session_state.pairs.pop(i)
                st.experimental_rerun()
else:
    st.info("No pairs yet. Add your first one above.")

st.markdown("---")

# Global target instruction (optional, kept for compatibility)
st.markdown("### Global Target System Instruction (optional)")
st.caption(
    "If you fill per-pair targets above, the CSV will save a JSON list aligned with pairs in the "
    "`target_system_instruction` column. If you leave per-pair targets empty, this global text will be saved instead."
)
target_sys_instr_global = st.text_area(
    "Global Target System Instruction (fallback)",
    value=str(row.get("system_instruction", "")),
    height=100,
    placeholder="Paste a single system prompt to test against (legacy mode).",
)

# --------------- Save ---------------
save_col1, save_col2 = st.columns([1, 2])
with save_col1:
    can_save = len(st.session_state.pairs) > 0
    if st.button(
        "💾 Save Distractors",
        type="primary",
        use_container_width=True,
        disabled=not can_save,
    ):
        try:
            # Build payload for `target_system_instruction` column:
            # - If any per-pair targets given, store a JSON list aligned to pairs.
            # - Otherwise, store the global text (legacy behavior).
            per_pair_targets: List[str] = [
                p.get("target_instruction", "").strip() for p in st.session_state.pairs
            ]
            has_any_per_pair = any(bool(x) for x in per_pair_targets)
            if has_any_per_pair:
                target_payload = json.dumps(per_pair_targets, ensure_ascii=False)
            else:
                target_payload = target_sys_instr_global.strip()

            out_path = save_distractor_row_multi(
                domain=domain,
                src_row=row.to_dict(),
                pairs=st.session_state.pairs,
                target_sys_instr_payload=target_payload,
            )
            st.session_state.last_saved = time.time()
            st.success(f"Saved to `{out_path}`")
        except Exception as e:
            st.error(f"Failed to save: {e}")

with save_col2:
    if st.session_state.last_saved:
        st.caption("Last save just now ✅")

st.markdown("---")
st.caption(
    "Distractors are saved per domain in `data/distractors/` (created automatically)."
)
