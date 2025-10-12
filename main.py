import pandas as pd
import json
import os

splits = {
    'train': 'canttalkaboutthis_topic_control_mixtral.jsonl',
    'test': 'canttalkaboutthis_topic_control_human_test_set.jsonl'
}
data = pd.read_json(
    "hf://datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset/" + splits["train"],
    lines=True
)

unique_domains = data['domain'].unique()
print("Available domains:")
for i, dom in enumerate(unique_domains):
    print(f"{i + 1}. {dom}")

selected_index = int(input("Select a domain by number: ")) - 1
selected_domain = unique_domains[selected_index]
print(f"\nYou selected domain: {selected_domain}\n")

output_filename = f"annotations_{selected_domain}_distractor_pairs.json"

# Load existing annotated pairs if file exists, or start fresh
if os.path.exists(output_filename):
    with open(output_filename, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    # Build a set of (entry_idx, pair_idx) for quick skipping
    annotated_pairs = {(ann['entry_idx'], ann['pair_idx']) for ann in annotations}
    print(f"Loaded {len(annotated_pairs)} existing annotations, resuming...")
else:
    annotations = []
    annotated_pairs = set()

filtered_data = data[data['domain'] == selected_domain]

for entry_idx, row in filtered_data.iterrows():
    sys_instr = row.get('system_instruction', '')
    print(f"\nSystem instructions:\n{sys_instr}\n")

    distractors = row['distractors']
    conversation_blocks = row['conversation_with_distractors']

    print(f"--- Entry {entry_idx+1} distractor pairs ---")

    for pair_idx, d in enumerate(distractors):
        # Skip if already annotated this pair
        if (entry_idx, pair_idx) in annotated_pairs:
            print(f"Skipping annotated pair {pair_idx+1} of Entry {entry_idx+1}")
            continue

        distractor_question = d['distractor'].strip()
        expected_bot_turn = d['bot turn'].strip()

        found_response = None
        for conv_block in conversation_blocks:
            for i in range(len(conv_block) - 1):
                if conv_block[i]['role'] == 'user' and conv_block[i]['content'].strip() == distractor_question:
                    if conv_block[i+1]['role'] == 'assistant':
                        found_response = conv_block[i+1]['content'].strip()
                        break
            if found_response is not None:
                break

        if found_response is None:
            found_response = "[Assistant response not found]"

        print(f"\nPair {pair_idx+1} of Entry {entry_idx+1}:")
        print(f"User distractor question: {distractor_question}")
        print(f"Assistant response: {found_response}")

        label = input("Label (on-topic/distractor): ").strip()
        distractor_type = "NA"
        difficulty = "NA"
        fooled_llm = "no"
        notes = ""
        if label.lower() == 'distractor':
            distractor_type = input("If distractor, type (realistic/adversarial): ").strip()
            difficulty = input("Difficulty (easy/medium/hard): ").strip()
            fooled_llm = input("Did the distractor fool the LLM? (yes/no): ").strip().lower()
            notes = input("Notes (optional): ").strip()

        annotations.append({
            "entry_idx": entry_idx,
            "pair_idx": pair_idx,
            "role": "user",
            "content": distractor_question,
            "label": label,
            "type": distractor_type,
            "difficulty": difficulty,
            "fooled_llm": fooled_llm,
            "notes": notes
        })

        # Save after every annotation to preserve progress
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        cont = input("Press Enter to continue to next pair, or 'q' to quit: ")
        if cont.lower() == 'q':
            break

    cont_outer = input("Press Enter to continue to next entry, or 'q' to quit: ")
    if cont_outer.lower() == 'q':
        break

print(f"\nSaved annotations to {output_filename}")
