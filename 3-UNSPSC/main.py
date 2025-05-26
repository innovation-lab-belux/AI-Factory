import os
import json
import pandas as pd
from typing import TypedDict, List, Optional, Dict, Any
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # Optional: for persistence

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    print("Warning: OPENAI_API_KEY not set. Please set it as an environment variable or in the script.")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"

# --- State Definition ---
class ItemToClassify(TypedDict):
    name: str
    description: str
    original_input: Optional[Dict[str, Any]] # To store the original row from Excel

class ClassificationState(TypedDict):
    item: ItemToClassify
    unspsc_df: Optional[pd.DataFrame]
    unspsc_data_file_path: str # Path to the UNSPSC CSV/Excel file
    unspsc_column_mapping: Dict[str, str] # Mapping for UNSPSC columns

    # Hierarchical selections
    selected_segment: Optional[Dict] # e.g., {"code": "43000000", "name": "Information Technology"}
    selected_family: Optional[Dict]
    selected_class: Optional[Dict]
    # Final results
    top_5_commodities: Optional[List[Dict]] # List of {"code": ..., "name": ...}
    # Error handling
    error_message: Optional[str]
    current_step: Optional[str] # For tracking progress
    current_thinking: List[str] # For logging the "thinking" process

# --- UNSPSC Data Handling ---
def load_and_preprocess_unspsc_data(csv_path: str, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Loads UNSPSC data from a CSV file and preprocesses it using the provided column mapping.
    """
    thinking_log = []
    thinking_log.append(f"Attempting to load UNSPSC data from CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype=str)
        thinking_log.append(f"Successfully loaded CSV. Shape: {df.shape}")
    except FileNotFoundError:
        thinking_log.append(f"Error: UNSPSC CSV file not found at {csv_path}")
        raise FileNotFoundError(f"UNSPSC CSV file not found at {csv_path}")
    except Exception as e:
        thinking_log.append(f"Error reading UNSPSC CSV: {str(e)}")
        raise e

    # These are the internal names the script will use.
    # The column_mapping maps these internal names to the actual names in the user's CSV.
    internal_expected_cols = {
        'Commodity': ('Commodity Code', 'Commodity Name'),
        'Class': ('Class Code', 'Class Name'),
        'Family': ('Family Code', 'Family Name'),
        'Segment': ('Segment Code', 'Segment Name'),
    }

    # Rename columns based on user's mapping to internal names
    renamed_cols_count = 0
    for level_internal_names in internal_expected_cols.values():
        for internal_name_part in level_internal_names: # e.g., "Segment Code", "Segment Name"
            # Construct the key for the user's mapping dict, e.g., "Segment_Code_Col"
            # The mapping dict should look like:
            # { 'Segment_Code': 'Actual_Col_Name_In_CSV_For_Segment_Code', ... }
            user_col_name_key = internal_name_part.replace(" ", "_") # e.g. "Segment_Code"
            
            actual_csv_col_name = column_mapping.get(user_col_name_key)

            if actual_csv_col_name and actual_csv_col_name in df.columns:
                if actual_csv_col_name != internal_name_part: # Avoid renaming if names already match
                    df.rename(columns={actual_csv_col_name: internal_name_part}, inplace=True)
                    thinking_log.append(f"Renamed CSV column '{actual_csv_col_name}' to internal '{internal_name_part}'")
                renamed_cols_count +=1
            elif not actual_csv_col_name:
                 thinking_log.append(f"Warning: Mapping key '{user_col_name_key}' not found in UNSPSC_COLUMN_MAPPING.")
            elif actual_csv_col_name not in df.columns:
                 thinking_log.append(f"Warning: Mapped column '{actual_csv_col_name}' (for internal '{internal_name_part}') not found in CSV headers: {list(df.columns)}.")


    # Validate that all necessary internal columns now exist
    missing_cols = []
    for level, (code_col, name_col) in internal_expected_cols.items():
        if code_col not in df.columns:
            missing_cols.append(code_col)
        if name_col not in df.columns:
            missing_cols.append(name_col)

    if missing_cols:
        err_msg = f"Missing critical UNSPSC columns after mapping: {', '.join(missing_cols)}. Please check your UNSPSC_COLUMN_MAPPING and CSV file."
        thinking_log.append(f"Error: {err_msg}")
        raise ValueError(err_msg)
    thinking_log.append("All expected internal columns are present after mapping.")

    # Clean whitespace
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    thinking_log.append("Cleaned whitespace from object columns.")

    # Drop duplicates at the commodity level, as it's the most granular
    # Ensure the commodity code column (internal name) is used here
    commodity_code_internal_col = internal_expected_cols['Commodity'][0]
    if commodity_code_internal_col in df.columns:
        initial_rows = len(df)
        df.drop_duplicates(subset=[commodity_code_internal_col], inplace=True, keep='first')
        thinking_log.append(f"Dropped {initial_rows - len(df)} duplicate rows based on '{commodity_code_internal_col}'.")
    else:
        thinking_log.append(f"Warning: Cannot drop duplicates as '{commodity_code_internal_col}' column not found.")
    
    # Print thinking log from this function
    for log_entry in thinking_log:
        print(f"  [DataLoad] {log_entry}")
    return df


def get_unique_level_codes(df: pd.DataFrame, level_code_col: str, level_name_col: str) -> List[Dict]:
    """Helper to get unique codes and names for a given UNSPSC level."""
    if level_code_col not in df.columns or level_name_col not in df.columns:
        print(f"  [GetUnique] Warning: Columns {level_code_col} or {level_name_col} not found for get_unique_level_codes.")
        return []
    unique_df = df[[level_code_col, level_name_col]].drop_duplicates().dropna()
    return [{'code': str(row[level_code_col]), 'name': str(row[level_name_col])} for _, row in unique_df.iterrows()]

# --- LLM Interaction ---
def call_llm_for_classification(item_name: str, item_description: str,
                                target_level_name: str, candidates: List[Dict],
                                current_thinking: List[str],
                                previous_selections: Optional[Dict] = None,
                                top_n: int = 1) -> List[Dict]:
    """
    Calls GPT-4o-mini to classify an item.
    """
    current_thinking.append(f"LLM Call for: {target_level_name}, Item: '{item_name[:50]}...'")
    if not candidates:
        current_thinking.append(f"  Error: No candidates provided for {target_level_name} classification.")
        return [{"error": f"No candidates provided for {target_level_name} classification."}]

    current_thinking.append(f"  Number of candidates for {target_level_name}: {len(candidates)}")
    candidate_strings = [f"{c['code']} - {c['name']}" for c in candidates[:150]] # Limit candidates to manage token count & prompt size
    if len(candidates) > 150:
        current_thinking.append(f"  Warning: Truncated candidate list for {target_level_name} from {len(candidates)} to 150 for LLM prompt.")

    history_prompt = ""
    if previous_selections:
        history_prompt = "Based on previous selections:\n"
        for level, sel in previous_selections.items():
            history_prompt += f"- {level}: {sel}\n"

    common_prompt_part = f"""
Item Name: {item_name}
Item Description: {item_description}
{history_prompt}
You are an expert UNSPSC classifier.
"""

    if top_n == 1:
        prompt = f"""{common_prompt_part}Select the single most relevant UNSPSC {target_level_name} for this item from the list below.
Return your answer as a single JSON object with "code" and "name" keys. Example: {{"code": "CODE_HERE", "name": "NAME_HERE"}}

{target_level_name}s:
- {"\n- ".join(candidate_strings)}

JSON Response:
"""
        response_format_type = "json_object"
    else: # Requesting top N
        prompt = f"""{common_prompt_part}Identify the top {top_n} most relevant UNSPSC {target_level_name}s for this item from the list below.
Return your answer as a JSON object containing a key "results" which is a list of objects, each with "code" and "name".
Example: {{"results": [{{"code": "CODE1", "name": "NAME1"}}, {{...}}]}}

{target_level_name}s:
- {"\n- ".join(candidate_strings)}

JSON Response (ensure it's a single JSON object with a "results" list):
"""
        response_format_type = "json_object"

    current_thinking.append(f"  Prompt for {target_level_name} (first 300 chars): {prompt[:300].replace(os.linesep, ' ')}...")
    
    llm_response_content = None # Initialize to ensure it's defined
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": response_format_type}
        )
        llm_response_content = completion.choices[0].message.content
        current_thinking.append(f"  LLM Raw Response for {target_level_name}: {llm_response_content}")

        if not llm_response_content:
            current_thinking.append(f"  Error: LLM returned empty content for {target_level_name}.")
            return [{"error": f"LLM returned empty content for {target_level_name}."}]

        result_json = json.loads(llm_response_content)

        if top_n == 1:
            if isinstance(result_json, dict) and 'code' in result_json and 'name' in result_json:
                selected_code = str(result_json['code']) # Ensure code is string for comparison
                # Validate if the selected code was in the candidates
                if any(str(c['code']) == selected_code for c in candidates):
                    current_thinking.append(f"  LLM Parsed & Validated for {target_level_name}: {result_json}")
                    return [result_json]
                else:
                    current_thinking.append(f"  Error: LLM selected code {selected_code} not in provided {target_level_name} candidates.")
                    return [{"error": f"LLM selected code {selected_code} not in provided {target_level_name} candidates."}]
            else:
                current_thinking.append(f"  Error: LLM response for {target_level_name} not in expected format: {llm_response_content}")
                return [{"error": f"LLM response for {target_level_name} not in expected format: {llm_response_content}"}]
        else: # top_n > 1
            if isinstance(result_json, dict) and "results" in result_json and isinstance(result_json["results"], list):
                validated_results = []
                for res_item in result_json["results"]:
                    if isinstance(res_item, dict) and 'code' in res_item and 'name' in res_item:
                         if any(str(c['code']) == str(res_item['code']) for c in candidates):
                            validated_results.append(res_item)
                if validated_results:
                    current_thinking.append(f"  LLM Parsed & Validated Top {len(validated_results)} for {target_level_name}: {validated_results}")
                    return validated_results[:top_n]
                else:
                    current_thinking.append(f"  Error: LLM returned no valid items in 'results' list for {target_level_name} or items not in candidates.")
                    return [{"error": f"LLM returned no valid items in 'results' list for {target_level_name} or items not in candidates."}]
            else:
                current_thinking.append(f"  Error: LLM response for top N {target_level_name} not in expected format (missing 'results' list): {llm_response_content}")
                return [{"error": f"LLM response for top N {target_level_name} not in expected format (missing 'results' list): {llm_response_content}"}]

    except json.JSONDecodeError:
        current_thinking.append(f"  Error: Failed to parse LLM JSON response for {target_level_name}: {llm_response_content}")
        return [{"error": f"Failed to parse LLM JSON response for {target_level_name}: {llm_response_content}"}]
    except Exception as e:
        current_thinking.append(f"  Error: Exception calling LLM for {target_level_name}: {str(e)}")
        return [{"error": f"Error calling LLM for {target_level_name}: {str(e)}"}]

# --- Graph Nodes ---
def start_node(state: ClassificationState) -> Dict:
    state['current_step'] = "start"
    state['current_thinking'] = ["Process Start: Initializing and loading UNSPSC data."]
    print(f"\n--- Item: {state['item']['name']} ---")
    print("Step: Start - Loading UNSPSC data.")
    try:
        df = load_and_preprocess_unspsc_data(state["unspsc_data_file_path"], state["unspsc_column_mapping"])
        state['current_thinking'].append(f"Successfully loaded and preprocessed UNSPSC data. Shape: {df.shape}")
        print("  [StartNode] UNSPSC data loaded successfully.")
        return {"unspsc_df": df, "error_message": None, "current_thinking": state['current_thinking']}
    except Exception as e:
        err_msg = f"Failed to load UNSPSC data: {str(e)}"
        state['current_thinking'].append(f"Error in StartNode: {err_msg}")
        print(f"  [StartNode] Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}

def classify_segment_node(state: ClassificationState) -> Dict:
    state['current_step'] = "classify_segment"
    state['current_thinking'].append("Step: Classify Segment")
    print("Step: Classify Segment")
    if state.get("error_message") or not isinstance(state.get("unspsc_df"), pd.DataFrame):
        state['current_thinking'].append("  Skipping segment classification due to prior error or missing DataFrame.")
        return {"current_thinking": state['current_thinking']}

    item = state['item']
    unspsc_df = state['unspsc_df']
    # Use internal column names
    segments = get_unique_level_codes(unspsc_df, 'Segment Code', 'Segment Name')
    state['current_thinking'].append(f"  Found {len(segments)} unique Segments.")
    if not segments:
        err_msg = "No Segments found in UNSPSC data."
        state['current_thinking'].append(f"  Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}

    result = call_llm_for_classification(item['name'], item['description'], "Segment", segments, state['current_thinking'])
    if "error" in result[0]:
        err_msg = f"Segment classification error: {result[0]['error']}"
        state['current_thinking'].append(f"  Error: {err_msg}")
        print(f"  [SegmentNode] Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}
    
    selected_segment = result[0]
    state['current_thinking'].append(f"  Selected Segment: {selected_segment['code']} - {selected_segment['name']}")
    print(f"  [SegmentNode] Selected Segment: {selected_segment['code']} - {selected_segment['name']}")
    return {"selected_segment": selected_segment, "error_message": None, "current_thinking": state['current_thinking']}

def classify_family_node(state: ClassificationState) -> Dict:
    state['current_step'] = "classify_family"
    state['current_thinking'].append("Step: Classify Family")
    print("Step: Classify Family")
    if state.get("error_message") or not state.get("selected_segment"):
        state['current_thinking'].append("  Skipping family classification due to prior error or no segment selected.")
        return {"current_thinking": state['current_thinking']}

    item = state['item']
    unspsc_df = state['unspsc_df']
    selected_segment = state['selected_segment']
    segment_code = selected_segment['code']
    state['current_thinking'].append(f"  Filtering families for Segment: {segment_code} - {selected_segment['name']}")

    family_df = unspsc_df[unspsc_df['Segment Code'] == segment_code]
    families = get_unique_level_codes(family_df, 'Family Code', 'Family Name')
    state['current_thinking'].append(f"  Found {len(families)} unique Families under selected Segment.")
    if not families:
        err_msg = f"No Families found for Segment {segment_code}."
        state['current_thinking'].append(f"  Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}

    previous_selections = {"Segment": f"{selected_segment['code']} - {selected_segment['name']}"}
    result = call_llm_for_classification(item['name'], item['description'], "Family", families, state['current_thinking'], previous_selections)

    if "error" in result[0]:
        err_msg = f"Family classification error: {result[0]['error']}"
        state['current_thinking'].append(f"  Error: {err_msg}")
        print(f"  [FamilyNode] Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}
    
    selected_family = result[0]
    state['current_thinking'].append(f"  Selected Family: {selected_family['code']} - {selected_family['name']}")
    print(f"  [FamilyNode] Selected Family: {selected_family['code']} - {selected_family['name']}")
    return {"selected_family": selected_family, "error_message": None, "current_thinking": state['current_thinking']}

def classify_class_node(state: ClassificationState) -> Dict:
    state['current_step'] = "classify_class"
    state['current_thinking'].append("Step: Classify Class")
    print("Step: Classify Class")
    if state.get("error_message") or not state.get("selected_family"):
        state['current_thinking'].append("  Skipping class classification due to prior error or no family selected.")
        return {"current_thinking": state['current_thinking']}

    item = state['item']
    unspsc_df = state['unspsc_df']
    selected_segment = state['selected_segment']
    selected_family = state['selected_family']
    family_code = selected_family['code']
    state['current_thinking'].append(f"  Filtering classes for Family: {family_code} - {selected_family['name']}")
    
    class_df = unspsc_df[unspsc_df['Family Code'] == family_code]
    classes = get_unique_level_codes(class_df, 'Class Code', 'Class Name')
    state['current_thinking'].append(f"  Found {len(classes)} unique Classes under selected Family.")
    if not classes:
        err_msg = f"No Classes found for Family {family_code}."
        state['current_thinking'].append(f"  Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}

    previous_selections = {
        "Segment": f"{selected_segment['code']} - {selected_segment['name']}",
        "Family": f"{selected_family['code']} - {selected_family['name']}"
    }
    result = call_llm_for_classification(item['name'], item['description'], "Class", classes, state['current_thinking'], previous_selections)

    if "error" in result[0]:
        err_msg = f"Class classification error: {result[0]['error']}"
        state['current_thinking'].append(f"  Error: {err_msg}")
        print(f"  [ClassNode] Error: {err_msg}")
        return {"error_message": err_msg, "current_thinking": state['current_thinking']}

    selected_class = result[0]
    state['current_thinking'].append(f"  Selected Class: {selected_class['code']} - {selected_class['name']}")
    print(f"  [ClassNode] Selected Class: {selected_class['code']} - {selected_class['name']}")
    return {"selected_class": selected_class, "error_message": None, "current_thinking": state['current_thinking']}

def classify_commodity_node(state: ClassificationState) -> Dict:
    state['current_step'] = "classify_commodity"
    state['current_thinking'].append("Step: Classify Commodity (Top 5)")
    print("Step: Classify Commodity (Top 5)")
    if state.get("error_message") or not state.get("selected_class"):
        state['current_thinking'].append("  Skipping commodity classification due to prior error or no class selected.")
        return {"current_thinking": state['current_thinking']}

    item = state['item']
    unspsc_df = state['unspsc_df']
    selected_segment = state['selected_segment']
    selected_family = state['selected_family']
    selected_class = state['selected_class']
    class_code = selected_class['code']
    state['current_thinking'].append(f"  Filtering commodities for Class: {class_code} - {selected_class['name']}")

    commodity_df = unspsc_df[unspsc_df['Class Code'] == class_code]
    commodities = get_unique_level_codes(commodity_df, 'Commodity Code', 'Commodity Name')
    # Filter out any higher-level codes that might be identical to commodity codes if data is messy
    # (e.g., if Class Code '43211600' also appears as a Commodity Code '43211600')
    commodities = [c for c in commodities if str(c['code'])[-2:] != "00" or str(c['code']) == str(class_code)] # Keep if it's the exact class code (some classes are also commodities)
    state['current_thinking'].append(f"  Found {len(commodities)} unique Commodities under selected Class (after filtering).")

    if not commodities:
        err_msg = f"No valid Commodities found for Class {class_code}."
        state['current_thinking'].append(f"  Error: {err_msg}")
        # Fallback: Consider the Class itself as a potential commodity if no sub-commodities exist
        # This is a common scenario in UNSPSC where a Class is the most granular level.
        class_as_commodity = {"code": selected_class['code'], "name": selected_class['name']}
        state['current_thinking'].append(f"  Fallback: Using selected Class as the top commodity: {class_as_commodity}")
        print(f"  [CommodityNode] No sub-commodities found. Using Class as commodity: {class_as_commodity['code']} - {class_as_commodity['name']}")
        return {"top_5_commodities": [class_as_commodity], "error_message": None, "current_thinking": state['current_thinking']}


    previous_selections = {
        "Segment": f"{selected_segment['code']} - {selected_segment['name']}",
        "Family": f"{selected_family['code']} - {selected_family['name']}",
        "Class": f"{selected_class['code']} - {selected_class['name']}"
    }
    results = call_llm_for_classification(item['name'], item['description'], "Commodity",
                                          commodities, state['current_thinking'], previous_selections, top_n=5)

    if not results or "error" in results[0]:
        error_msg_detail = results[0]['error'] if results and "error" in results[0] else "Unknown error during commodity selection"
        err_msg = f"Commodity classification error: {error_msg_detail}"
        state['current_thinking'].append(f"  Error: {err_msg}")
        print(f"  [CommodityNode] Error: {err_msg}")
        # Fallback to class if commodity selection fails
        class_as_commodity = {"code": selected_class['code'], "name": selected_class['name']}
        state['current_thinking'].append(f"  Fallback due to error: Using selected Class as the top commodity: {class_as_commodity}")
        print(f"  [CommodityNode] Error selecting commodities. Using Class as commodity: {class_as_commodity['code']} - {class_as_commodity['name']}")
        return {"top_5_commodities": [class_as_commodity], "error_message": None, "current_thinking": state['current_thinking']} # Return class as commodity, clear error for this step

    state['current_thinking'].append(f"  Selected Top Commodities: {results}")
    print(f"  [CommodityNode] Selected Top {len(results)} Commodities.")
    return {"top_5_commodities": results, "error_message": None, "current_thinking": state['current_thinking']}

def final_result_node(state: ClassificationState) -> Dict:
    state['current_step'] = "finished"
    state['current_thinking'].append("Step: Final Result")
    print("\n--- Classification Complete for Item ---")
    
    final_summary_parts = []
    item = state['item']
    final_summary_parts.append(f"Item: {item['name']} - {item['description']}")
    print(f"Item: {item['name']} - {item['description']}")

    if state.get("error_message"):
        err_msg = f"Error during classification: {state['error_message']}"
        final_summary_parts.append(err_msg)
        print(err_msg)
        state['current_thinking'].append(err_msg)
        # Even with an error, show what was selected if anything
    
    if state.get('selected_segment'):
        seg_info = f"  Selected Segment: {state['selected_segment']['code']} - {state['selected_segment']['name']}"
        final_summary_parts.append(seg_info); print(seg_info)
    if state.get('selected_family'):
        fam_info = f"  Selected Family: {state['selected_family']['code']} - {state['selected_family']['name']}"
        final_summary_parts.append(fam_info); print(fam_info)
    if state.get('selected_class'):
        cls_info = f"  Selected Class: {state['selected_class']['code']} - {state['selected_class']['name']}"
        final_summary_parts.append(cls_info); print(cls_info)

    if state.get("top_5_commodities"):
        com_header = "\n  Top Commodity Matches:"
        final_summary_parts.append(com_header); print(com_header)
        for comm in state["top_5_commodities"]:
            com_info = f"    - Code: {comm['code']}, Name: {comm['name']}"
            final_summary_parts.append(com_info); print(com_info)
        # For the graph state, just return the commodities or error
        output_for_state = state["top_5_commodities"]
    else:
        no_com_msg = "Classification finished, but no commodity codes were determined."
        if state.get("error_message"): # If there was an earlier error, it's already noted
             pass
        elif state.get("selected_class"): # If we got to class but no commodities
             no_com_msg += f" Stuck at Class: {state['selected_class']['name']}"
        final_summary_parts.append(no_com_msg); print(no_com_msg)
        output_for_state = no_com_msg # Or an error structure

    state['current_thinking'].extend(final_summary_parts)
    # print("\nFull Thinking Log for this item:")
    # for entry in state['current_thinking']:
    #     print(f"  THINK: {entry}")
    
    return {"final_output_summary": output_for_state, "current_thinking": state['current_thinking']}


# --- Graph Definition ---
def should_continue(state: ClassificationState) -> str:
    decision_reason = ""
    next_node = "end_process_error" # Default to error end

    if state.get("error_message"):
        decision_reason = f"Error encountered: {state['error_message']}. Ending process."
        next_node = "end_process_error"
    else:
        current_step = state.get("current_step")
        if current_step == "start":
            if state.get("unspsc_df") is not None:
                decision_reason = "Data loaded, proceeding to Segment classification."
                next_node = "classify_segment_node"
            else:
                decision_reason = "Data loading failed in start_node." # Error should be set
                next_node = "end_process_error"
        elif current_step == "classify_segment":
            if state.get("selected_segment"):
                decision_reason = "Segment selected, proceeding to Family classification."
                next_node = "classify_family_node"
            else: # Error should have been set by the node if selection failed
                decision_reason = "Segment not selected or error in segment node."
                next_node = "end_process_error"
        elif current_step == "classify_family":
            if state.get("selected_family"):
                decision_reason = "Family selected, proceeding to Class classification."
                next_node = "classify_class_node"
            else:
                decision_reason = "Family not selected or error in family node."
                next_node = "end_process_error"
        elif current_step == "classify_class":
            if state.get("selected_class"):
                decision_reason = "Class selected, proceeding to Commodity classification."
                next_node = "classify_commodity_node"
            else:
                decision_reason = "Class not selected or error in class node."
                next_node = "end_process_error"
        elif current_step == "classify_commodity":
            if state.get("top_5_commodities"):
                decision_reason = "Commodities selected, proceeding to final result."
                next_node = "final_result_node"
            else: # Error should have been set by the node
                decision_reason = "Commodities not selected or error in commodity node."
                next_node = "end_process_error"
        elif current_step == "finished": # Should not happen via conditional edge but as a safety
            decision_reason = "Process already finished."
            return END # Langgraph END
        else:
            decision_reason = f"Unknown current_step: {current_step} or logic error. Ending."
            next_node = "end_process_error"

    state['current_thinking'].append(f"Decision: {decision_reason} -> Next Node: {next_node if next_node != END else 'END'}")
    print(f"  [Decision] Current Step: {state.get('current_step')}, Error: {state.get('error_message') is not None}. -> {decision_reason} -> Next: {next_node}")
    return next_node


# Initialize graph
workflow = StateGraph(ClassificationState)

# Add nodes
workflow.add_node("start_node", start_node)
workflow.add_node("classify_segment_node", classify_segment_node)
workflow.add_node("classify_family_node", classify_family_node)
workflow.add_node("classify_class_node", classify_class_node)
workflow.add_node("classify_commodity_node", classify_commodity_node)
workflow.add_node("final_result_node", final_result_node) # Successful end
workflow.add_node("end_process_error", final_result_node) # Error end, routes to same summary node

# Define edges
workflow.set_entry_point("start_node")

# Conditional edges guide the process through classification levels or to an error summary
workflow.add_conditional_edges("start_node", should_continue)
workflow.add_conditional_edges("classify_segment_node", should_continue)
workflow.add_conditional_edges("classify_family_node", should_continue)
workflow.add_conditional_edges("classify_class_node", should_continue)
workflow.add_conditional_edges("classify_commodity_node", should_continue)

# Terminal edges
workflow.add_edge("final_result_node", END) # From successful completion
workflow.add_edge("end_process_error", END) # From error path

# Compile the graph
# memory = SqliteSaver.from_conn_string(":memory:") # Optional for state persistence/debugging
app = workflow.compile() # checkpointer=memory

# --- Main Execution Logic ---
def classify_item_from_excel_row(
    row_data: Dict[str, Any],
    unspsc_file_path: str,
    unspsc_col_map: Dict[str, str],
    item_name_col: str,
    item_desc_col: str
) -> List[Dict]:
    """
    Classifies a single item represented by a dictionary (a row from an Excel file).
    """
    item_name = str(row_data.get(item_name_col, ""))
    item_description = str(row_data.get(item_desc_col, ""))

    if not item_name and not item_description:
        print(f"Skipping row due to missing name and description: {row_data}")
        return [{"error": "Missing item name and description"}]

    initial_state = ClassificationState(
        item=ItemToClassify(name=item_name, description=item_description, original_input=row_data),
        unspsc_data_file_path=unspsc_file_path,
        unspsc_column_mapping=unspsc_col_map,
        selected_segment=None,
        selected_family=None,
        selected_class=None,
        top_5_commodities=None,
        error_message=None,
        current_step=None,
        unspsc_df=None, # Will be loaded by start_node
        current_thinking=[]
    )

    final_state = app.invoke(initial_state)

    if final_state.get("error_message") and not final_state.get("top_5_commodities"): # If error and no fallback commodities
        return [{"error": final_state["error_message"], "code": None, "name": None}]
    # If top_5_commodities exists (even with a prior error that was handled with a fallback), return them
    return final_state.get("top_5_commodities", [{"error": "No commodities classified and no specific error message.", "code": None, "name": None}])


if __name__ == "__main__":
    # --- User Configuration ---
    # 1. Set your OpenAI API Key as an environment variable `OPENAI_API_KEY` or directly in the script.
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY" or not OPENAI_API_KEY:
        print("Critical Error: OpenAI API key is not set. Exiting.")
        print("Please set the OPENAI_API_KEY environment variable or update it in the script.")
        exit(1)
    else:
        print(f"Using OpenAI API Key: ...{OPENAI_API_KEY[-4:]}")

    # 2. Provide the path to your UNSPSC codes CSV file
    #    **** IMPORTANT: UPDATE THIS PATH ****
    UNSPSC_DATA_FILE_PATH = "3-UNSPSC/UNSPSC.xlsx" # Example, use your actual file path

    # 3. **** IMPORTANT: UPDATE THIS MAPPING ****
    #    Map internal script names to YOUR CSV column names.
    #    The keys (e.g., 'Segment_Code') are what the script expects internally.
    #    The values (e.g., 'Your Segment Code Column Name') are the actual column headers in your CSV.
    UNSPSC_COLUMN_MAPPING = {
        'Segment_Code': 'Segment',        # Replace with actual column name for Segment codes
        'Segment_Name': 'Segment Title',  # Replace with actual column name for Segment names/titles
        'Family_Code': 'Family',          # Replace with actual column name for Family codes
        'Family_Name': 'Family Title',    # Replace with actual column name for Family names/titles
        'Class_Code': 'Class',            # Replace with actual column name for Class codes
        'Class_Name': 'Class Title',      # Replace with actual column name for Class names/titles
        'Commodity_Code': 'Commodity',    # Replace with actual column name for Commodity codes
        'Commodity_Name': 'Commodity Title'# Replace with actual column name for Commodity names/titles
    }
    print(f"Using UNSPSC data file: {UNSPSC_DATA_FILE_PATH}")
    print(f"Using UNSPSC column mapping: {UNSPSC_COLUMN_MAPPING}")


    # 4. Provide the path to the Excel file containing items to classify
    #    **** IMPORTANT: UPDATE THIS PATH ****
    ITEMS_TO_CLASSIFY_EXCEL_PATH = "path/to/your/items_to_classify.xlsx"

    # 5. Specify the column names in your items Excel for name and description
    #    **** IMPORTANT: UPDATE THESE IF YOUR COLUMN NAMES ARE DIFFERENT ****
    ITEM_NAME_COLUMN = "Item Name"
    ITEM_DESCRIPTION_COLUMN = "Description"

    # 6. (Optional) Specify the output Excel file path
    OUTPUT_EXCEL_PATH = "classified_items_output.xlsx"

    # --- Load items to classify ---
    try:
        print(f"Loading items to classify from: {ITEMS_TO_CLASSIFY_EXCEL_PATH}")
        # Check if ITEMS_TO_CLASSIFY_EXCEL_PATH is a placeholder
        if "path/to/your/items_to_classify.xlsx" in ITEMS_TO_CLASSIFY_EXCEL_PATH:
            print("\nWARNING: ITEMS_TO_CLASSIFY_EXCEL_PATH is a placeholder.")
            print("Simulating with a dummy item. Please update the path to your actual items file.")
            items_df = pd.DataFrame([{
                ITEM_NAME_COLUMN: "High Performance Laptop",
                ITEM_DESCRIPTION_COLUMN: "16GB RAM, 512GB SSD, Latest Gen Processor for professional use"
            },{
                ITEM_NAME_COLUMN: "Industrial Safety Gloves",
                ITEM_DESCRIPTION_COLUMN: "Heavy-duty, cut-resistant gloves for construction workers"
            }])
        else:
            items_df = pd.read_excel(ITEMS_TO_CLASSIFY_EXCEL_PATH)
        print(f"Loaded {len(items_df)} items to classify.")
    except FileNotFoundError:
        print(f"Error: Items to classify Excel file not found at '{ITEMS_TO_CLASSIFY_EXCEL_PATH}'. Please check the path.")
        exit(1)
    except Exception as e:
        print(f"Error reading items Excel file: {e}")
        exit(1)

    results_list = []
    for index, row in items_df.iterrows():
        print(f"\n>>>> Processing item {index + 1} / {len(items_df)}: {row.get(ITEM_NAME_COLUMN, 'N/A')}")
        row_dict = row.to_dict()
        top_commodities = classify_item_from_excel_row(
            row_dict,
            UNSPSC_DATA_FILE_PATH,
            UNSPSC_COLUMN_MAPPING,
            ITEM_NAME_COLUMN,
            ITEM_DESCRIPTION_COLUMN
        )
        
        output_row = row_dict.copy()
        if top_commodities and not ("error" in top_commodities[0] and top_commodities[0]["error"]):
             for i, comm in enumerate(top_commodities[:5]): # Ensure only top 5 are written
                output_row[f'UNSPSC_Match_{i+1}_Code'] = comm.get('code')
                output_row[f'UNSPSC_Match_{i+1}_Name'] = comm.get('name')
        else: # Handle error or no classification
            output_row['UNSPSC_Match_1_Code'] = top_commodities[0].get('code', 'ERROR') if top_commodities else 'N/A'
            output_row['UNSPSC_Match_1_Name'] = top_commodities[0].get('error', 'Classification Failed') if top_commodities else 'N/A'
            for i in range(1, 5): # Clear other match columns if error
                 output_row[f'UNSPSC_Match_{i+1}_Code'] = ''
                 output_row[f'UNSPSC_Match_{i+1}_Name'] = ''


        results_list.append(output_row)
        print("-" * 50)

    # Save results to a new Excel file
    output_df = pd.DataFrame(results_list)
    try:
        output_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print(f"\nClassification processing complete. Results saved to '{OUTPUT_EXCEL_PATH}'")
    except Exception as e:
        print(f"\nError saving results to Excel: {e}")
        print("Results (first 10 rows):")
        print(output_df.head(10).to_string())
