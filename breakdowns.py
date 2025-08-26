import streamlit as st

# Initialize session state
if "breakdowns" not in st.session_state:
    st.session_state["breakdowns"] = {
        "Device": {
            "mobile": {"weight": 0.5, "factor": 1.1},
            "desktop": {"weight": 0.4, "factor": 1.5},
            "tablet": {"weight": 0.1, "factor": 1.3},
        }
    }

breakdowns = st.session_state["breakdowns"]

# Add new breakdown
with st.sidebar:
    st.markdown("### Add Breakdown")
    new_breakdown = st.text_input("New breakdown name")
    if st.button("Add breakdown") and new_breakdown:
        if new_breakdown not in breakdowns:
            breakdowns[new_breakdown] = {}

st.markdown("## Breakdown Configuration")

# UI for each breakdown
to_delete = []
for breakdown, items in breakdowns.items():
    with st.expander(f"📊 {breakdown}", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(f"❌ Delete '{breakdown}'", key=f"delete_{breakdown}"):
                to_delete.append(breakdown)

        st.markdown("### Items")

        # Add new item
        new_item = st.text_input(f"New item for {breakdown}", key=f"new_{breakdown}")
        if st.button(f"Add item to {breakdown}", key=f"add_{breakdown}"):
            if new_item and new_item not in items:
                items[new_item] = {"weight": 0.0, "factor": 1.0}

        item_weights = {}
        for item, values in items.items():
            c1, c2, c3 = st.columns([3, 3, 1])
            with c1:
                weight = st.slider(
                    f"{item} weight", min_value=0.0, max_value=1.0, step=0.01,
                    value=values["weight"], key=f"w_{breakdown}_{item}"
                )
            with c2:
                factor = st.slider(
                    f"{item} factor", min_value=0.0, max_value=3.0, step=0.1,
                    value=values["factor"], key=f"f_{breakdown}_{item}"
                )
            with c3:
                if st.button("🗑️", key=f"del_{breakdown}_{item}"):
                    del items[item]
                    break
            item_weights[item] = weight
            values["factor"] = factor

        # Normalize weights
        total = sum(item_weights.values())
        if total > 0:
            for item in item_weights:
                items[item]["weight"] = round(item_weights[item] / total, 3)

# Delete marked breakdowns
for b in to_delete:
    del breakdowns[b]

# Display final dict
st.markdown("## Final `BREAKDOWNS` Dict")
import json
st.code("BREAKDOWNS = " + json.dumps(breakdowns, indent=4))

# Optional: copy to clipboard (Streamlit doesn't have native clipboard API)
st.text_area("Copyable Dict", value=json.dumps(breakdowns, indent=4), height=300)
