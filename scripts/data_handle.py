import numpy as np

# make a dataset where you might take previous_angle, as input
# as argument to the function pass which states you wantto have returned as 
# inputs and as targets as well.
def get_states_from_data(data, state_names, previous_states=0):
    def extract_state_at_shift(shift):
        state = {}

        for name in state_names:
            name_lower = name.lower()

            if name_lower == "endpoint":
                dx = data["xe[m]"] - data["x0[m]"]
                dy = data["ye[m]"] - data["y0[m]"]
                values = np.column_stack([dx.values, dy.values])

            else:
                matching_cols = [
                    col for col in data.columns
                    if col.split("[")[0].strip().lower() == name_lower
                ]

                values = data[matching_cols].values

            if shift > 0:
                values = values[:-shift]
            else:
                values = values

            state[name] = values

        return state

    states = {}

    # remove first rows so current and previous states have same length
    start = previous_states

    current_state = extract_state_at_shift(0)
    states["state"] = {
        name: values[start:]
        for name, values in current_state.items()
    }

    for p in range(1, previous_states + 1):
        previous_state = extract_state_at_shift(p)
        states[f"previous_state{p}"] = {
            name: values[start - p:]
            for name, values in previous_state.items()
        }

    return states
