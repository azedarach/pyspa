import numpy as np

from .time_utils import get_days_in_year

def get_fixed_state_lengths(viterbi, dates=None):
    T = np.size(viterbi)

    result = {"state": [], "start": [], "stop": [], "length": []}
    if dates is not None:
        if np.size(viterbi) != np.size(dates):
            raise RuntimeError(
                "Viterbi path length does not match number of dates")
        result["year"] = []
        result["month"] = []

    length = 1
    for idx in range(0, T - 1):
        current_state = viterbi[idx]
        next_state = viterbi[idx + 1]
        if current_state == next_state:
            length += 1
            if idx == T - 2:
                result["state"].append(next_state)
                result["start"].append(idx + 2 - length)
                result["stop"].append(idx + 1)
                result["length"].append(length)
                if dates is not None:
                    result["year"].append(dates[idx + 1].year)
                    result["month"].append(dates[idx + 1].month)
        else:
            result["state"].append(current_state)
            result["start"].append(idx + 1 - length)
            result["stop"].append(idx)
            result["length"].append(length)
            if dates is not None:
                result["year"].append(dates[idx].year)
                result["month"].append(dates[idx].month)
            length = 1

    for field in result:
        result[field] = np.asarray(result[field])

    return result

def assign_persistent_state_ids(states, viterbi,
                                persistence_threshold=1):
    """Identify persistent states for each step of Viterbi path."""
    n_states = np.size(states)
    T = np.size(viterbi)

    state_ids = np.zeros(T)
    idx = 0
    while idx < T:
        current_state = viterbi[idx]
        look_ahead_pos = idx + 1
        switched = False
        while not switched and look_ahead_pos < T:
            if viterbi[look_ahead_pos] == current_state:
                look_ahead_pos += 1
            else:
                switched = True
        length = look_ahead_pos - idx
        if length >= persistence_threshold:
            state_ids[idx:look_ahead_pos] = current_state
            idx += length
        else:
            state_ids[idx] = n_states
            idx += 1

    return state_ids

def get_annual_occupancies(states, viterbi, dates,
                           persistence_threshold=1):
    """Return the fraction of days spent in each state in each year."""
    n_states = np.size(states)

    if np.size(viterbi) != np.size(dates):
        raise ValueError(
        "length of Viterbi path does not match number of dates")

    state_ids = assign_persistent_state_ids(
        states, viterbi, persistence_threshold=persistence_threshold)

    years = np.array([d.year for d in dates])
    unique_years = np.unique(years)
    yearly_counts = np.zeros((np.size(unique_years), n_states + 1))
    for idx, year in enumerate(unique_years):
        for state in range(n_states + 1):
            mask = (years == year) & (state_ids == state)
            yearly_counts[idx, state] = np.sum(mask)

    yearly_sums = np.sum(yearly_counts, axis=1)
    yearly_fracs = np.divide(yearly_counts, yearly_sums[:, np.newaxis])

    normalized = np.sum(yearly_fracs, axis=1)

    if not np.allclose(normalized, np.ones(np.size(normalized))):
        raise AssertionError("occupancy fractions do not sum to 1")

    return {"years": unique_years, "yearly_counts": yearly_counts,
            "yearly_sums": yearly_sums, "yearly_fracs": yearly_fracs}
