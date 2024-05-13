# Diversified Late Acceptance Search
# Majid Namazi, Conrad Sanderson, M.A. Hakim Newton, M.M.A. Polash, Abdul Sattar
from main import ClusteringProblem


def dla_hc(problem: ClusteringProblem, history_depth, max_iter=5000):
    current_state = problem.generate_random_state()
    current_state_score = problem.value(current_state)
    history_pos = 0
    history = [current_state_score] * history_depth
    history_min = current_state_score
    history_min_count = history_depth
    best_state = current_state
    best_state_score = current_state_score

    iteration = 0
    while iteration < max_iter:
        old_score = current_state_score
        action = problem.random_action(current_state)

        candidate_state = problem.result(current_state, action)
        candidate_state_score = problem.value(candidate_state)

        if candidate_state_score == current_state_score or candidate_state_score > history_min:
            # accept
            current_state = candidate_state
            current_state_score = candidate_state_score
            if current_state_score > best_state_score:
                best_state = current_state
                best_state_score = current_state_score
                print("new best", best_state_score)

        if current_state_score < history[history_pos]:
            history[history_pos] = current_state_score
        elif current_state_score > history[history_pos] and current_state_score > old_score:
            if history[history_pos] == history_min:
                history_min_count -= 1
            history[history_pos] = current_state_score
            if history_min_count == 0:
                history_min = min(history)
                for m in history:
                    if m == history_min:
                        history_min_count += 1

        history_pos = history_pos + 1 if history_pos < history_depth - 1 else 0
        iteration += 1
    return best_state, best_state_score

