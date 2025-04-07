import random


def generate_array(alphabet, length, repeat_prob=0.5):
    if length <= 0:
        return []

    # Pick the first element uniformly at random.
    result = [random.choice(alphabet)]

    for _ in range(1, length):
        current = result[-1]
        current_index = alphabet.index(current)

        # Initialize probability list with 0s.
        probabilities = [0] * len(alphabet)
        # Fixed probability for repeating the same character.
        probabilities[current_index] = repeat_prob

        # Compute raw weights for all elements other than the current one.
        raw_weights = []
        for i in range(len(alphabet)):
            if i == current_index:
                raw_weights.append(0)  # no weight needed for the current element
            else:
                # Weight inversely proportional to the distance from the current index.
                raw_weights.append(1 / abs(i - current_index))

        # Sum of raw weights for non-current elements.
        weight_sum = sum(raw_weights)

        # Distribute the remaining probability (1 - repeat_prob) according to the normalized weights.
        for i in range(len(alphabet)):
            if i != current_index:
                probabilities[i] = (raw_weights[i] / weight_sum) * (1 - repeat_prob)

        # Choose the next element based on the computed probabilities.
        next_element = random.choices(alphabet, weights=probabilities, k=1)[0]
        result.append(next_element)

    return result


def generate_array_grouped(alphabet, length, group_size, group_repeat_prob=0.66):
    if length <= 0:
        return []

    # Partition the alphabet into groups.
    groups = [alphabet[i : i + group_size] for i in range(0, len(alphabet), group_size)]

    # Helper function to determine the group index for an element.
    def get_group_index(element):
        idx = alphabet.index(element)
        return idx // group_size

    # Start with a randomly chosen element.
    result = [random.choice(alphabet)]

    for _ in range(1, length):
        current = result[-1]
        current_group_index = get_group_index(current)
        num_groups = len(groups)

        # Build group-level probabilities.
        group_probs = []
        for i in range(num_groups):
            if i == current_group_index:
                group_probs.append(group_repeat_prob)
            else:
                # Distribute the remaining probability evenly among the other groups.
                group_probs.append(
                    (1 - group_repeat_prob) / (num_groups - 1) if num_groups > 1 else 0
                )

        # Choose a group based on these probabilities.
        chosen_group_index = random.choices(
            range(num_groups), weights=group_probs, k=1
        )[0]
        # Then choose an element uniformly from the chosen group.
        next_element = random.choice(groups[chosen_group_index])
        result.append(next_element)

    return result


# Example usage:
alphabet = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
generated_array = generate_array(alphabet, 100, repeat_prob=0.5)
print("Generated array:", generated_array)

alphabet = [1, 2, 3, 4, 5]
generated_array = generate_array_grouped(
    alphabet, 100, group_size=2, group_repeat_prob=0.66
)
print("Generated array (grouped):", generated_array)
