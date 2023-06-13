def find_repeating_substring(strings: list) -> str:
    """
    Takes a list of strings and returns the longest repeating substring pattern among them.
    """
    # Find the shortest string in the list
    shortest_string = min(strings, key=len)
    # Find the maximum length of the repeating substring
    max_length = len(shortest_string)
    # Iterate over all possible substring lengths
    for length in range(max_length, 0, -1):
        # Iterate over all possible starting positions
        for start in range(len(shortest_string) - length + 1):
            # Get the substring to search for
            substring = shortest_string[start : start + length]
            # Check if the substring repeats in all strings
            if all(substring in s for s in strings):
                return substring
    # If no repeating substring is found, return an empty string
    return ""


def remove_repeating_substring(strings: list) -> list:
    """
    Takes a list of strings and removes the longest repeating substring pattern among them.
    """
    # Find the repeating substring pattern
    repeating_substring = find_repeating_substring(strings)
    print(f"Found repeating substring: `{repeating_substring}`")
    # Remove the repeating substring pattern from all strings
    return [s.replace(repeating_substring, "") for s in strings]
