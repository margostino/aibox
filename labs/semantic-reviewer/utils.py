import json
import re


def _handle_error(error) -> str:
    return str(error).replace("Could not parse LLM output:", "")


def is_yaml_file(file_path: str, prefix: str) -> bool:
    return file_path.startswith(prefix) and (file_path.endswith('.yml') or file_path.endswith('.yaml'))


def normalize_review(review: str) -> dict:
    if len(review) == 0:
        return {}
    # TODO: validate
    try:
        review_parts = review.replace("```json", "```").split("```")
        if len(review_parts) == 3:
            sanitized_review = review_parts[1].strip()
        elif len(review_parts) == 1:
            sanitized_review = review_parts[0].strip()
        else:
            sanitized_review = "{\"error\": \"unable to parse review\"}"

        return json.loads(sanitized_review)
    except:
        error = f"unable to parse review: {review}"
        return {"error": error}


def sanitize_show_by_files(show_commit: str, prefix_file: str) -> str:
    sanitized_show_commit = ""
    should_skip = False
    pattern = re.compile(r'^diff --git a/(.*) b/(.*)')
    files_count = 0
    for line in show_commit.split('\n'):
        match = pattern.match(line)
        if match:
            file_path_a, file_path_b = match.groups()
            file_path = file_path_b if file_path_b else file_path_a
            if is_yaml_file(file_path, prefix_file):
                files_count += 1
                should_skip = False
                sanitized_show_commit += line + "\n"
            else:
                should_skip = True
        elif not should_skip:
            sanitized_show_commit += line + "\n"

    if files_count == 0:
        sanitized_show_commit = None

    return sanitized_show_commit
