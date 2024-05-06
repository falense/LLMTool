def insert_newlines(string, n):
    words = string.split(' ')
    lines = []
    current_line = ''
    for word in words:
        if len(current_line) + len(word) + 1 > n:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word
    lines.append(current_line)
    return '\n'.join(lines)
