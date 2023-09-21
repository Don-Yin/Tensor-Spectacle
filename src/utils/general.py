import shutil


def banner(content: str, symbol="-"):
    """
    Print a banner to the terminal to screen width
    e.g., ---- hello world ----
    """
    terminal_width, _ = shutil.get_terminal_size()
    content = " " + content.strip() + " "
    content = content.center(terminal_width, symbol)
    print(content)
