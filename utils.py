import ast

def try_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        print(f"Maformed value: {value}")
        return value