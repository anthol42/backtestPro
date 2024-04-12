import json
from mako.lookup import TemplateLookup
import os
import sys
import pdoc
import glob
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.getcwd())

def render_python_code(code):
    # Highlight Python code
    highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())

    # Wrap the highlighted code in HTML tags
    html_code = highlighted_code

    return html_code


def render_page(path: str, html: str = None):
    if html is None:
        # Load the html file
        with open(f"doc/{path}/main.html", "r") as f:
            html = f.read()

    # Load config
    with open(f"doc/{path}/config.json", "r") as f:
        config = json.load(f)

    # Define variables for template substitution
    params = {
        "html_lang": "en",
        "absolute_path": "/finBacktest/build",
        "page_title": config["title"],
        "page_desc": config["description"],
        "page_id": config["id"],
        "page_content": html

    }

    # Load the Mako template from file
    template_lookup = TemplateLookup(directories=["doc/templates/basic"])
    template = template_lookup.get_template("frame.mako")

    # Render the template with the provided variables
    rendered_page = template.render(**params, pdoc=pdoc)

    # Print or use the rendered HTML page
    with open(f"build/{config['out_path']}", "w") as f:
        f.write(rendered_page)


def render_mako_page(path: str):
    template_lookup = TemplateLookup(directories=[f'doc/{path}'])

    # Get a template
    template = template_lookup.get_template('main.mako')

    # Open python code
    files = glob.glob(f"doc/{path}/*.py")
    params = {}
    for file in files:
        with open(file, "r") as f:
            code_raw = f.read()
            code = render_python_code(code_raw)
            params[os.path.basename(file).replace(".py", "")] = code
            params[f'{os.path.basename(file).replace(".py", "")}_raw'] = code_raw.replace("\n", "\\n")

    # Render the template with the provided variables
    html = template.render(**params)

    render_page(path, html)
