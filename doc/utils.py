import json
import re

import cssutils
from mako.lookup import TemplateLookup
import os
import sys
import pdoc
import glob
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter
import nbformat

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


def render_notebook_raw(notebook_path: str) -> str:
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Configure the HTML exporter
    html_exporter = HTMLExporter()

    # Convert the notebook to HTML
    (body, resources) = html_exporter.from_notebook_node(nb)

    return body

def extract_css(notebook_path: str):
    html = render_notebook_raw(notebook_path)
    soup = BeautifulSoup(html, 'html.parser')
    style_elements = soup.find_all('style')
    css = ''
    for style in style_elements:
        css += style.get_text() + '\n'
    for style in style_elements:
        style.extract()

    # We remove the body spect because it is already defined in the main template
    css = re.sub(r'body {.*?}', '', css, flags=re.DOTALL)

    with open('doc/assets/notebook.css', 'w', encoding='utf-8') as f:
        f.write(css)

def build_notebook_object(notebook_path: str):
    html = render_notebook_raw(notebook_path)

    # Remove style elements from the HTML content
    soup = BeautifulSoup(html, 'html.parser')
    style_elements = soup.find_all('style')
    for style in style_elements:
        style.extract()

    # Keep only the content of the body tag
    body = soup.body

    # Change the attributes of the root main tag
    main = body.find('main')
    main["class"] = "jp-Notebook"
    main["data-jp-theme-light"] = "true"
    main["data-jp-theme-name"] = "JupyterLab Light"

    return body.contents[1]


def render_tutorials():
    with open("doc/templates/notebooks/rendering_scripts.html", "r") as f:
        rendering_scripts = f.read()

    # Get notebooks
    notebooks = [os.path.basename(f.replace(".ipynb", "")) for f in glob.glob("doc/tutorials/notebooks/*.ipynb")]

    # Build the style file
    extract_css(f"doc/tutorials/notebooks/{notebooks[0]}.ipynb")

    # Load the template
    template_lookup = TemplateLookup(directories=["doc/templates/notebooks"])
    template = template_lookup.get_template("frame.mako")

    for notebook in notebooks:
        # Build the notebook object
        notebook_object = build_notebook_object(f"doc/tutorials/notebooks/{notebook}.ipynb")

        params = {
            "html_lang": "en",
            "absolute_path": "/finBacktest/build",
            "page_title": notebook,
            "page_desc": "A tutorial on how to use the BacktestPro Framework.",
            "notebook_content": notebook_object,
            "rendering_scripts": rendering_scripts,
            "available_files": notebooks,
            "page_id": notebook
        }

        # Render the template with the provided variables
        html = template.render(**params, pdoc=pdoc)
        if not os.path.exists("build/tutorials"):
            os.makedirs("build/tutorials")

        # Write the rendered page to file
        with open(f"build/tutorials/{notebook}.html", "w") as f:
            f.write(html)

    # Render the tutorial home page
    with open("doc/tutorials/home.html", "r") as f:
        content = f.read()
    params = {
        "html_lang": "en",
        "absolute_path": "/finBacktest/build",
        "page_title": "Tutorial Home",
        "page_desc": "A tutorial on how to use the BacktestPro Framework.",
        "notebook_content": content,
        "rendering_scripts": rendering_scripts,
        "available_files": notebooks,
        "page_id": "home"
    }
    html = template.render(**params, pdoc=pdoc)
    with open(f"build/tutorials/index.html", "w") as f:
        f.write(html)