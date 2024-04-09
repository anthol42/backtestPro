import json
from mako.lookup import TemplateLookup
import os
import sys
import pdoc
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.getcwd())

def render_page(path: str):
    if os.path.exists(f"doc/{path}/main.html"):
        # Load the html file
        with open(f"doc/{path}/main.html", "r") as f:
            html = f.read()
    else:
        with open(f"doc/{path}/main.mako", "r") as f:
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