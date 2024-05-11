import shutil
import cairosvg
from PIL import Image
import io
import pdoc
import re
from pdoc.cli import recursive_write_files, _generate_lunr_search
from typing import Optional
from pathlib import PurePath
import inspect
from enum import EnumType
import os
import sys
from utils import render_page, render_mako_page, render_tutorials
from src import backtest
import utils
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.getcwd())

FLATTEN_DOC = True
BASE_OUR_PATH = '../anthol42.github.io/backtestpro'
ABSOLUTE_PATH = '/backtestpro'

utils.ABSOLUTE_PATH = ABSOLUTE_PATH
"""
If True, the documentation will be flattened, meaning that the submodules will be part of the main module.
We will have the following structure:
backtest
    data
    engine
    indicators
    serve
        renderers
Instead of having a structure based on the file tree.
"""
VERBOSE = 1    # 0: No output, 1: Ignored objects (Because no docstring)
pdoc.cli.args.output_dir = f"{BASE_OUR_PATH}/docs"
pdoc.tpl_lookup.directories.insert(0, './doc/templates/pdoc')
template_config = {'lunr_search': {'index_docstrings': True},
                   'list_class_variables_in_index': True,
                   'show_source_code': False,
                   'show_inherited_members': False,
                   'absolute_path': ABSOLUTE_PATH,
                   'pdoc': pdoc,
                   'backtest': backtest}
documented_modules = {
    "data",
    "engine",
    "indicators",
    "serve"
}

# Make docFilter
def docFilter(doc: pdoc.Doc):
    if doc is None:
        return False
    path = doc.module.name.split(".")
    if len(path) > 1 and path[1] in documented_modules:
        return True
    else:
        return False

mod = pdoc.Module(pdoc.import_module("src/backtest"), docfilter=docFilter)
def delete_no_doc(module: pdoc.Module):
    """
    Recursively Delete the objects that have no docstring or their docstring is "NO DOC"
    :param module: The module to clean
    """
    for key, value in list(module.doc.items()):
        if isinstance(value, pdoc.Module):
            delete_no_doc(value)
        elif isinstance(value, pdoc.Class) and not isinstance(value.obj, EnumType):   # Enum are handled differently
            if value.docstring == "NO DOC":
                print(f"Deleting {key} from {module.name}") if VERBOSE > 0 else None
                del module.doc[key]
            for method_name, method in list(value.doc.items()):
                if method.docstring == "":
                    print(f"Deleting {method_name} from {key} in {module.name}") if VERBOSE > 0 else None
                    del value.doc[method_name]
            if len(value.doc) == 0:
                print(f"Deleting {key} from {module.name}") if VERBOSE > 0 else None
                del module.doc[key]
        elif isinstance(value.obj, EnumType):
            if value.docstring == "NO DOC":
                print(f"Deleting {key} from {module.name}") if VERBOSE > 0 else None
                del module.doc[key]
        elif value.docstring == "":
            print(f"Deleting {key} from {module.name}") if VERBOSE > 0 else None
            del module.doc[key]
delete_no_doc(mod)

def rename_module_alias(module: pdoc.Module):
    members = {name: obj for name, obj in inspect.getmembers(module.obj)}
    modules = {name: m for name, m in module.doc.items() if isinstance(m, pdoc.Module)}
    for alias, m in members.items():
        if inspect.ismodule(m) and alias[0] != "_" and m.__file__ is not None:
            name = PurePath(m.__file__).name.split(".")[0]
            if alias != name and name != "__init__":
                pdoc_module = modules[name]
                pdoc_module.name = module.name + "." + alias
def flatten_module(module: pdoc.Module, ref_module: Optional[dict[str, pdoc.Module]] = None,
                   whitelist: Optional[dict[str, set[str]]] = None, refModuleName: Optional[str] = None):
    """
    Make the documentation looks more like the true api by linking objects imported from __init__ file inside index
    pages.  It also resolves alias modules (Not done by default in pdoc).
    This method is recursive.
    :param module: The module to flatten
    :param ref_module: The reference module where to insert the objects
    :param whitelist: The whitelist of objects to insert (Imported by the __init__ file)
    :param refModuleName: The reference module name
    :return: None
    """
    if whitelist is None:
        whitelist = {}
    if ref_module is None:
        ref_module = {module.name: module}
    if PurePath(module.obj.__file__).name == "__init__.py":
        members = [name for name, obj in inspect.getmembers(module.obj) if name[0] != "_" and not inspect.ismodule(obj)]
        whitelist[module.name] = set(members)
        sub_modules = [name for name, m in module.module.doc.items() if isinstance(m, pdoc.Module)]
        rename_module_alias(module)
        for name in sub_modules:
            flatten_module(module.doc[name], ref_module, whitelist, refModuleName=module.name)

    # Inserting the objects in the main module
    if ref_module.get(refModuleName) is not None:
        objects = {name: item for name, item in module.doc.items() if not isinstance(item, pdoc.Module) and name in whitelist[refModuleName]}
        ref_module[refModuleName].doc.update(objects)
    return module

if FLATTEN_DOC:
    # Data
    flatten_module(mod.doc["data"])
    # Indicators
    flatten_module(mod.doc["indicators"])
    # Engine
    flatten_module(mod.doc["engine"])
    # Serve
    flatten_module(mod.doc["serve"])
    flatten_module(mod.doc["serve"].doc["renderers"])

pdoc.link_inheritance()

def parse_params(doc: list[str]) -> dict[str, str]:
    params = {}
    for line in doc:
        name = line.split(":")[0].strip()
        value = ":".join(line.split(":")[1:]).strip()
        params[name] = value
    return params

def parse_raise(doc: list[str]) -> list[(str, str)]:
    raises = []
    for line in doc:
        name = line.split(":")[0].strip()
        value = ":".join(line.split(":")[1:]).strip()
        raises.append((name, value))
    return raises


def doc2markdown(doc: str):
    if doc is None or doc == "":
        return ""
    params = []
    return_parsed = ""
    raises = []
    for key, value in re.findall(r":(param|return|raise)(.*?)(?=:param|:return|:raise|$)", doc.replace("\n", "")):
        if key == "param":
            params.append(value)
        elif key == "return":
            return_parsed = value.strip()
        elif key == "raise":
            raises.append(value)

    params = parse_params(params)
    raises = parse_raise(raises)
    try:
        explanation = re.match(r"^(.*?)(?=:(param|return|raise))", doc, re.DOTALL).group(1).strip()
    except:
        explanation = doc
    out = explanation
    if len(params) > 0:
        out += "\n\nArgs:\n-----=\n"
        for key, value in params.items():
            out += f"**```{key}```**: {value}  \n"
    if return_parsed != "":
        out += "\n\nReturns:\n---------\n"
        out += return_parsed

    if len(raises) > 0:
        out += "\n\nRaises:\n-----=\n"
        for name, value in raises:
            out += f"**```{name}```**: {value}  \n"
    return out

# Iterate over every documented module and convert the rst parameters definition to the another style
def convert_rst(docs: dict[str, pdoc.Doc]):
    for key, doc in docs.items():
        if hasattr(doc, "doc"):
            doc.docstring = doc2markdown(doc.docstring)
            convert_rst(doc.doc)
        else:
            doc.docstring = doc2markdown(doc.docstring)

# Convert the rst parameters definition to a custom markdown style.
convert_rst(mod.doc)

modules = [mod]
for module in modules:
    recursive_write_files(module, ext='.html', **template_config)


lunr_config = pdoc._get_config(**template_config).get('lunr_search')
if lunr_config is not None:
    _generate_lunr_search(
        modules, lunr_config.get("index_docstrings", True), template_config)


# Render the home page
render_mako_page("home", out_path=BASE_OUR_PATH, abs_path=ABSOLUTE_PATH)

# Render the get started page
render_mako_page("get_started", out_path=BASE_OUR_PATH, abs_path=ABSOLUTE_PATH)

render_page("about", out_path=BASE_OUR_PATH)

# Finally, handle assets
shutil.copytree("doc/assets", f"{BASE_OUR_PATH}/assets", dirs_exist_ok=True)

# Render the tutorials
render_tutorials(out_path=BASE_OUR_PATH)


def svg_to_ico(svg_file, output_ico, sizes=((64, 64), )):
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(url=svg_file, output_width=max(s[0] for s in sizes), output_height=max(s[0] for s in sizes))

    # Convert PNG to ICO
    with Image.open(io.BytesIO(png_data)) as img:
        img.save(output_ico, sizes=sizes)

svg_to_ico("doc/assets/logo_small_light.svg", f"{BASE_OUR_PATH}/assets/favicon.ico", sizes=((64, 64),))