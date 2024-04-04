import pdoc
import re
from pdoc.cli import recursive_write_files, _generate_lunr_search
from typing import Optional

FLATTEN_DOC = True
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

pdoc.cli.args.output_dir = "build"
pdoc.tpl_lookup.directories.insert(0, './doc/templates')
template_config = {'lunr_search': {'index_docstrings': True},
                   'list_class_variables_in_index': True,
                   'show_source_code': False,
                   'show_inherited_members': True}
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

def flatten_module(module: pdoc.Module, ref_module: Optional[pdoc.Module] = None, blacklist: Optional[set[str]] = None):
    """
    Make all the objects in the submodules part of the main module
    """
    if blacklist is None:
        blacklist = set()
    # Inserting the objects in the main module
    if ref_module is not None:
        objects = {name: item for name, item in module.doc.items() if not isinstance(item, pdoc.Module)}
        for obj in objects.values():
            obj.module = ref_module
            if hasattr(obj, "doc") and len(obj.doc) > 0:
                for doc in obj.doc.values():
                    doc.module = ref_module
        ref_module.module.doc.update(objects)
    # Flattening sub-modules
    sub_modules = [name for name, m in module.module.doc.items() if isinstance(m, pdoc.Module)]
    if ref_module is None:
        ref_module = module
    for name in sub_modules:
        if name not in blacklist:
            flatten_module(module.doc[name], ref_module, blacklist)

    # Removing the sub-modules from the main module
    module.doc = {name: item for name, item in module.doc.items() if not isinstance(item, pdoc.Module) or name in blacklist}
    return module

if FLATTEN_DOC:
    # Data
    flatten_module(mod.doc["data"])
    # Indicators
    flatten_module(mod.doc["indicators"])
    # Engine
    flatten_module(mod.doc["engine"])
    # Serve
    flatten_module(mod.doc["serve"], blacklist={"renderers"})
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
