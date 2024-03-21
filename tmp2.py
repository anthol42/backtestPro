from weasyprint import HTML, CSS
import logging

logger = logging.getLogger('weasyprint')
logger.addHandler(logging.FileHandler('.cache/weasyprint.log'))


html = HTML("tmp.html")
css = CSS("tmp.css")

html.write_pdf(
    'tmp.pdf', stylesheets=[css])