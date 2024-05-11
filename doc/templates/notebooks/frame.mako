<!doctype html>
<html lang="${html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
  <link rel="icon" type="image/x-icon" href="${absolute_path}/assets/favicon.ico?">
  <link href='${absolute_path}/assets/bootstrap.min.css' rel='stylesheet', type='text/css'>
  <link href="${absolute_path}/assets/style.css" rel="stylesheet" type="text/css">
  <link href="${absolute_path}/assets/notebook.css" rel="stylesheet" type="text/css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" crossorigin="anonymous" />
  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
  <script src="${absolute_path}/assets/bootstrap.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
  <title>${page_title}</title>
  <meta name="description" content="${page_desc}" />
  ${rendering_scripts}
</head>
<body>
  <%include file="head.mako"/>
    <div class="row" style="margin-right: 0;">
    <!-- Table of Contents Section -->
    <div id="tableOfContents" class="col-md-3" style="width: 300px">
    <nav id="toc">
        <ul class="nav flex-column">
            <li class="nav-item nav-item-dynamic">
                <a class="nav-link"  style="${'color: rgb(34, 34, 34)' if page_id == 'home' else 'color: rgb(137, 137, 137)'}" href="${absolute_path}/tutorials">Home</a>
            </li>
            % for filename in available_files:
                <li class="nav-item nav-item-dynamic">
                    <a class="nav-link" style="${'color: rgb(34, 34, 34)' if page_id == filename else 'color: rgb(137, 137, 137)'}" href="${absolute_path}/tutorials/${filename}.html">${filename.replace("_", " ").title()}</a>
                </li>
            % endfor
        </ul>
    </nav>
    </div>
    <div class="toc-doc">
        <div class="notebook-container">
            ${notebook_content}
        </div>
    </div>
    </div>
  <div class="divider"></div>
  <footer id="footer">
    <%include file="footer.mako"/>
</footer>
</body>
  <script>
function copyToClipboard(button) {
      var preElement = button.parentNode.previousElementSibling;
      var codeElement = preElement.querySelector('code');
      var codeText = codeElement.textContent || codeElement.innerText;

      navigator.clipboard.writeText(codeText).then(function() {
        button.innerHTML = '<i class="far fa-check-circle"></i>';
        button.classList.add('copied');

        setTimeout(function() {
          button.innerHTML = '<i class="far fa-copy"></i>';
          button.classList.remove('copied');
        }, 3000);
      }).catch(function(error) {
        console.error('Failed to copy: ', error);
      });
    }
    function copyToClipboardText(button, id) {
    const codeText = codeTexts[id];
    navigator.clipboard.writeText(codeText).then(function() {
        button.innerHTML = '<i class="far fa-check-circle"></i>';
        button.classList.add('copied');

        setTimeout(function() {
          button.innerHTML = '<i class="far fa-copy"></i>';
          button.classList.remove('copied');
        }, 3000);
      }).catch(function(error) {
        console.error('Failed to copy: ', error);
      });
    }
        document.addEventListener('DOMContentLoaded', function() {
      // Get all section links
      const sectionLinks = document.querySelectorAll('#toc a');
    });
  </script>
</html>