<!doctype html>
<html lang="${html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
  <link rel="icon" type="image/x-icon" href="${absolute_path}/assets/favicon.ico">
  <link href='${absolute_path}/assets/bootstrap.min.css' rel='stylesheet', type='text/css'>
  <link href="${absolute_path}/assets/style.css" rel="stylesheet" type="text/css">
  <link href="${absolute_path}/assets/python_code.css" rel="stylesheet" type="text/css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" crossorigin="anonymous" />
  <script src="${absolute_path}/assets/bootstrap.min.js"></script>
  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
  <title>${page_title}</title>
  <meta name="description" content="${page_desc}" />
</head>
<body>
  <%include file="head.mako"/>
    ${page_content}
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

      // Function to handle scroll synchronization
      function syncScroll() {
        const fromTop = window.scrollY;
        sectionLinks.forEach(link => {
          const section = document.querySelector(link.hash);
          if (section.offsetTop <= fromTop + 100 && section.offsetTop + section.offsetHeight > fromTop + 100) {
            link.classList.add('active-section');
          } else {
            link.classList.remove('active-section');
          }
        });
      }

      // Add scroll event listener to sync scroll position
      window.addEventListener('scroll', syncScroll);

      // Add click event listeners to section links to smoothly scroll to the respective section
      sectionLinks.forEach(link => {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          const targetId = this.hash;
          const targetSection = document.querySelector(targetId);
          window.scrollTo({
            top: targetSection.offsetTop - 100, // Adjust as needed for any fixed headers
            behavior: 'smooth'
          });
        });
      });
    });
  </script>
</html>