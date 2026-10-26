// The page's behaviour: copy BibTeX, scroll to top, and the contents bar that highlights the section being read.
// Adapted from the Academic Project Page Template.

function copyBibTeX() {
  const code = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const label = button.querySelector('.copy-text');
  const done = () => {
    button.classList.add('copied');
    label.textContent = 'Copied';
    setTimeout(() => { button.classList.remove('copied'); label.textContent = 'Copy'; }, 2000);
  };
  const fallback = () => {
    const area = document.createElement('textarea');
    area.value = code.textContent;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    document.body.removeChild(area);
    done();
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(code.textContent).then(done, fallback);
  } else {
    fallback();
  }
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  const button = document.querySelector('.scroll-to-top');
  if (button) button.classList.toggle('visible', window.pageYOffset > 300);
});

function setupToc() {
  const links = [...document.querySelectorAll('.toc a')];
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  if (!sections.length) return;
  let last = null;
  function update() {
    // the last section whose top has passed a line a third of the way down the window
    const line = window.innerHeight / 3;
    let current = null;
    sections.forEach(s => { if (s.getBoundingClientRect().top <= line) current = s; });
    if (current === last) return;
    last = current;
    links.forEach(a => {
      const on = current !== null && a.getAttribute('href') === '#' + current.id;
      a.classList.toggle('active', on);
      const bar = a.parentElement;  // on a narrow bar, bring the highlighted entry into view
      if (on && (a.offsetLeft < bar.scrollLeft || a.offsetLeft + a.offsetWidth > bar.scrollLeft + bar.clientWidth)) {
        bar.scrollTo({ left: a.offsetLeft - 16 });
      }
    });
  }
  let pending = false;
  window.addEventListener('scroll', () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; update(); });
  }, { passive: true });
  update();
}

document.addEventListener('DOMContentLoaded', setupToc);
