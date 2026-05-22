document.addEventListener('DOMContentLoaded', () => {
  const getOffset = () => {
    const header = document.getElementById('header-desktop');
    const headerHeight = header ? header.offsetHeight : 0;
    return headerHeight + 32;
  };

  const scrollToHash = (hash, smooth = true) => {
    if (!hash || hash === '#') return;
    const id = decodeURIComponent(hash.slice(1));
    const target =
      document.getElementById(id) ||
      document.querySelector(`[id="${CSS.escape(id)}"]`);

    if (!target) return;

    const top = window.scrollY + target.getBoundingClientRect().top - getOffset();
    window.scrollTo({
      top: Math.max(top, 0),
      behavior: smooth ? 'smooth' : 'auto'
    });
  };

  const bindHashLinks = (root) => {
    root?.querySelectorAll('a[href^="#"]').forEach((link) => {
      link.addEventListener('click', (event) => {
        const href = link.getAttribute('href');
        if (!href || href === '#') return;
        event.preventDefault();
        history.replaceState(null, '', href);
        scrollToHash(href, true);
      });
    });
  };

  bindHashLinks(document.getElementById('toc-auto'));
  bindHashLinks(document.getElementById('toc-static'));
  bindHashLinks(document.getElementById('content'));

  if (window.location.hash) {
    requestAnimationFrame(() => {
      setTimeout(() => scrollToHash(window.location.hash, false), 60);
    });
  }

  window.addEventListener('hashchange', () => scrollToHash(window.location.hash, false));
});
