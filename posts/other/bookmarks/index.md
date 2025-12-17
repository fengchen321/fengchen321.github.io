# 我的书签收藏


&lt;div id=&#34;bookmarks-container&#34;&gt;
  &lt;div class=&#34;bm-loading&#34;&gt;加载中...&lt;/div&gt;
&lt;/div&gt;

&lt;style&gt;
.bm-loading { text-align: center; padding: 40px; color: var(--secondary-color, #666); }
.bm-error { text-align: center; padding: 40px; color: #e74c3c; }

.bm-folder {
  margin-bottom: 2rem;
  background: var(--card-background, #fff);
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
[data-theme=&#34;dark&#34;] .bm-folder { background: #1e1e1e; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }

.bm-folder-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--primary-color, #667eea);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.bm-folder-title::before { content: &#34;📁&#34;; }

.bm-subfolder {
  margin: 1rem 0;
  padding: 1rem;
  background: var(--background-color, #f8f9fa);
  border-radius: 8px;
}
[data-theme=&#34;dark&#34;] .bm-subfolder { background: #2a2a2a; }

.bm-subfolder-title {
  font-size: 1rem;
  font-weight: 500;
  margin-bottom: 0.75rem;
  color: var(--secondary-color, #555);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.bm-subfolder-title::before { content: &#34;📂&#34;; font-size: 0.9em; }

.bm-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 0.5rem;
}

.bm-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: var(--card-background, #fff);
  border: 1px solid var(--border-color, #e5e5e5);
  border-radius: 6px;
  text-decoration: none;
  color: var(--text-color, #333);
  font-size: 0.8rem;
  transition: all 0.2s;
  overflow: hidden;
}
[data-theme=&#34;dark&#34;] .bm-link { background: #333; border-color: #444; color: #ddd; }

.bm-link:hover {
  border-color: var(--primary-color, #667eea);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102,126,234,0.2);
}

.bm-link img {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.bm-link span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 768px) {
  .bm-grid { grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); }
  .bm-link { padding: 0.4rem 0.5rem; font-size: 0.75rem; }
}
&lt;/style&gt;

&lt;script&gt;
(function() {
  function getFavicon(url) {
    try {
      const domain = new URL(url).hostname;
      return `https://www.google.com/s2/favicons?domain=${domain}&amp;sz=32`;
    } catch { return &#39;&#39;; }
  }

  function renderLink(item) {
    if (!item.href) return &#39;&#39;;
    const favicon = getFavicon(item.href);
    return `&lt;a href=&#34;${item.href}&#34; target=&#34;_blank&#34; rel=&#34;noopener&#34; class=&#34;bm-link&#34; title=&#34;${item.name}&#34;&gt;
      ${favicon ? `&lt;img src=&#34;${favicon}&#34; alt=&#34;&#34; loading=&#34;lazy&#34;&gt;` : &#39;&#39;}
      &lt;span&gt;${item.name}&lt;/span&gt;
    &lt;/a&gt;`;
  }

  function renderFolder(name, items) {
    let linksHtml = &#39;&#39;;
    let subfoldersHtml = &#39;&#39;;

    items.forEach(item =&gt; {
      if (item.href) {
        linksHtml &#43;= renderLink(item);
      } else if (typeof item === &#39;object&#39;) {
        Object.entries(item).forEach(([subName, subItems]) =&gt; {
          if (Array.isArray(subItems)) {
            subfoldersHtml &#43;= `
              &lt;div class=&#34;bm-subfolder&#34;&gt;
                &lt;div class=&#34;bm-subfolder-title&#34;&gt;${subName}&lt;/div&gt;
                &lt;div class=&#34;bm-grid&#34;&gt;${subItems.map(renderLink).join(&#39;&#39;)}&lt;/div&gt;
              &lt;/div&gt;`;
          }
        });
      }
    });

    return `&lt;div class=&#34;bm-folder&#34;&gt;
      &lt;div class=&#34;bm-folder-title&#34;&gt;${name}&lt;/div&gt;
      ${linksHtml ? `&lt;div class=&#34;bm-grid&#34;&gt;${linksHtml}&lt;/div&gt;` : &#39;&#39;}
      ${subfoldersHtml}
    &lt;/div&gt;`;
  }

  function render(data) {
    const container = document.getElementById(&#39;bookmarks-container&#39;);
    if (!data || !Object.keys(data).length) {
      container.innerHTML = &#39;&lt;div class=&#34;bm-error&#34;&gt;没有书签数据&lt;/div&gt;&#39;;
      return;
    }
    container.innerHTML = Object.entries(data).map(([name, items]) =&gt;
      Array.isArray(items) ? renderFolder(name, items) : &#39;&#39;
    ).join(&#39;&#39;);
  }

  fetch(&#39;/bookmarks.json&#39;)
    .then(r =&gt; r.json())
    .then(render)
    .catch(() =&gt; {
      document.getElementById(&#39;bookmarks-container&#39;).innerHTML =
        &#39;&lt;div class=&#34;bm-error&#34;&gt;加载失败，请刷新重试&lt;/div&gt;&#39;;
    });
})();
&lt;/script&gt;


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/other/bookmarks/  

