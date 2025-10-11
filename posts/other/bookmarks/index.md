# 📚 我的书签收藏

后续整理 ...

&lt;div id=&#34;bookmarks-container&#34; style=&#34;margin-top: 20px;&#34;&gt;
  &lt;div style=&#34;text-align: center; padding: 40px; color: #666;&#34;&gt;
    &lt;p&gt;书签加载中...&lt;/p&gt;
  &lt;/div&gt;
&lt;/div&gt;

&lt;script&gt;
function renderBookmarks(data) {
  const container = document.getElementById(&#39;bookmarks-container&#39;);
  
  if (!data || Object.keys(data).length === 0) {
    container.innerHTML = &#39;&lt;div style=&#34;text-align: center; padding: 40px; color: #999;&#34;&gt;没有书签数据&lt;/div&gt;&#39;;
    return;
  }

  let html = &#39;&lt;div class=&#34;bookmarks-layout&#34;&gt;&#39;;
  
  Object.entries(data).forEach(([folderName, folderItems]) =&gt; {
    html &#43;= `
      &lt;div class=&#34;main-folder&#34;&gt;
        &lt;div class=&#34;folder-header&#34;&gt;
          &lt;svg class=&#34;folder-icon&#34; viewBox=&#34;0 0 24 24&#34; fill=&#34;currentColor&#34;&gt;
            &lt;path d=&#34;M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z&#34;/&gt;
          &lt;/svg&gt;
          ${folderName}
        &lt;/div&gt;
        &lt;div class=&#34;folder-content&#34;&gt;
    `;
    
    if (Array.isArray(folderItems)) {
      folderItems.forEach(item =&gt; {
        if (typeof item === &#39;object&#39; &amp;&amp; !item.href) {
          // 子文件夹
          Object.entries(item).forEach(([subFolderName, subItems]) =&gt; {
            html &#43;= `
              &lt;div class=&#34;sub-folder&#34;&gt;
                &lt;div class=&#34;sub-folder-header&#34;&gt;
                  &lt;svg class=&#34;sub-folder-icon&#34; viewBox=&#34;0 0 24 24&#34; fill=&#34;currentColor&#34;&gt;
                    &lt;path d=&#34;M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z&#34;/&gt;
                  &lt;/svg&gt;
                  ${subFolderName}
                &lt;/div&gt;
                &lt;div class=&#34;links-grid&#34;&gt;
            `;
            
            if (Array.isArray(subItems)) {
              subItems.forEach(subItem =&gt; {
                if (subItem.href) {
                  // 使用实际的网站favicon
                  const domain = new URL(subItem.href).hostname;
                  const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&amp;sz=32`;
                  
                  html &#43;= `
                    &lt;a href=&#34;${subItem.href}&#34; target=&#34;_blank&#34; class=&#34;bookmark-link&#34; title=&#34;${subItem.name}&#34;&gt;
                      &lt;img src=&#34;${faviconUrl}&#34; alt=&#34;&#34; class=&#34;favicon&#34; onerror=&#34;this.style.display=&#39;none&#39;; this.nextElementSibling.style.display=&#39;flex&#39;;&#34;&gt;
                      &lt;svg class=&#34;fallback-icon&#34; viewBox=&#34;0 0 24 24&#34; fill=&#34;currentColor&#34; style=&#34;display: none;&#34;&gt;
                        &lt;path d=&#34;M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z&#34;/&gt;
                      &lt;/svg&gt;
                      &lt;span class=&#34;link-text&#34;&gt;${subItem.name}&lt;/span&gt;
                    &lt;/a&gt;
                  `;
                }
              });
            }
            
            html &#43;= &#39;&lt;/div&gt;&lt;/div&gt;&#39;;
          });
        } 
        else if (item.href) {
          // 直接书签 - 使用网站favicon
          const domain = new URL(item.href).hostname;
          const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&amp;sz=32`;
          
          html &#43;= `
            &lt;a href=&#34;${item.href}&#34; target=&#34;_blank&#34; class=&#34;bookmark-link&#34; title=&#34;${item.name}&#34;&gt;
              &lt;img src=&#34;${faviconUrl}&#34; alt=&#34;&#34; class=&#34;favicon&#34; onerror=&#34;this.style.display=&#39;none&#39;; this.nextElementSibling.style.display=&#39;flex&#39;;&#34;&gt;
              &lt;svg class=&#34;fallback-icon&#34; viewBox=&#34;0 0 24 24&#34; fill=&#34;currentColor&#34; style=&#34;display: none;&#34;&gt;
                &lt;path d=&#34;M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z&#34;/&gt;
              &lt;/svg&gt;
              &lt;span class=&#34;link-text&#34;&gt;${item.name}&lt;/span&gt;
            &lt;/a&gt;
          `;
        }
      });
    }
    
    html &#43;= &#39;&lt;/div&gt;&lt;/div&gt;&#39;;
  });
  
  html &#43;= &#39;&lt;/div&gt;&#39;;
  container.innerHTML = html;
}

// 初始化加载
document.addEventListener(&#39;DOMContentLoaded&#39;, () =&gt; {
  fetch(&#39;/bookmarks.json&#39;)
    .then(response =&gt; response.json())
    .then(data =&gt; renderBookmarks(data))
    .catch(error =&gt; {
      console.error(&#39;加载书签失败:&#39;, error);
      document.getElementById(&#39;bookmarks-container&#39;).innerHTML = 
        &#39;&lt;div style=&#34;text-align: center; padding: 40px; color: #e74c3c;&#34;&gt;加载书签失败，请刷新页面重试&lt;/div&gt;&#39;;
    });
});

// 添加CSS样式
const style = document.createElement(&#39;style&#39;);
style.textContent = `
  .bookmarks-layout {
    display: grid;
    grid-template-columns: 1fr);
    gap: 30px;
    margin-top: 20px;
    max-width: 1400px;
    margin-left: auto;
    margin-right: auto;
  }

  .main-folder {
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    border: 1px solid #f0f0f0;
    transition: transform 0.3s, box-shadow 0.3s;
  }

  .main-folder:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
  }

  .folder-header {
    font-weight: 700;
    margin-bottom: 25px;
    padding-bottom: 20px;
    border-bottom: 3px solid #667eea;
    color: #2d3748;
    display: flex;
    align-items: center;
    font-size: 22px;
    letter-spacing: -0.5px;
  }

  .folder-icon {
    width: 24px;
    height: 24px;
    margin-right: 15px;
    color: #667eea;
  }

  .folder-content {
    display: grid;
    grid-template-columns: repeat(2, 1fr);  /* 子文件夹2列 */
    gap: 20px;
 }

  .sub-folder {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    transition: transform 0.2s;
  }

  .sub-folder:hover {
    transform: translateX(5px);
  }

  .sub-folder-header {
    font-weight: 600;
    margin-bottom: 16px;
    color: #4a5568;
    display: flex;
    align-items: center;
    font-size: 16px;
    letter-spacing: -0.3px;
  }

  .sub-folder-icon {
    width: 18px;
    height: 18px;
    margin-right: 10px;
    color: #a78bfa;
  }

  .links-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);  /* 控制每个子文件夹内的列数 */
    gap: 8px;  /* 控制链接之间的间距 */
  }

  .bookmark-link {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 10px 6px;
    background: white;
    border-radius: 12px;
    text-decoration: none;
    color: #4a5568;
    font-size: 12px;  /* 控制文字大小 */
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
    min-height: 70px;  /* 控制链接卡片高度 */
    text-align: center;
    line-height: 1.4;
    position: relative;
    overflow: hidden;
  }

  .bookmark-link::before {
    content: &#39;&#39;;
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    transition: left 0.5s;
  }

  .bookmark-link:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
  }

  .bookmark-link:hover::before {
    left: 100%;
  }

  .bookmark-link:hover .link-text {
    color: white;
  }

  .favicon {
    width: 18px;
    height: 18px;
    margin-bottom: 6px;
    border-radius: 4px;
    transition: transform 0.3s ease;
  }

  .bookmark-link:hover .favicon {
    transform: scale(1.2);
  }

  .fallback-icon {
    width: 16px;
    height: 16px;
    margin-bottom: 6px;
    color: #f59e0b;
    transition: transform 0.3s ease;
  }

  .bookmark-link:hover .fallback-icon {
    color: white;
    transform: scale(1.2);
  }

  .link-text {
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    word-break: break-word;
    font-weight: 500;
    transition: color 0.3s ease;
  }

  /* 响应式设计 */
  @media (max-width: 1200px) {
    .bookmarks-layout {
      grid-template-columns: 1fr;
      gap: 25px;
      max-width: 800px;
    }
    
    .links-grid {
      grid-template-columns: repeat(5, 1fr);
    }
  }

  @media (max-width: 768px) {
    .bookmarks-layout {
      gap: 20px;
    }
    
    .main-folder {
      padding: 20px;
    }
    
    .links-grid {
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
    }
    
    .bookmark-link {
      min-height: 60px;
      font-size: 11px;
      padding: 8px 4px;
    }
    
    .folder-header {
      font-size: 20px;
    }
  }

  @media (max-width: 480px) {
    .links-grid {
      grid-template-columns: repeat(3, 1fr);
    }
    
    .bookmark-link {
      min-height: 55px;
    }
  }

  @media (max-width: 360px) {
    .links-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
`;
document.head.appendChild(style);
&lt;/script&gt;


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/other/bookmarks/  

