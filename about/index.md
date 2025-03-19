# About


# 👨‍💻 fengchen

**折腾爱好者 | 思考探索 | 保持好奇**

**C&#43;&#43;** 学习者

## 💡 生活即探索，折腾不停歇！🚀

## 🎲 小游戏：猜数字
### 规则
1. 系统会随机选择一个 1 到 10 之间的数字。
2. 你需要通过点击按钮猜出这个数字。
3. 每次点击后，系统会告诉你猜得太大还是太小。
4. 你可以反复点击按钮直到猜对为止。

---

### 游戏开始：
&lt;details&gt;
  &lt;summary&gt;点击开始游戏&lt;/summary&gt;

  **请选择一个 1-10 之间的数字。**

  &lt;div id=&#34;buttonsContainer&#34;&gt;&lt;/div&gt;
  &lt;div id=&#34;feedback&#34;&gt;&lt;/div&gt;
  &lt;div id=&#34;reward&#34; style=&#34;display:none;&#34;&gt;
    &lt;div id=&#34;eggMessage&#34;&gt;&lt;/div&gt;
  &lt;/div&gt;

  &lt;script&gt;
    let secretNumber = Math.floor(Math.random() * 10) &#43; 1;
    let attempts = 0;

    // 动态生成数字按钮
    function generateButtons() {
      let container = document.getElementById(&#39;buttonsContainer&#39;);
      container.innerHTML = &#39;&#39;;  // 清空之前的按钮
      for (let i = 1; i &lt;= 10; i&#43;&#43;) {
        let button = document.createElement(&#39;button&#39;);
        button.innerText = i;
        button.onclick = () =&gt; guessNumber(i);
        container.appendChild(button);
      }
    }

    function guessNumber(userGuess) {
      attempts&#43;&#43;;

      // 清空之前的反馈
      document.getElementById(&#39;feedback&#39;).innerHTML = &#39;&#39;;

      if (userGuess === secretNumber) {
        showReward();
        disableButtons();
      } else if (userGuess &lt; secretNumber) {
        document.getElementById(&#39;feedback&#39;).innerHTML = `😢 你的猜测 ${userGuess} 太小了，再试试！`;
      } else {
        document.getElementById(&#39;feedback&#39;).innerHTML = `😢 你的猜测 ${userGuess} 太大了，再试试！`;
      }
    }

    function showReward() {
      let eggMessage = &#39;&#39;;
      if (attempts &lt;= 3) {
        eggMessage = `🎉 哇！你真厉害，猜了 ${attempts} 次就猜对了！🎉&lt;br&gt;&lt;strong&gt;彩蛋&lt;/strong&gt;：你是数字猜测天才！`;
      } else if (attempts &lt;= 6) {
        eggMessage = `🎉 很棒！你用了 ${attempts} 次才猜对！&lt;br&gt;&lt;strong&gt;彩蛋&lt;/strong&gt;：你有很好的直觉！`;
      } else {
        eggMessage = `🎉 恭喜你终于猜对了！&lt;br&gt;&lt;strong&gt;彩蛋&lt;/strong&gt;：虽然用了 ${attempts} 次，但你还是完成了任务！`;
      }

      document.getElementById(&#39;eggMessage&#39;).innerHTML = eggMessage;
      document.getElementById(&#39;reward&#39;).style.display = &#39;block&#39;;
    }

    function disableButtons() {
      let buttons = document.querySelectorAll(&#39;#buttonsContainer button&#39;);
      buttons.forEach(button =&gt; {
        button.disabled = true;  // 禁用所有按钮
      });
    }
    // 初始化游戏
    generateButtons();
  &lt;/script&gt;

&lt;/details&gt;

---

> 作者:   
> URL: http://fengchen321.github.io/about/  

