(function () {

  const biasFactor = 7;
  let runningBias = 0.5;

  const addArticle = () => {
    chrome.runtime.sendMessage({action: 'ADD_ARTICLE'}, (data) => {
      $.post('http://dev1.b12.io:8000/popnews/save', JSON.stringify({
        'url': data.url,
        'title': data.title,
        'icon': data.icon
      }), updateArticleList, dataType = 'json');
    });
  }

  const updateArticleList = () => {
    // Refresh existing bookmarks.
    $.get('http://dev1.b12.io:8000/popnews/stats', (data) => {
      window.console.log(data);
      $('#sidebar-content').html(''); // Clear existing fields.
      const allArticles = data.articles;
      const biasSum = 0;
      const biasCount = 0;
      $.each(allArticles, (idx, article) => {
        renderArticle(article.article__icon, article.article__title, article.article__bias);
        if (article.article__bias) {
          biasSum += article.article__bias;
          biasCount += 1;
        }
      });
      // Update running bias.
      runningBias = (biasCount === 0) ? runningBias : (biasSum / biasCount);
      renderBiasMeter();
    });
  }

  const calcBiasOffset = (bias) => {
    bias = (bias < 0 || bias > 1) ? runningBias : bias;
    const freeSpace = 100 - (100 / biasFactor);
    const roundToFraction = Math.round(bias * (biasFactor - 1)) / (biasFactor - 1);
    const percent = roundToFraction * freeSpace;
    return percent;
  }

  const renderBiasMeter = () => {
    const biasOffset = calcBiasOffset(runningBias);
    $('#pop-bias-meter').append(`
      <div class="sidebar-article-bias">
        <div class="article-bias-rating" style="left:${biasOffset}%;"></div>
      </div>
    `);
  }

  const renderArticle = (iconUrl='', headline='Untitled', bias=0.5) => {
    const biasOffset = calcBiasOffset(bias);
    $('#sidebar-content').prepend(`
      <div class="sidebar-article">
        <div class="article-icon">
          <img src="${iconUrl}" alt="${headline}" />
        </div>
        <div class="article-content">
          ${headline}
        </div>
      </div>
      <div class="sidebar-article-bias">
        <div class="article-bias-rating" style="left:${biasOffset}%;"></div>
      </div>
    `);
  }

  // Initialize sidebar.
  if (!$('#pop-main-container').length) {
    const $innerContent = $('body').html();
    $('body').html(`
      <div id="pop-main-container">
        ${$innerContent}
      </div>
      <div id="pop-sidebar-container">
        <div id="sidebar-header">
          <div id="pop-main-logo">
            <div id="pop-logo-text"></div>
          </div>
        </div>
        <div id="sidebar-content"></div>
        <div id="sidebar-footer">
          <div id="pop-bias-meter"></div>
          <div id="pop-bias-scale">
            <div id="pop-bias-fl" class="pop-bias" style="text-align:left;">Liberal</div>
            <div id="pop-bias-md" class="pop-bias" style="text-align:center;">Neutral</div>
            <div id="pop-bias-fr" class="pop-bias" style="text-align:right;">Conservative</div>
          </div>
        </div>
      </div>
    `);

    $('#pop-main-logo').on('click', addArticle);

    updateArticleList();
  }

  $('#pop-sidebar-container').toggleClass('active');
  $('#pop-main-container').toggleClass('active');

})();