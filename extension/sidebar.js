(function () {

  const addArticle = () => {
    chrome.runtime.sendMessage({action: 'ADD_ARTICLE'}, (data) => {
      window.console.log(data);
      $.post('http://localhost:8000/popnews/save', JSON.stringify({
        'url': data.url,
        'title': data.title,
        'icon': data.icon
      }), updateArticleList, dataType = 'json');
    });
  }

  const updateArticleList = () => {
    // Refresh existing bookmarks.
    $.get('http://localhost:8000/popnews/stats', (data) => {
      $('#sidebar-content').html(''); // Clear existing fields.
      const allArticles = data.articles;
      $.each(allArticles, (idx, article) => renderArticle(article.article__icon, article.article__title));
    });
  }

  const renderArticle = (iconUrl='', headline='') => {
    $('#sidebar-content').prepend(`
      <div class="sidebar-article">
        <div class="article-icon">
          <img src="${iconUrl}" alt="${headline}" />
        </div>
        <div class="article-content">
          ${headline}
        </div>
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
            <div id="pop-logo-text">
              Pop
            </div>
          </div>
        </div>
        <div id="sidebar-content"></div>
        <div id="sidebar-footer"></div>
      </div>
    `);

    $('#pop-main-logo').on('click', addArticle);

    updateArticleList();
  }

  $('#pop-sidebar-container').toggleClass('active');
  $('#pop-main-container').toggleClass('active');

})();