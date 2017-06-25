(function () {

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

    $.get('http://localhost:8000/popnews/stats', (data) => {
      const allArticles = data.articles;
      $.each(allArticles, (idx, article) => renderArticle('http://ashleyhlai.com/favicon.ico', article.article__url));
    });
  }

  $('#pop-sidebar-container').toggleClass('active');
  $('#pop-main-container').toggleClass('active');

})();